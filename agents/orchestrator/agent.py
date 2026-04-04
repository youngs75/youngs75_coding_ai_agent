"""Orchestrator 에이전트.

사용자 입력을 분석하여 적합한 하위 에이전트로 라우팅하고,
A2A 프로토콜로 위임한 결과를 반환한다.

흐름:
    START → [CLASSIFY] → [DELEGATE] → [RESPOND] → END

사용 예:
    config = OrchestratorConfig(agent_endpoints=[...])
    agent = OrchestratorAgent(config=config)
    result = await agent.graph.ainvoke(
        {"messages": [HumanMessage("오늘 AI 뉴스 알려줘")]},
        config={"configurable": config.to_langgraph_configurable()},
    )
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, ClassVar

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from .config import OrchestratorConfig
from .schemas import OrchestratorState

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM_PROMPT = """\
당신은 사용자의 요청을 분석하여 가장 적합한 에이전트를 선택하는 라우터입니다.

사용 가능한 에이전트:
{agent_descriptions}

규칙:
1. 사용자의 요청 의도를 파악하고, 위 목록에서 가장 적합한 에이전트 이름을 정확히 하나만 선택하세요.
2. 에이전트 이름만 출력하세요. 다른 텍스트는 포함하지 마세요.
3. 어떤 에이전트에도 맞지 않으면 "none"을 출력하세요.
"""


async def classify(state: OrchestratorState, config: RunnableConfig) -> dict:
    """사용자 입력을 분석하여 적합한 에이전트를 선택한다."""
    oc = OrchestratorConfig.from_runnable_config(config)
    llm = oc.get_model("default")

    agent_descriptions = oc.get_agent_descriptions()
    system_prompt = CLASSIFY_SYSTEM_PROMPT.format(agent_descriptions=agent_descriptions)

    # 마지막 사용자 메시지 추출
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )

    selected = response.content.strip().lower()

    # 등록된 에이전트 이름과 매칭
    agent_names = [ep.name.lower() for ep in oc.agent_endpoints]
    if selected not in agent_names:
        # 부분 매칭 시도
        for name in agent_names:
            if name in selected or selected in name:
                selected = name
                break
        else:
            selected = "none"

    logger.info(f"라우팅 결정: '{user_message[:50]}...' → {selected}")
    return {"selected_agent": selected}


async def delegate(state: OrchestratorState, config: RunnableConfig) -> dict:
    """선택된 에이전트에 A2A 프로토콜로 요청을 위임한다."""
    oc = OrchestratorConfig.from_runnable_config(config)
    selected = state.get("selected_agent", "none")

    if selected == "none":
        return {
            "agent_response": "죄송합니다. 현재 등록된 에이전트 중 적합한 것을 찾지 못했습니다. 질문을 다시 한번 구체적으로 말씀해 주세요."
        }

    url = oc.get_endpoint_url(selected)
    if not url:
        return {
            "agent_response": f"에이전트 '{selected}'의 엔드포인트를 찾을 수 없습니다."
        }

    # 마지막 사용자 메시지 추출
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    try:
        # A2A 프로토콜로 요청
        from a2a.client import A2AClient
        from a2a.client.helpers import create_text_message_object
        from a2a.types import MessageSendParams, SendMessageRequest

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as hc:
            client = A2AClient(httpx_client=hc, url=url)
            msg = create_text_message_object(content=user_message)
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(message=msg),
            )
            response = await client.send_message(request)

        # 응답에서 텍스트 추출
        result = response.root
        if hasattr(result, "result"):
            obj = result.result
            if hasattr(obj, "artifacts") and obj.artifacts:
                for artifact in obj.artifacts:
                    for part in artifact.parts or []:
                        root = getattr(part, "root", part)
                        if hasattr(root, "text") and root.text and len(root.text) > 5:
                            return {"agent_response": root.text}
            if hasattr(obj, "status"):
                return {"agent_response": f"[에이전트 작업 상태: {obj.status}]"}

        return {"agent_response": "[에이전트 응답 파싱 실패]"}

    except Exception as e:
        logger.error(f"에이전트 '{selected}' 호출 실패: {e}")
        return {
            "agent_response": f"에이전트 '{selected}' 호출 중 오류가 발생했습니다: {e}"
        }


async def respond(state: OrchestratorState, config: RunnableConfig) -> dict:
    """에이전트 응답을 최종 메시지로 포맷팅한다."""
    selected = state.get("selected_agent", "unknown")
    response = state.get("agent_response", "")

    if selected and selected != "none":
        content = f"[{selected}] {response}"
    else:
        content = response

    return {"messages": [AIMessage(content=content)]}


class OrchestratorAgent(BaseGraphAgent):
    """사용자 요청을 분석하고 적합한 에이전트로 라우팅하는 오케스트레이터."""

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLASSIFY": "classify",
        "DELEGATE": "delegate",
        "RESPOND": "respond",
    }

    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self._orch_config = config or OrchestratorConfig()
        super().__init__(
            config=self._orch_config,
            state_schema=OrchestratorState,
            agent_name="OrchestratorAgent",
            **kwargs,
        )

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("CLASSIFY"), classify)
        graph.add_node(self.get_node_name("DELEGATE"), delegate)
        graph.add_node(self.get_node_name("RESPOND"), respond)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLASSIFY"))
        graph.add_edge(self.get_node_name("CLASSIFY"), self.get_node_name("DELEGATE"))
        graph.add_edge(self.get_node_name("DELEGATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("RESPOND"), END)
