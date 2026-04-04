"""Orchestrator 에이전트.

사용자 입력을 분석하여 적합한 하위 에이전트로 라우팅하고,
A2A 프로토콜로 위임한 결과를 반환한다.

흐름:
    START → [CLASSIFY] ─→ [DELEGATE]    → [RESPOND] → END
                        └→ [COORDINATE] ↗

    복합 작업 감지 시 coordinator 모드로 병렬 워커 오케스트레이션.

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
from youngs75_a2a.core.context_manager import ContextManager
from youngs75_a2a.core.subagents.registry import SubAgentRegistry
from .config import OrchestratorConfig
from .coordinator import CoordinatorMode
from .schemas import OrchestratorState

# 오케스트레이터용 컨텍스트 매니저 (서브에이전트 호출 시 히스토리 필터링)
_orchestrator_context_manager = ContextManager()

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM_PROMPT = """\
당신은 사용자의 요청을 분석하여 가장 적합한 에이전트를 선택하는 라우터입니다.

사용 가능한 에이전트:
{agent_descriptions}

규칙:
1. 사용자의 요청 의도를 파악하고, 위 목록에서 가장 적합한 에이전트 이름을 정확히 하나만 선택하세요.
2. 에이전트 이름만 출력하세요. 다른 텍스트는 포함하지 마세요.
3. 어떤 에이전트에도 맞지 않으면 "none"을 출력하세요.
4. 요청이 여러 에이전트의 협업이 필요한 복합 작업이면 "coordinate"를 출력하세요.
   복합 작업 예시: "코드를 작성하고 리뷰해줘", "조사해서 보고서를 작성해줘" 등
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

    # 코디네이터 모드 감지
    if selected == "coordinate" or selected == "__coordinator__":
        logger.info(f"라우팅 결정: '{user_message[:50]}...' → __coordinator__")
        return {"selected_agent": "__coordinator__"}

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


async def _invoke_local_agent(agent_name: str, user_message: str) -> str | None:
    """로컬 에이전트를 직접 호출한다 (A2A 엔드포인트 없을 때 폴백)."""
    try:
        if agent_name in ("coding_assistant", "coder"):
            from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
            from youngs75_a2a.agents.coding_assistant.config import CodingConfig

            agent = await CodingAssistantAgent.create(config=CodingConfig())
            result = await agent.graph.ainvoke({
                "messages": [HumanMessage(content=user_message)],
                "iteration": 0,
                "max_iterations": 2,
            })
            return result.get("generated_code") or result.get("messages", [{}])[-1].content

        if agent_name in ("deep_research", "researcher"):
            from youngs75_a2a.agents.deep_research.agent import DeepResearchAgent
            from youngs75_a2a.agents.deep_research.config import ResearchConfig

            agent = DeepResearchAgent(config=ResearchConfig())
            result = await agent.graph.ainvoke({
                "messages": [HumanMessage(content=user_message)],
            })
            return result.get("final_report", "")

        if agent_name in ("simple_react", "react"):
            from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent
            from youngs75_a2a.agents.simple_react.config import SimpleReActConfig

            agent = await SimpleMCPReActAgent.create(config=SimpleReActConfig())
            result = await agent.graph.ainvoke({
                "messages": [HumanMessage(content=user_message)],
            })
            msgs = result.get("messages", [])
            return msgs[-1].content if msgs else None

    except Exception as e:
        logger.warning(f"로컬 에이전트 '{agent_name}' 호출 실패: {e}")
    return None


async def delegate(state: OrchestratorState, config: RunnableConfig) -> dict:
    """선택된 에이전트에 요청을 위임한다.

    우선순위:
    1. A2A 프로토콜 (HTTP 엔드포인트가 접근 가능한 경우)
    2. 로컬 에이전트 직접 호출 (폴백)
    """
    oc = OrchestratorConfig.from_runnable_config(config)
    selected = state.get("selected_agent", "none")

    if selected == "none":
        return {
            "agent_response": "죄송합니다. 현재 등록된 에이전트 중 적합한 것을 찾지 못했습니다. 질문을 다시 한번 구체적으로 말씀해 주세요."
        }

    # 서브에이전트용 히스토리 필터링 후 마지막 사용자 메시지 추출
    truncated_messages = _orchestrator_context_manager.truncate_for_subagent(
        state["messages"], last_n_turns=3
    )
    user_message = ""
    for msg in reversed(truncated_messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # A2A 프로토콜 시도
    url = oc.get_endpoint_url(selected)
    if url:
        try:
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

        except Exception as e:
            logger.warning(f"A2A 호출 실패 ({selected}), 로컬 폴백: {e}")

    # 로컬 에이전트 폴백
    local_result = await _invoke_local_agent(selected, user_message)
    if local_result:
        return {"agent_response": local_result}

    return {
        "agent_response": f"에이전트 '{selected}'를 호출할 수 없습니다 (A2A 엔드포인트 미설정, 로컬 에이전트 미지원)."
    }


async def coordinate(state: OrchestratorState, config: RunnableConfig) -> dict:
    """복합 작업을 서브태스크로 분해하고 병렬 워커에 위임한다."""
    oc = OrchestratorConfig.from_runnable_config(config)
    llm = oc.get_model("default")

    # 사용자 메시지 추출
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # SubAgentRegistry 구성 — 등록된 에이전트 엔드포인트를 SubAgentSpec으로 변환
    registry = SubAgentRegistry()
    from youngs75_a2a.core.subagents.schemas import SubAgentSpec

    for ep in oc.agent_endpoints:
        registry.register(
            SubAgentSpec(
                name=ep.name,
                description=ep.description,
                capabilities=[ep.name],
                endpoint=ep.url,
            )
        )

    coordinator = CoordinatorMode(
        registry=registry,
        context_manager=_orchestrator_context_manager,
    )

    try:
        result = await coordinator.run(
            task=user_message,
            context=state["messages"],
            llm=llm,
        )
        return {"agent_response": result["synthesized_response"]}
    except Exception as e:
        logger.error(f"코디네이터 실행 실패: {e}")
        return {"agent_response": f"복합 작업 처리 중 오류가 발생했습니다: {e}"}


async def respond(state: OrchestratorState, config: RunnableConfig) -> dict:
    """에이전트 응답을 최종 메시지로 포맷팅한다."""
    selected = state.get("selected_agent", "unknown")
    response = state.get("agent_response", "")

    if selected and selected != "none":
        content = f"[{selected}] {response}"
    else:
        content = response

    return {"messages": [AIMessage(content=content)]}


def _route_after_classify(state: OrchestratorState) -> str:
    """classify 노드 이후 라우팅 결정.

    복합 작업(coordinate)이면 coordinate 노드로,
    그 외에는 기존 delegate 노드로 라우팅한다.
    """
    if state.get("selected_agent") == "__coordinator__":
        return "coordinate"
    return "delegate"


class OrchestratorAgent(BaseGraphAgent):
    """사용자 요청을 분석하고 적합한 에이전트로 라우팅하는 오케스트레이터.

    단일 에이전트 위임(delegate)과 복합 작업 병렬 오케스트레이션(coordinate)을 지원.
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLASSIFY": "classify",
        "DELEGATE": "delegate",
        "COORDINATE": "coordinate",
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
        graph.add_node(self.get_node_name("COORDINATE"), coordinate)
        graph.add_node(self.get_node_name("RESPOND"), respond)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLASSIFY"))
        # classify 이후 조건부 라우팅: delegate 또는 coordinate
        graph.add_conditional_edges(
            self.get_node_name("CLASSIFY"),
            _route_after_classify,
            {
                "delegate": self.get_node_name("DELEGATE"),
                "coordinate": self.get_node_name("COORDINATE"),
            },
        )
        graph.add_edge(self.get_node_name("DELEGATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("COORDINATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("RESPOND"), END)
