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
from langgraph.types import interrupt

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
4. 코드 생성/개발 요청은 항상 coding_assistant를 선택하세요.
   프론트엔드+백엔드, 여러 파일, 풀스택 등 하나의 프로젝트를 구현하는 요청은 모두 코딩 작업입니다.
5. "coordinate"는 서로 다른 종류의 에이전트가 순차적으로 필요한 경우에만 사용하세요.
   예시: "기술을 조사한 뒤 그 결과로 코드를 작성해줘" (research → coding 순차 협업)
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


async def _invoke_planner(user_message: str) -> str | None:
    """Planner Agent를 호출하여 구현 계획을 생성한다."""
    try:
        from youngs75_a2a.agents.planner.agent import PlannerAgent
        from youngs75_a2a.agents.planner.config import PlannerConfig

        planner = await PlannerAgent.create(config=PlannerConfig())
        result = await planner.graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "user_request": user_message,
            }
        )
        return result.get("plan_text", "")
    except Exception as e:
        logger.warning(f"Planner 호출 실패, 계획 없이 진행: {e}")
        return None


async def _invoke_local_agent(
    agent_name: str,
    user_message: str,
    *,
    task_plan: str | None = None,
) -> str | None:
    """로컬 에이전트를 직접 호출한다 (A2A 엔드포인트 없을 때 폴백)."""
    try:
        if agent_name in ("coding_assistant", "coder"):
            from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
            from youngs75_a2a.agents.coding_assistant.config import CodingConfig

            # 계획이 있으면 사용자 메시지에 포함
            effective_message = user_message
            if task_plan:
                effective_message = f"{user_message}\n\n{task_plan}"

            agent = await CodingAssistantAgent.create(config=CodingConfig())
            result = await agent.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=effective_message)],
                    "iteration": 0,
                    "max_iterations": 2,
                }
            )
            code = (
                result.get("generated_code") or result.get("messages", [{}])[-1].content
            )
            # 파일 저장 결과를 응답에 포함
            written = result.get("written_files", [])
            if written:
                code += "\n\n📁 저장된 파일:\n" + "\n".join(f"  • {f}" for f in written)
            return code

        if agent_name in ("deep_research", "researcher"):
            from youngs75_a2a.agents.deep_research.agent import DeepResearchAgent
            from youngs75_a2a.agents.deep_research.config import ResearchConfig

            agent = DeepResearchAgent(config=ResearchConfig())
            result = await agent.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                }
            )
            return result.get("final_report", "")

        if agent_name in ("simple_react", "react"):
            from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent
            from youngs75_a2a.agents.simple_react.config import SimpleReActConfig

            agent = await SimpleMCPReActAgent.create(config=SimpleReActConfig())
            result = await agent.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                }
            )
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
                            if (
                                hasattr(root, "text")
                                and root.text
                                and len(root.text) > 5
                            ):
                                return {"agent_response": root.text}
                if hasattr(obj, "status"):
                    return {"agent_response": f"[에이전트 작업 상태: {obj.status}]"}

        except Exception as e:
            logger.warning(f"A2A 호출 실패 ({selected}), 로컬 폴백: {e}")

    # 로컬 에이전트 폴백 — 계획이 있으면 함께 전달
    task_plan = state.get("task_plan")
    local_result = await _invoke_local_agent(
        selected, user_message, task_plan=task_plan
    )
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


async def plan(state: OrchestratorState, config: RunnableConfig) -> dict:
    """코딩 태스크에 대해 Planner Agent로 구현 계획을 수립한다.

    coding_assistant 위임 전에 실행되어 구조화된 계획을 생성한다.
    계획 수립 후 interrupt()로 사용자 승인을 대기한다 (Human-in-the-loop).
    비코딩 태스크는 패스스루 (계획 없이 바로 delegate).
    """
    selected = state.get("selected_agent", "none")

    # 코딩 에이전트 대상만 계획 수립
    if selected not in ("coding_assistant", "coder"):
        return {"task_plan": None}

    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    logger.info("Planner Agent 호출: '%s...'", user_message[:50])
    plan_text = await _invoke_planner(user_message)

    if plan_text:
        logger.info("계획 수립 완료 (%d chars)", len(plan_text))
        # Human-in-the-loop: 계획을 사용자에게 보여주고 승인 대기
        approved = interrupt(plan_text)
        if not approved:
            logger.info("사용자가 계획을 거부함")
            return {"task_plan": None}
        logger.info("사용자가 계획을 승인함")
        return {"task_plan": plan_text}

    logger.info("계획 수립 스킵 (planner 미응답)")
    return {"task_plan": None}


def _route_after_classify(state: OrchestratorState) -> str:
    """classify 노드 이후 라우팅 결정.

    복합 작업(coordinate)이면 coordinate 노드로,
    코딩 작업이면 plan 노드로 (계획 수립 후 delegate),
    그 외에는 delegate로 직행.
    """
    selected = state.get("selected_agent", "")
    if selected == "__coordinator__":
        return "coordinate"
    if selected in ("coding_assistant", "coder"):
        return "plan"
    return "delegate"


class OrchestratorAgent(BaseGraphAgent):
    """사용자 요청을 분석하고 적합한 에이전트로 라우팅하는 오케스트레이터.

    단일 에이전트 위임(delegate)과 복합 작업 병렬 오케스트레이션(coordinate)을 지원.
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLASSIFY": "classify",
        "PLAN": "plan",
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
        graph.add_node(self.get_node_name("PLAN"), plan)
        graph.add_node(self.get_node_name("DELEGATE"), delegate)
        graph.add_node(self.get_node_name("COORDINATE"), coordinate)
        graph.add_node(self.get_node_name("RESPOND"), respond)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLASSIFY"))
        # classify 이후 조건부 라우팅: plan(코딩) / delegate(기타) / coordinate(복합)
        graph.add_conditional_edges(
            self.get_node_name("CLASSIFY"),
            _route_after_classify,
            {
                "plan": self.get_node_name("PLAN"),
                "delegate": self.get_node_name("DELEGATE"),
                "coordinate": self.get_node_name("COORDINATE"),
            },
        )
        # plan → delegate (계획 수립 후 코딩 에이전트에 위임)
        graph.add_edge(self.get_node_name("PLAN"), self.get_node_name("DELEGATE"))
        graph.add_edge(self.get_node_name("DELEGATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("COORDINATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("RESPOND"), END)
