"""Deep Research A2A 에이전트.

DeepResearchAgent를 확장하여:
- Supervisor를 A2A 프로토콜로 외부 에이전트에 위임
- HITL 승인 루프 지원 (LangGraph interrupt 기반)

사용 예:
    agent = DeepResearchA2AAgent(config=ResearchConfig(enable_hitl=True))
    result = await agent.graph.ainvoke(
        {"messages": [HumanMessage("양자 컴퓨팅의 최신 동향")]},
        config={"configurable": ResearchConfig().to_langgraph_configurable()},
    )
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.agents.deep_research.config import ResearchConfig
from youngs75_a2a.agents.deep_research.schemas import HITLAgentState
from youngs75_a2a.agents.deep_research.nodes.clarify import clarify_with_user
from youngs75_a2a.agents.deep_research.nodes.brief import write_research_brief
from youngs75_a2a.agents.deep_research.nodes.report import final_report_generation
from youngs75_a2a.agents.deep_research.nodes.hitl import (
    hitl_final_approval,
    revise_final_report,
)
from youngs75_a2a.agents.deep_research.subgraphs.supervisor import (
    build_supervisor_subgraph,
)

logger = logging.getLogger(__name__)


async def call_supervisor_a2a(state: HITLAgentState, config: RunnableConfig) -> dict:
    """Supervisor를 A2A 프로토콜로 외부 에이전트에 위임한다.

    A2A 엔드포인트가 접근 불가능하면 로컬 supervisor 서브그래프로 폴백한다.
    """
    rc = ResearchConfig.from_runnable_config(config)
    endpoint = rc.a2a_agent_endpoints.get("supervisor", "http://localhost:8092")

    # A2A 클라이언트로 supervisor 호출 시도
    try:
        import httpx

        payload = {
            "research_brief": state.get("research_brief", ""),
            "supervisor_messages": [
                {
                    "role": "human",
                    "content": get_buffer_string(
                        state.get("supervisor_messages") or []
                    ),
                }
            ],
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{endpoint}/send", json=payload)
            resp.raise_for_status()
            result = resp.json()

        # A2A 응답에서 notes/raw_notes 추출
        notes = result.get("notes") or []
        raw_notes = result.get("raw_notes") or []

        return {
            "notes": notes,
            "raw_notes": raw_notes,
        }

    except Exception as e:
        logger.warning(f"A2A supervisor 호출 실패, 로컬 폴백: {e}")

        # 로컬 supervisor 서브그래프로 폴백
        supervisor_graph = build_supervisor_subgraph()
        result = await supervisor_graph.ainvoke(
            {
                "supervisor_messages": state.get("supervisor_messages") or [],
                "research_brief": state.get("research_brief", ""),
            },
            config=config,
        )
        return {
            "notes": result.get("notes") or [],
            "raw_notes": result.get("raw_notes") or [],
        }


def route_after_final_report(state: HITLAgentState, config: RunnableConfig) -> str:
    """최종 보고서 이후 라우팅: HITL 활성화 시 승인 게이트로, 아니면 종료."""
    rc = ResearchConfig.from_runnable_config(config)
    if rc.enable_hitl:
        return "hitl_final_approval"
    return "__end__"


class DeepResearchA2AAgent(BaseGraphAgent):
    """A2A 통합 + HITL 지원 Deep Research 에이전트."""

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLARIFY": "clarify_with_user",
        "BRIEF": "write_research_brief",
        "SUPERVISOR": "research_supervisor",
        "REPORT": "final_report_generation",
        "HITL_APPROVAL": "hitl_final_approval",
        "REVISE": "revise_final_report",
    }

    def __init__(
        self,
        *,
        config: ResearchConfig | None = None,
        use_a2a_supervisor: bool = True,
        **kwargs: Any,
    ) -> None:
        self._research_config = config or ResearchConfig()
        self._use_a2a_supervisor = use_a2a_supervisor

        # HITL에는 checkpointer가 필요 (interrupt → resume)
        kwargs.setdefault("checkpointer", InMemorySaver())

        super().__init__(
            config=self._research_config,
            state_schema=HITLAgentState,
            agent_name="DeepResearchA2AAgent",
            **kwargs,
        )

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("CLARIFY"), clarify_with_user)
        graph.add_node(self.get_node_name("BRIEF"), write_research_brief)

        # A2A supervisor 또는 로컬 supervisor
        if self._use_a2a_supervisor:
            graph.add_node(self.get_node_name("SUPERVISOR"), call_supervisor_a2a)
        else:
            graph.add_node(
                self.get_node_name("SUPERVISOR"), build_supervisor_subgraph()
            )

        graph.add_node(self.get_node_name("REPORT"), final_report_generation)
        graph.add_node(self.get_node_name("HITL_APPROVAL"), hitl_final_approval)
        graph.add_node(self.get_node_name("REVISE"), revise_final_report)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLARIFY"))
        # clarify_with_user는 Command로 라우팅
        graph.add_edge(self.get_node_name("BRIEF"), self.get_node_name("SUPERVISOR"))
        graph.add_edge(self.get_node_name("SUPERVISOR"), self.get_node_name("REPORT"))
        graph.add_conditional_edges(
            self.get_node_name("REPORT"),
            route_after_final_report,
        )
        # revise_final_report는 Command로 research_supervisor로 되돌아감
