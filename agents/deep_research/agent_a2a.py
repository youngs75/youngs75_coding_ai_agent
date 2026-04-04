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
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.memory.store import MemoryStore
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
    """A2A 통합 + HITL 지원 Deep Research 에이전트.

    Phase 14: DeepResearchAgent와 동일한 메모리 연동 (Semantic + Episodic).
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLARIFY": "clarify_with_user",
        "RETRIEVE_MEMORY": "retrieve_memory",
        "BRIEF": "write_research_brief",
        "SUPERVISOR": "research_supervisor",
        "REPORT": "final_report_generation",
        "RECORD_EPISODIC": "record_episodic",
        "HITL_APPROVAL": "hitl_final_approval",
        "REVISE": "revise_final_report",
    }

    def __init__(
        self,
        *,
        config: ResearchConfig | None = None,
        use_a2a_supervisor: bool = True,
        memory_store: MemoryStore | None = None,
        **kwargs: Any,
    ) -> None:
        self._research_config = config or ResearchConfig()
        self._use_a2a_supervisor = use_a2a_supervisor
        self._memory_store = memory_store

        # HITL에는 checkpointer가 필요 (interrupt → resume)
        kwargs.setdefault("checkpointer", InMemorySaver())

        super().__init__(
            config=self._research_config,
            state_schema=HITLAgentState,
            agent_name="DeepResearchA2AAgent",
            **kwargs,
        )

    # ── 메모리 노드 (DeepResearchAgent와 동일 로직 위임) ──

    async def _retrieve_memory(self, state: HITLAgentState) -> dict[str, Any]:
        """Semantic/Episodic Memory를 검색하여 상태에 주입한다."""
        if not self._memory_store:
            return {}

        from langchain_core.messages import get_buffer_string as _buf

        messages = state.get("messages", [])
        query = _buf(messages[-3:]) if messages else ""
        result: dict[str, Any] = {}

        try:
            semantic_items = self._memory_store.list_by_type(MemoryType.SEMANTIC)
            if semantic_items:
                result["semantic_context"] = [item.content for item in semantic_items]
        except Exception:
            pass

        try:
            episodes = self._memory_store.search(
                query=query, memory_type=MemoryType.EPISODIC, limit=3
            )
            if episodes:
                result["episodic_log"] = [e.content for e in episodes]
        except Exception:
            pass

        return result

    async def _record_episodic(self, state: HITLAgentState) -> dict[str, Any]:
        """연구 완료 후 결과를 Episodic Memory에 기록한다."""
        if not self._memory_store:
            return {}

        try:
            research_brief = state.get("research_brief", "")
            final_report = state.get("final_report", "")
            summary = final_report[:200] if final_report else "보고서 없음"

            item = MemoryItem(
                type=MemoryType.EPISODIC,
                content=f"[연구 완료] {research_brief[:100]} | 결과: {summary}",
                tags=["research", "deep_research", "a2a"],
                metadata={
                    "research_brief": research_brief[:200],
                    "report_length": len(final_report),
                },
            )
            self._memory_store.put(item)
        except Exception:
            pass

        return {}

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("CLARIFY"), clarify_with_user)
        graph.add_node(self.get_node_name("RETRIEVE_MEMORY"), self._retrieve_memory)
        graph.add_node(self.get_node_name("BRIEF"), write_research_brief)

        # A2A supervisor 또는 로컬 supervisor
        if self._use_a2a_supervisor:
            graph.add_node(self.get_node_name("SUPERVISOR"), call_supervisor_a2a)
        else:
            graph.add_node(
                self.get_node_name("SUPERVISOR"), build_supervisor_subgraph()
            )

        graph.add_node(self.get_node_name("REPORT"), final_report_generation)
        graph.add_node(self.get_node_name("RECORD_EPISODIC"), self._record_episodic)
        graph.add_node(self.get_node_name("HITL_APPROVAL"), hitl_final_approval)
        graph.add_node(self.get_node_name("REVISE"), revise_final_report)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLARIFY"))
        # clarify → retrieve_memory → brief
        graph.add_edge(
            self.get_node_name("RETRIEVE_MEMORY"),
            self.get_node_name("BRIEF"),
        )
        graph.add_edge(self.get_node_name("BRIEF"), self.get_node_name("SUPERVISOR"))
        graph.add_edge(self.get_node_name("SUPERVISOR"), self.get_node_name("REPORT"))
        # report → record_episodic → (hitl or end)
        graph.add_edge(
            self.get_node_name("REPORT"),
            self.get_node_name("RECORD_EPISODIC"),
        )
        graph.add_conditional_edges(
            self.get_node_name("RECORD_EPISODIC"),
            lambda state, config: route_after_final_report(state, config),
        )
        # revise_final_report는 Command로 research_supervisor로 되돌아감
