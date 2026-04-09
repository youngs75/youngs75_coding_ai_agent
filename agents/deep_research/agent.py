"""Deep Research 에이전트.

다단계 연구 워크플로우:
  질문 명확화 → 메모리 검색 → 연구 ���리프 → Supervisor(병렬 연구) → 최종 보고서 → 메모리 기록

Phase 10 통합:
- BaseGraphAgent의 context_manager를 통한 컨텍스트 윈도우 관리
- project_context를 시��템 프롬프트에 주입

Phase 14 통합:
- CoALA 메모리 체계: Semantic(프로젝트 규칙) + Episodic(연구 이력) 연동
- retrieve_memory 노드: clarify 후 관련 과거 연구/규칙 자동 주입
- record_episodic 노드: 보고서 완료 후 연구 이력 자동 기록

사용 예:
    agent = DeepResearchAgent(config=ResearchConfig(), memory_store=store)
    result = await agent.graph.ainvoke(
        {"messages": [HumanMessage("양자 컴퓨팅의 최�� 동향")]},
        config={"configurable": ResearchConfig().to_langgraph_configurable()},
    )
    print(result["final_report"])
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from langchain_core.messages import get_buffer_string
from langgraph.graph import START, END, StateGraph

from coding_agent.core.base_agent import BaseGraphAgent
from coding_agent.core.memory.schemas import MemoryItem, MemoryType
from coding_agent.core.memory.store import MemoryStore
from coding_agent.agents.deep_research.config import ResearchConfig
from coding_agent.agents.deep_research.schemas import AgentState
from coding_agent.agents.deep_research.nodes.clarify import clarify_with_user
from coding_agent.agents.deep_research.nodes.brief import write_research_brief
from coding_agent.agents.deep_research.nodes.report import final_report_generation
from coding_agent.agents.deep_research.subgraphs.supervisor import (
    build_supervisor_subgraph,
)

logger = logging.getLogger(__name__)


class DeepResearchAgent(BaseGraphAgent):
    """다단계 심층 연구 에이전트.

    Phase 10: BaseGraphAgent에서 상속받은 context_manager, project_context 활용.
    Phase 14: CoALA 메모리 연동 (Semantic + Episodic).
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLARIFY": "clarify_with_user",
        "RETRIEVE_MEMORY": "retrieve_memory",
        "BRIEF": "write_research_brief",
        "SUPERVISOR": "research_supervisor",
        "REPORT": "final_report_generation",
        "RECORD_EPISODIC": "record_episodic",
    }

    def __init__(
        self,
        *,
        config: ResearchConfig | None = None,
        memory_store: MemoryStore | None = None,
        **kwargs: Any,
    ) -> None:
        self._research_config = config or ResearchConfig()
        self._memory_store = memory_store

        super().__init__(
            config=self._research_config,
            state_schema=AgentState,
            agent_name="DeepResearchAgent",
            **kwargs,
        )

    # ── 메모리 노드 ──────────────────────────────────────────

    async def _retrieve_memory(self, state: AgentState) -> dict[str, Any]:
        """Semantic/Episodic Memory를 검색하여 상태에 주입한다.

        clarify 후 brief 전에 실행되어 프로젝트 규칙과
        과거 유사 연구 이력을 자동으로 컨텍스트에 포함시킨다.
        """
        if not self._memory_store:
            return {}

        # 사용자 질문에서 검색 쿼리 추출
        messages = state.get("messages", [])
        query = get_buffer_string(messages[-3:]) if messages else ""

        result: dict[str, Any] = {}

        # Semantic Memory: 프로젝트 규칙/컨벤션
        try:
            semantic_items = self._memory_store.list_by_type(MemoryType.SEMANTIC)
            if semantic_items:
                result["semantic_context"] = [item.content for item in semantic_items]
        except Exception:
            pass

        # Episodic Memory: 과거 유��� 연구 이력
        try:
            episodes = self._memory_store.search(
                query=query,
                memory_type=MemoryType.EPISODIC,
                limit=3,
            )
            if episodes:
                result["episodic_log"] = [e.content for e in episodes]
        except Exception:
            pass

        return result

    async def _record_episodic(self, state: AgentState) -> dict[str, Any]:
        """연구 완료 후 결과를 Episodic Memory에 기록한다."""
        if not self._memory_store:
            return {}

        try:
            research_brief = state.get("research_brief", "")
            final_report = state.get("final_report", "")

            # 보고서 요약 (200자 제한)
            summary = final_report[:200] if final_report else "보고서 없음"

            content = f"[연구 완료] {research_brief[:100]} | 결과: {summary}"
            item = MemoryItem(
                type=MemoryType.EPISODIC,
                content=content,
                tags=["research", "deep_research"],
                metadata={
                    "research_brief": research_brief[:200],
                    "report_length": len(final_report),
                },
            )
            self._memory_store.put(item)
        except Exception:
            pass

        return {}

    # ── 그래프 구성 ─────────────────────────────────────────

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("CLARIFY"), clarify_with_user)
        graph.add_node(self.get_node_name("RETRIEVE_MEMORY"), self._retrieve_memory)
        graph.add_node(self.get_node_name("BRIEF"), write_research_brief)
        graph.add_node(self.get_node_name("SUPERVISOR"), build_supervisor_subgraph())
        graph.add_node(self.get_node_name("REPORT"), final_report_generation)
        graph.add_node(self.get_node_name("RECORD_EPISODIC"), self._record_episodic)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLARIFY"))
        # clarify → retrieve_memory → brief → supervisor → report → record → END
        graph.add_edge(
            self.get_node_name("RETRIEVE_MEMORY"),
            self.get_node_name("BRIEF"),
        )
        graph.add_edge(self.get_node_name("BRIEF"), self.get_node_name("SUPERVISOR"))
        graph.add_edge(self.get_node_name("SUPERVISOR"), self.get_node_name("REPORT"))
        graph.add_edge(
            self.get_node_name("REPORT"),
            self.get_node_name("RECORD_EPISODIC"),
        )
        graph.add_edge(self.get_node_name("RECORD_EPISODIC"), END)
