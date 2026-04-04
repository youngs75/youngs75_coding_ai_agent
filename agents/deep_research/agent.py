"""Deep Research 에이전트.

다단계 연구 워크플로우:
  질문 명확화 → 연구 브리프 → Supervisor(병렬 연구) → 최종 보고서

Phase 10 통합:
- BaseGraphAgent의 context_manager를 통한 컨텍스트 윈도우 관리
- project_context를 시스템 프롬프트에 주입

사용 예:
    agent = DeepResearchAgent(config=ResearchConfig())
    result = await agent.graph.ainvoke(
        {"messages": [HumanMessage("양자 컴퓨팅의 최신 동향")]},
        config={"configurable": ResearchConfig().to_langgraph_configurable()},
    )
    print(result["final_report"])
"""

from __future__ import annotations

from typing import Any, ClassVar

from langgraph.graph import START, END, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.agents.deep_research.config import ResearchConfig
from youngs75_a2a.agents.deep_research.schemas import AgentState
from youngs75_a2a.agents.deep_research.nodes.clarify import clarify_with_user
from youngs75_a2a.agents.deep_research.nodes.brief import write_research_brief
from youngs75_a2a.agents.deep_research.nodes.report import final_report_generation
from youngs75_a2a.agents.deep_research.subgraphs.supervisor import (
    build_supervisor_subgraph,
)


class DeepResearchAgent(BaseGraphAgent):
    """다단계 심층 연구 에이전트.

    Phase 10: BaseGraphAgent에서 상속받은 context_manager, project_context 활용.
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLARIFY": "clarify_with_user",
        "BRIEF": "write_research_brief",
        "SUPERVISOR": "research_supervisor",
        "REPORT": "final_report_generation",
    }

    def __init__(
        self,
        *,
        config: ResearchConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self._research_config = config or ResearchConfig()

        super().__init__(
            config=self._research_config,
            state_schema=AgentState,
            agent_name="DeepResearchAgent",
            **kwargs,
        )

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("CLARIFY"), clarify_with_user)
        graph.add_node(self.get_node_name("BRIEF"), write_research_brief)
        graph.add_node(self.get_node_name("SUPERVISOR"), build_supervisor_subgraph())
        graph.add_node(self.get_node_name("REPORT"), final_report_generation)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLARIFY"))
        # clarify_with_user는 Command로 라우팅하므로 명시적 엣지 불필요
        graph.add_edge(self.get_node_name("BRIEF"), self.get_node_name("SUPERVISOR"))
        graph.add_edge(self.get_node_name("SUPERVISOR"), self.get_node_name("REPORT"))
        graph.add_edge(self.get_node_name("REPORT"), END)
