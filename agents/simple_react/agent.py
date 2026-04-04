"""MCP 도구를 사용하는 Simple ReAct 에이전트.

단일 노드로 create_agent를 사용하여 MCP 도구를 호출하는 에이전트.
비동기 초기화(MCP 도구 로딩)가 필요하므로 await create()로 생성한다.

Phase 10 통합:
- BaseGraphAgent의 context_manager를 통한 컨텍스트 윈도우 관리
- project_context를 시스템 프롬프트에 주입

사용 예:
    agent = await SimpleMCPReActAgent.create(
        config=SimpleReActConfig(),
        model=ChatOpenAI(model="deepseek/deepseek-v3.2"),
    )
    result = await agent.graph.ainvoke({"messages": [HumanMessage("AI 최신 트렌드")]})
"""

from __future__ import annotations

from typing import Any, ClassVar

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.base_state import BaseGraphState
from youngs75_a2a.core.mcp_loader import MCPToolLoader
from .config import SimpleReActConfig


class SimpleMCPReActAgent(BaseGraphAgent):
    """MCP 도구를 사용하는 Simple ReAct 에이전트.

    Phase 10: BaseGraphAgent에서 상속받은 context_manager, project_context 활용.
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {"REACT": "react_agent"}

    def __init__(
        self,
        *,
        config: SimpleReActConfig | None = None,
        model: BaseChatModel | None = None,
        **kwargs: Any,
    ) -> None:
        self._react_config = config or SimpleReActConfig()
        self._mcp_loader = MCPToolLoader(self._react_config.mcp_servers)
        self._tools: list[Any] = []

        if model is None and self._react_config:
            model = self._react_config.get_model()

        kwargs.pop("auto_build", None)
        super().__init__(
            config=self._react_config,
            model=model,
            state_schema=BaseGraphState,
            agent_name="SimpleMCPReActAgent",
            auto_build=False,
            **kwargs,
        )

    async def async_init(self) -> None:
        """MCP 도구를 비동기로 로딩한다."""
        self._tools = await self._mcp_loader.load()

    def _get_system_prompt(self) -> str:
        """프로젝트 컨텍스트가 포함된 시스템 프롬프트를 반환한다."""
        base_prompt = self._react_config.system_prompt or ""
        return self._build_system_prompt(base_prompt)

    def init_nodes(self, graph: StateGraph) -> None:
        # 프로젝트 컨텍스트가 주입된 시스템 프롬프트 사용
        prompt = self._get_system_prompt()
        # create_agent: langgraph deprecated → langchain 마이그레이션
        react_agent = create_agent(
            model=self.model,
            tools=self._tools,
            system_prompt=prompt if prompt else None,
        )
        graph.add_node(self.get_node_name("REACT"), react_agent)

    def init_edges(self, graph: StateGraph) -> None:
        graph.set_entry_point(self.get_node_name("REACT"))
        graph.set_finish_point(self.get_node_name("REACT"))
