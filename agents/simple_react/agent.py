"""MCP 도구를 사용하는 Simple ReAct 에이전트.

단일 노드로 create_react_agent를 사용하여 MCP 도구를 호출하는 에이전트.
비동기 초기화(MCP 도구 로딩)가 필요하므로 await create()로 생성한다.

사용 예:
    agent = await SimpleMCPReActAgent.create(
        config=SimpleReActConfig(),
        model=ChatOpenAI(model="gpt-5.4"),
    )
    result = await agent.graph.ainvoke({"messages": [HumanMessage("AI 최신 트렌드")]})
"""

from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.base_state import BaseGraphState
from youngs75_a2a.core.mcp_loader import MCPToolLoader
from .config import SimpleReActConfig


class SimpleMCPReActAgent(BaseGraphAgent):
    """MCP 도구를 사용하는 Simple ReAct 에이전트."""

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

    def init_nodes(self, graph: StateGraph) -> None:
        react_agent = create_react_agent(
            model=self.model,
            tools=self._tools,
            prompt=self._react_config.system_prompt,
        )
        graph.add_node(self.get_node_name("REACT"), react_agent)

    def init_edges(self, graph: StateGraph) -> None:
        graph.set_entry_point(self.get_node_name("REACT"))
        graph.set_finish_point(self.get_node_name("REACT"))
