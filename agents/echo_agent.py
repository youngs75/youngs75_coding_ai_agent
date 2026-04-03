"""입력을 그대로 반환하는 간단한 Echo 에이전트."""

from __future__ import annotations

from typing import ClassVar

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.base_state import BaseGraphState


class EchoAgent(BaseGraphAgent):
    """마지막 입력 메시지를 그대로 반환하는 Echo 에이전트."""

    NODE_NAMES: ClassVar[dict[str, str]] = {"ECHO": "echo"}

    def __init__(self) -> None:
        super().__init__(
            state_schema=BaseGraphState,
            agent_name="EchoAgent",
        )

    def _echo_node(self, state: BaseGraphState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": [AIMessage(content="")]}

        last_message = messages[-1]
        return {"messages": [AIMessage(content=last_message.content)]}

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("ECHO"), self._echo_node)

    def init_edges(self, graph: StateGraph) -> None:
        graph.set_entry_point(self.get_node_name("ECHO"))
        graph.set_finish_point(self.get_node_name("ECHO"))
