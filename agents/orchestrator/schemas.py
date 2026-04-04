"""Orchestrator 에이전트 상태 스키마."""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """Orchestrator 상태.

    messages: 사용자와의 대화 이력
    selected_agent: 라우팅 결정된 에이전트 이름
    agent_response: 하위 에이전트의 응답
    """

    messages: Annotated[list[BaseMessage], add_messages]
    selected_agent: Optional[str]
    agent_response: Optional[str]
