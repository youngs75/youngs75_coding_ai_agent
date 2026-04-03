"""CLI 세션 관리.

대화 히스토리, 에이전트 상태, 메모리 스토어를 세션 단위로 관리한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from youngs75_a2a.core.memory.store import MemoryStore


class SessionInfo(BaseModel):
    """세션 메타데이터."""

    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str = "coding_assistant"
    message_count: int = 0


class CLISession:
    """CLI 세션 — 대화 상태와 메모리를 관리한다."""

    def __init__(self, agent_name: str = "coding_assistant"):
        self.info = SessionInfo(agent_name=agent_name)
        self.memory = MemoryStore()
        self._history: list[dict[str, str]] = []

    @property
    def session_id(self) -> str:
        return self.info.session_id

    def add_message(self, role: str, content: str) -> None:
        """대화 메시지 추가."""
        self._history.append({"role": role, "content": content})
        self.info.message_count += 1

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
        self.info.message_count = 0

    def switch_agent(self, agent_name: str) -> None:
        """에이전트 전환."""
        self.info.agent_name = agent_name
