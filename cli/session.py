"""CLI 세션 관리.

대화 히스토리, 에이전트 상태, 메모리 스토어를 세션 단위로 관리한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from youngs75_a2a.core.memory.store import MemoryStore
from youngs75_a2a.core.skills.registry import SkillRegistry


class SessionInfo(BaseModel):
    """세션 메타데이터."""

    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str = "coding_assistant"
    message_count: int = 0


class CLISession:
    """CLI 세션 — 대화 상태와 메모리를 관리한다."""

    def __init__(
        self,
        agent_name: str = "coding_assistant",
        skill_registry: SkillRegistry | None = None,
        checkpointer: Any | None = None,
    ):
        self.info = SessionInfo(agent_name=agent_name)
        self.memory = MemoryStore()
        self.skills = skill_registry or SkillRegistry()
        self.checkpointer = checkpointer
        self._history: list[dict[str, str]] = []
        self._agents: dict[str, Any] = {}

    @property
    def session_id(self) -> str:
        return self.info.session_id

    @property
    def thread_id(self) -> str:
        """멀티턴 대화 상태 유지를 위한 스레드 ID."""
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

    def get_cached_agent(self, name: str) -> Any | None:
        """캐싱된 에이전트 인스턴스 반환."""
        return self._agents.get(name)

    def cache_agent(self, name: str, agent: Any) -> None:
        """에이전트 인스턴스 캐싱."""
        self._agents[name] = agent

    def switch_agent(self, agent_name: str) -> None:
        """에이전트 전환."""
        self.info.agent_name = agent_name

    def get_history_summary(self, limit: int = 10) -> list[dict[str, str]]:
        """최근 대화 히스토리 요약을 반환한다."""
        return list(self._history[-limit:])

    def activate_skill(self, name: str) -> str | None:
        """스킬을 L2 레벨로 활성화하고 이름을 반환한다.

        Returns:
            활성화된 스킬 이름 또는 None (스킬이 없을 경우)
        """
        skill = self.skills.activate(name)
        if skill is None:
            return None
        return skill.name
