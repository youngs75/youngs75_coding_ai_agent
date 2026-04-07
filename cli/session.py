"""CLI 세션 관리.

대화 히스토리, 에이전트 상태, 메모리 스토어를 세션 단위로 관리한다.

Phase 10 통합:
- project_context: 프로젝트 컨텍스트 문자열 (에이전트에 전달)
- permission_manager: 도구 권한 관리자 (에이전트에 전달)
- tool_executor: 병렬 도구 실행기 (에이전트에 전달)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from youngs75_a2a.core.memory.store import MemoryStore
from youngs75_a2a.core.parallel_tool_executor import ParallelToolExecutor
from youngs75_a2a.core.skills.registry import SkillRegistry
from youngs75_a2a.core.tool_permissions import ToolPermissionManager


class SessionInfo(BaseModel):
    """세션 메타데이터."""

    session_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str = "coding_assistant"
    message_count: int = 0


class CLISession:
    """CLI 세션 — 대화 상태와 메모리를 관리한다.

    Phase 10 통합 속성:
    - project_context: 에이전트 시스템 프롬프트에 주입할 프로젝트 컨텍스트
    - permission_manager: 도구 실행 권한 관리자
    - tool_executor: 병렬 도구 실행기
    """

    def __init__(
        self,
        agent_name: str = "coding_assistant",
        skill_registry: SkillRegistry | None = None,
        checkpointer: Any | None = None,
    ):
        self.info = SessionInfo(agent_name=agent_name)
        # Procedural Memory를 .ai/memory/에 영속화 (세션 간 학습 유지)
        # 워크스페이스가 .ai 디렉토리를 포함하면 영속화 활성화
        persist_dir = Path.cwd() / ".ai" / "memory"
        self.memory = MemoryStore(
            persist_dir=persist_dir if persist_dir.parent.exists() else None,
        )
        self.skills = skill_registry or SkillRegistry()
        self.checkpointer = checkpointer
        self._history: list[dict[str, str]] = []
        self._agents: dict[str, Any] = {}

        # Phase 10 통합 필드
        self.project_context: str | None = None
        self.permission_manager: ToolPermissionManager | None = None
        self.tool_executor: ParallelToolExecutor | None = None

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
