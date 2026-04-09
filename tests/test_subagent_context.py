"""SubagentContextFilter 테스트.

검증 항목:
- build_init_state: 최소 state 생성, _EXCLUDED_STATE_KEYS 필터링
- compact_result: written_files 중복제거, code_summary 자르기
- build_phase_task_message: 메시지 구조, 크기 제한, 통합 리마인더
- get_excluded_state_keys: frozenset 반환
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from coding_agent.core.subagent_context import (
    SubagentContextFilter,
    _EXCLUDED_STATE_KEYS,
    get_excluded_state_keys,
)


class TestGetExcludedStateKeys:
    def test_returns_frozenset(self):
        result = get_excluded_state_keys()
        assert isinstance(result, frozenset)

    def test_contains_messages(self):
        assert "messages" in get_excluded_state_keys()

    def test_contains_skill_context(self):
        assert "skill_context" in get_excluded_state_keys()

    def test_contains_project_context(self):
        assert "project_context" in get_excluded_state_keys()


class TestBuildInitState:
    def test_minimal_state(self):
        state = SubagentContextFilter.build_init_state("build a kanban board")
        assert len(state["messages"]) == 1
        assert isinstance(state["messages"][0], HumanMessage)
        assert state["messages"][0].content == "build a kanban board"
        assert state["iteration"] == 0
        assert state["max_iterations"] == 11

    def test_env_approved(self):
        state = SubagentContextFilter.build_init_state("task", env_approved=True)
        assert state["env_approved"] is True

    def test_env_not_approved_by_default(self):
        state = SubagentContextFilter.build_init_state("task")
        assert "env_approved" not in state

    def test_custom_max_iterations(self):
        state = SubagentContextFilter.build_init_state("task", max_iterations=5)
        assert state["max_iterations"] == 5

    def test_extra_state_filtered(self):
        """_EXCLUDED_STATE_KEYS에 포함된 키는 전달되지 않아야 한다."""
        extra = {
            "messages": [HumanMessage(content="should be excluded")],
            "skill_context": ["flask_vue body"],
            "project_context": ["file content"],
            "custom_key": "should be passed",
        }
        state = SubagentContextFilter.build_init_state("task", extra_state=extra)
        # 제외 대상은 없어야 함
        assert state["messages"][0].content == "task"  # messages는 task로 덮어쓰기
        assert "skill_context" not in state
        assert "project_context" not in state
        # 비제외 키는 통과
        assert state["custom_key"] == "should be passed"

    def test_extra_state_none(self):
        state = SubagentContextFilter.build_init_state("task", extra_state=None)
        assert "custom_key" not in state


class TestCompactResult:
    def test_basic_result(self):
        result = {
            "written_files": ["a.py", "b.py"],
            "test_passed": True,
            "exit_reason": "",
            "generated_code": "print('hello')",
        }
        compacted = SubagentContextFilter.compact_result(result)
        assert compacted["test_passed"] is True
        assert compacted["exit_reason"] == ""
        assert compacted["code_summary"] == "print('hello')"

    def test_code_summary_truncation(self):
        long_code = "x" * 1000
        result = {"generated_code": long_code, "written_files": []}
        compacted = SubagentContextFilter.compact_result(result)
        assert len(compacted["code_summary"]) == 503  # 500 + "..."
        assert compacted["code_summary"].endswith("...")

    def test_written_files_dedup_strings(self):
        result = {
            "written_files": ["a.py", "b.py", "a.py", "c.py"],
            "generated_code": "",
        }
        compacted = SubagentContextFilter.compact_result(result)
        paths = [str(f) for f in compacted["written_files"]]
        assert len(paths) == 3
        assert "a.py" in paths

    def test_empty_result(self):
        compacted = SubagentContextFilter.compact_result({})
        assert compacted["written_files"] == []
        assert compacted["code_summary"] == ""

    def test_failed_result(self):
        result = {"test_passed": False, "exit_reason": "budget_exceeded"}
        compacted = SubagentContextFilter.compact_result(result)
        assert compacted["test_passed"] is False
        assert compacted["exit_reason"] == "budget_exceeded"


class TestBuildPhaseTaskMessage:
    def test_basic_structure(self):
        msg = SubagentContextFilter.build_phase_task_message(
            user_message="칸반보드 만들어줘",
            plan_summary="Flask + Vue 칸반보드",
            architecture="2-tier REST API + SPA",
            phase={"title": "백엔드 API", "instructions": "Flask routes 구현"},
            phase_index=0,
            total_phases=3,
            prior_written_files=[],
        )
        assert "Phase 1/3" in msg
        assert "백엔드 API" in msg
        assert "Flask routes 구현" in msg
        assert "칸반보드 만들어줘" in msg

    def test_user_message_truncation(self):
        long_msg = "a" * 1000
        msg = SubagentContextFilter.build_phase_task_message(
            user_message=long_msg,
            plan_summary="summary",
            architecture="arch",
            phase={"title": "test", "instructions": "do stuff"},
            phase_index=0,
            total_phases=1,
            prior_written_files=[],
            max_user_message_chars=100,
        )
        # 원본 1000자가 100자 + "..."로 잘려야 함
        assert "a" * 100 + "..." in msg
        assert "a" * 101 not in msg

    def test_prior_files_listed(self):
        msg = SubagentContextFilter.build_phase_task_message(
            user_message="request",
            plan_summary="summary",
            architecture="arch",
            phase={"title": "Phase 2", "instructions": "build frontend"},
            phase_index=1,
            total_phases=2,
            prior_written_files=["backend/app.py", "backend/models.py"],
        )
        assert "`backend/app.py`" in msg
        assert "`backend/models.py`" in msg
        assert "read_file" in msg  # MCP 도구 사용 안내

    def test_integration_reminder_after_phase_0(self):
        msg = SubagentContextFilter.build_phase_task_message(
            user_message="request",
            plan_summary="summary",
            architecture="arch",
            phase={"title": "Phase 2", "instructions": "instructions"},
            phase_index=1,
            total_phases=3,
            prior_written_files=["a.py"],
        )
        assert "통합" in msg or "import" in msg

    def test_no_integration_reminder_for_phase_0(self):
        msg = SubagentContextFilter.build_phase_task_message(
            user_message="request",
            plan_summary="summary",
            architecture="arch",
            phase={"title": "Phase 1", "instructions": "instructions"},
            phase_index=0,
            total_phases=3,
            prior_written_files=[],
        )
        # Phase 0에는 통합 리마인더 없음
        assert "⚠️" not in msg

    def test_message_size_reasonable(self):
        """Phase 메시지가 ~4KB 이하여야 한다."""
        msg = SubagentContextFilter.build_phase_task_message(
            user_message="x" * 500,
            plan_summary="y" * 200,
            architecture="z" * 200,
            phase={"title": "Test Phase", "instructions": "w" * 500},
            phase_index=2,
            total_phases=3,
            prior_written_files=[f"file{i}.py" for i in range(20)],
        )
        assert len(msg) < 4096, f"Message too large: {len(msg)} chars"

    def test_instructions_as_list(self):
        msg = SubagentContextFilter.build_phase_task_message(
            user_message="request",
            plan_summary="summary",
            architecture="arch",
            phase={"title": "Phase 1", "instructions": ["step 1", "step 2", "step 3"]},
            phase_index=0,
            total_phases=1,
            prior_written_files=[],
        )
        assert "- step 1" in msg
        assert "- step 2" in msg
