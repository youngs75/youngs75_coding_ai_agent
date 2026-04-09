"""SubAgent 8단계 상태 머신 + 수명주기 테스트."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from coding_agent.core.subagents import (
    VALID_TRANSITIONS,
    SubAgentEvent,
    SubAgentInstance,
    SubAgentRegistry,
    SubAgentSpec,
    SubAgentStatus,
)


# ── 공통 fixture ──


@pytest.fixture()
def registry() -> SubAgentRegistry:
    """테스트용 레지스트리 — coder, reviewer 두 에이전트 등록."""
    reg = SubAgentRegistry(cost_sensitivity=0.3)
    reg.register(
        SubAgentSpec(
            name="coder",
            description="코드 생성 에이전트",
            capabilities=["code_generation"],
            cost_weight=1.0,
        )
    )
    reg.register(
        SubAgentSpec(
            name="reviewer",
            description="코드 리뷰 에이전트",
            capabilities=["code_review"],
            cost_weight=0.5,
        )
    )
    return reg


# ── SubAgent Lifecycle ──


class TestSubAgentLifecycle:
    """8단계 상태 머신 전이 경로 테스트."""

    def test_full_lifecycle(self, registry: SubAgentRegistry):
        """정상 경로: created -> assigned -> running -> completed -> destroyed."""
        inst = registry.create_instance("coder", task_summary="피보나치 구현")
        assert inst is not None
        assert inst.state == SubAgentStatus.CREATED

        # CREATED -> ASSIGNED
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED, reason="작업 할당")
        assert ev is not None
        assert ev.from_state == SubAgentStatus.CREATED
        assert ev.to_state == SubAgentStatus.ASSIGNED

        # ASSIGNED -> RUNNING
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING, reason="실행 시작")
        assert ev is not None

        # RUNNING -> COMPLETED
        ev = registry.transition_state(
            inst.agent_id, SubAgentStatus.COMPLETED,
            reason="작업 완료", result_summary="피보나치 함수 구현 완료",
        )
        assert ev is not None

        # COMPLETED -> DESTROYED
        ev = registry.destroy_instance(inst.agent_id, reason="정리")
        assert ev is not None
        assert registry.get_instance(inst.agent_id).state == SubAgentStatus.DESTROYED

    def test_failure_retry_path(self, registry: SubAgentRegistry):
        """실패 후 재시도: running -> failed -> assigned -> running -> completed."""
        inst = registry.create_instance("coder", task_summary="리팩토링")
        assert inst is not None

        # CREATED -> ASSIGNED -> RUNNING
        registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING)

        # RUNNING -> FAILED
        ev = registry.transition_state(
            inst.agent_id, SubAgentStatus.FAILED,
            reason="API 호출 실패", error_message="timeout",
        )
        assert ev is not None
        assert registry.get_instance(inst.agent_id).error_message == "timeout"

        # FAILED -> ASSIGNED (재시도)
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED, reason="재시도")
        assert ev is not None

        # ASSIGNED -> RUNNING -> COMPLETED
        registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING)
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.COMPLETED, reason="재시도 성공")
        assert ev is not None

    def test_blocked_then_resume(self, registry: SubAgentRegistry):
        """차단 후 재개: running -> blocked -> running -> completed."""
        inst = registry.create_instance("coder", task_summary="의존성 대기")
        assert inst is not None

        registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING)

        # RUNNING -> BLOCKED
        ev = registry.transition_state(
            inst.agent_id, SubAgentStatus.BLOCKED, reason="의존 작업 미완료",
        )
        assert ev is not None

        # BLOCKED -> RUNNING (재개)
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING, reason="의존 작업 완료")
        assert ev is not None

        # RUNNING -> COMPLETED
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.COMPLETED)
        assert ev is not None

    def test_cancel_from_running(self, registry: SubAgentRegistry):
        """실행 중 취소: running -> cancelled -> destroyed."""
        inst = registry.create_instance("coder", task_summary="취소 테스트")
        assert inst is not None

        registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING)

        # RUNNING -> CANCELLED
        ev = registry.transition_state(
            inst.agent_id, SubAgentStatus.CANCELLED, reason="사용자 취소",
        )
        assert ev is not None

        # CANCELLED -> DESTROYED
        ev = registry.destroy_instance(inst.agent_id)
        assert ev is not None

    def test_invalid_transition_returns_none(self, registry: SubAgentRegistry):
        """잘못된 전이 시 None을 반환해야 한다."""
        inst = registry.create_instance("coder", task_summary="잘못된 전이 테스트")
        assert inst is not None
        # CREATED -> COMPLETED 는 유효하지 않은 전이
        ev = registry.transition_state(inst.agent_id, SubAgentStatus.COMPLETED)
        assert ev is None
        # 상태 변경 없음 확인
        assert registry.get_instance(inst.agent_id).state == SubAgentStatus.CREATED


# ── SubAgentInstance ──


class TestSubAgentInstance:
    """SubAgentInstance 메타데이터 테스트."""

    def test_instance_metadata(self, registry: SubAgentRegistry):
        """인스턴스 생성 시 agent_id, created_at, updated_at이 존재해야 한다."""
        inst = registry.create_instance("coder", task_summary="메타데이터 검증")
        assert inst is not None
        assert inst.agent_id  # 비어있지 않음
        assert isinstance(inst.created_at, datetime)
        assert isinstance(inst.updated_at, datetime)

    def test_instance_parent_tracking(self, registry: SubAgentRegistry):
        """parent_id 설정이 올바르게 동작하는지 확인."""
        parent = registry.create_instance("coder", task_summary="부모 작업")
        assert parent is not None

        child = registry.create_instance(
            "reviewer",
            task_summary="자식 작업",
            parent_id=parent.agent_id,
        )
        assert child is not None
        assert child.parent_id == parent.agent_id

        # parent_id로 필터링
        children = registry.list_instances(parent_id=parent.agent_id)
        assert len(children) == 1
        assert children[0].agent_id == child.agent_id


# ── SubAgent Cleanup ──


class TestSubAgentCleanup:
    """완료된 인스턴스 정리 테스트."""

    def test_cleanup_completed_instances(self, registry: SubAgentRegistry):
        """max_age 지난 완료 인스턴스가 정리되어야 한다."""
        inst = registry.create_instance("coder", task_summary="정리 대상")
        assert inst is not None

        registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING)
        registry.transition_state(inst.agent_id, SubAgentStatus.COMPLETED)

        # updated_at을 과거로 조작하여 max_age 초과 시뮬레이션
        completed_inst = registry.get_instance(inst.agent_id)
        assert completed_inst is not None
        completed_inst.updated_at = datetime.now(timezone.utc) - timedelta(seconds=600)

        cleaned = registry.cleanup_completed(max_age_seconds=300)
        assert cleaned == 1
        # 정리된 인스턴스는 더 이상 조회 불가
        assert registry.get_instance(inst.agent_id) is None

    def test_active_instances_property(self, registry: SubAgentRegistry):
        """active_instances가 활성 상태(CREATED/ASSIGNED/RUNNING/BLOCKED)만 반환."""
        # 활성 인스턴스 생성
        active = registry.create_instance("coder", task_summary="활성 작업")
        assert active is not None
        registry.transition_state(active.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(active.agent_id, SubAgentStatus.RUNNING)

        # 완료 인스턴스 생성
        done = registry.create_instance("reviewer", task_summary="완료 작업")
        assert done is not None
        registry.transition_state(done.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(done.agent_id, SubAgentStatus.RUNNING)
        registry.transition_state(done.agent_id, SubAgentStatus.COMPLETED)

        active_list = registry.active_instances
        assert len(active_list) == 1
        assert active_list[0].agent_id == active.agent_id


# ── Event Log ──


class TestEventLog:
    """상태 전이 이벤트 로깅 테스트."""

    def test_event_logged_on_transition(self, registry: SubAgentRegistry):
        """전이마다 이벤트가 기록되어야 한다."""
        inst = registry.create_instance("coder", task_summary="이벤트 테스트")
        assert inst is not None

        registry.transition_state(inst.agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(inst.agent_id, SubAgentStatus.RUNNING)
        registry.transition_state(inst.agent_id, SubAgentStatus.COMPLETED)

        events = registry.event_log
        assert len(events) == 3
        assert events[0].from_state == SubAgentStatus.CREATED
        assert events[0].to_state == SubAgentStatus.ASSIGNED
        assert events[1].from_state == SubAgentStatus.ASSIGNED
        assert events[1].to_state == SubAgentStatus.RUNNING
        assert events[2].from_state == SubAgentStatus.RUNNING
        assert events[2].to_state == SubAgentStatus.COMPLETED

    def test_event_log_contains_reason(self, registry: SubAgentRegistry):
        """이벤트에 reason 필드가 포함되어야 한다."""
        inst = registry.create_instance("coder", task_summary="reason 테스트")
        assert inst is not None

        registry.transition_state(
            inst.agent_id, SubAgentStatus.ASSIGNED, reason="스케줄러에 의한 할당",
        )
        events = registry.event_log
        assert len(events) == 1
        assert events[0].reason == "스케줄러에 의한 할당"
        assert isinstance(events[0].timestamp, datetime)
