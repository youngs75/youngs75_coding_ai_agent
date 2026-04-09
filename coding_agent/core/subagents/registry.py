"""동적 SubAgent 레지스트리.

Puppeteer 패턴:
  R = r(quality) - λ·C(cost)
  quality: 해당 에이전트의 유사 태스크 성공률
  cost: cost_weight (레이턴시 + 토큰 비용 대리)

사용 통계를 추적하여 선택 품질을 지속적으로 개선한다.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

from coding_agent.core.subagents.schemas import (
    VALID_TRANSITIONS,
    SelectionResult,
    SubAgentEvent,
    SubAgentInstance,
    SubAgentSpec,
    SubAgentStatus,
    SubAgentUsageRecord,
)

logger = logging.getLogger(__name__)

# "사용 가능" 상태 집합 — 하위호환을 위해 AVAILABLE + CREATED 모두 포함
_AVAILABLE_STATES = {SubAgentStatus.AVAILABLE, SubAgentStatus.CREATED}

# 활성 상태 집합
_ACTIVE_STATES = {
    SubAgentStatus.CREATED,
    SubAgentStatus.ASSIGNED,
    SubAgentStatus.RUNNING,
    SubAgentStatus.BLOCKED,
}


class SubAgentRegistry:
    """서브에이전트 등록, 동적 선택, 사용 통계 추적, 인스턴스 수명주기 관리.

    Puppeteer 패턴에 따라 R = r(quality) - lambda * C(cost) 공식으로
    태스크에 최적인 에이전트를 동적 선택한다. 사용 통계를 추적하여
    선택 품질을 지속적으로 개선한다.

    Args:
        cost_sensitivity: 비용 민감도 lambda (0이면 품질만, 1이면 비용 위주).
    """

    def __init__(self, cost_sensitivity: float = 0.3):
        self._agents: dict[str, SubAgentSpec] = {}
        self._usage: list[SubAgentUsageRecord] = []
        self._lambda = cost_sensitivity

        # ── 인스턴스 수명주기 저장소 ──
        self._instances: dict[str, SubAgentInstance] = {}
        self._events: list[SubAgentEvent] = []

    def register(self, spec: SubAgentSpec) -> None:
        """서브에이전트를 레지스트리에 등록한다.

        Args:
            spec: 등록할 에이전트 사양.
        """
        self._agents[spec.name] = spec

    def unregister(self, name: str) -> bool:
        """서브에이전트를 레지스트리에서 제거한다.

        Args:
            name: 제거할 에이전트 이름.

        Returns:
            제거 성공 시 True, 미등록 시 False.
        """
        return self._agents.pop(name, None) is not None

    def get(self, name: str) -> SubAgentSpec | None:
        """이름으로 에이전트 사양을 조회한다.

        Args:
            name: 에이전트 이름.

        Returns:
            SubAgentSpec 또는 None (미등록 시).
        """
        return self._agents.get(name)

    def list_available(self) -> list[SubAgentSpec]:
        """사용 가능한(CREATED/AVAILABLE 상태) 에이전트 목록을 반환한다.

        Returns:
            사용 가능한 SubAgentSpec 리스트.
        """
        return [
            a for a in self._agents.values() if a.status in _AVAILABLE_STATES
        ]

    def select(
        self, task_type: str, required_capabilities: list[str] | None = None
    ) -> SelectionResult | None:
        """태스크에 최적인 에이전트를 동적 선택한다.

        R = r(quality) - λ·C(cost)

        Args:
            task_type: 태스크 유형 (예: "code_generation", "review")
            required_capabilities: 필수 능력 목록
        """
        candidates = self._filter_candidates(required_capabilities)
        if not candidates:
            return None

        best: SelectionResult | None = None
        for agent in candidates:
            quality = self._compute_quality(agent.name, task_type)
            cost = agent.cost_weight
            score = quality - self._lambda * cost
            reason = f"quality={quality:.2f}, cost={cost:.2f}, λ={self._lambda}"

            if best is None or score > best.score:
                best = SelectionResult(agent=agent, score=score, reason=reason)

        return best

    def record_usage(self, record: SubAgentUsageRecord) -> None:
        """사용 기록을 추가한다.

        Args:
            record: 추가할 사용 기록.
        """
        self._usage.append(record)

    def get_success_rate(self, agent_name: str, task_type: str | None = None) -> float:
        """에이전트의 성공률을 계산한다.

        Args:
            agent_name: 에이전트 이름.
            task_type: 태스크 유형 필터. None이면 전체 성공률.

        Returns:
            성공률 (0.0~1.0). 기록 없으면 0.5 (중립값).
        """
        relevant = [
            u
            for u in self._usage
            if u.agent_name == agent_name
            and (task_type is None or u.task_type == task_type)
        ]
        if not relevant:
            return 0.5  # 기록 없으면 중립값
        return sum(1 for u in relevant if u.success) / len(relevant)

    def get_failure_reasons(self, agent_name: str) -> dict[str, int]:
        """에이전트의 실패 원인별 발생 횟수를 반환한다.

        Args:
            agent_name: 에이전트 이름.

        Returns:
            실패 원인 문자열 → 발생 횟수 매핑.
        """
        reasons: dict[str, int] = defaultdict(int)
        for u in self._usage:
            if u.agent_name == agent_name and not u.success and u.failure_reason:
                reasons[u.failure_reason] += 1
        return dict(reasons)

    @property
    def usage_stats(self) -> dict[str, dict[str, int]]:
        """에이전트별 사용 통계."""
        stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "success": 0, "fail": 0}
        )
        for u in self._usage:
            stats[u.agent_name]["total"] += 1
            if u.success:
                stats[u.agent_name]["success"] += 1
            else:
                stats[u.agent_name]["fail"] += 1
        return dict(stats)

    # ── 인스턴스 수명주기 관리 ──

    def create_instance(
        self,
        spec_name: str,
        *,
        task_summary: str = "",
        role: str = "",
        parent_id: str | None = None,
    ) -> SubAgentInstance | None:
        """SubAgentSpec 기반으로 런타임 인스턴스를 생성한다.

        Args:
            spec_name: 참조할 SubAgentSpec 이름.
            task_summary: 할당된 작업 요약.
            role: 역할/전문성 설명.
            parent_id: 부모 에이전트/오케스트레이터 ID.

        Returns:
            생성된 SubAgentInstance 또는 None (spec 미등록 시).
        """
        if spec_name not in self._agents:
            return None

        instance = SubAgentInstance(
            spec_name=spec_name,
            role=role,
            task_summary=task_summary,
            parent_id=parent_id,
        )
        self._instances[instance.agent_id] = instance
        logger.info(
            "[SubAgent] %s CREATED (spec=%s, task=%s)",
            instance.agent_id[:8],
            spec_name,
            task_summary[:50] if task_summary else "-",
        )
        return instance

    def transition_state(
        self,
        agent_id: str,
        new_state: SubAgentStatus,
        *,
        reason: str = "",
        error_message: str | None = None,
        result_summary: str | None = None,
    ) -> SubAgentEvent | None:
        """인스턴스 상태를 전이한다.

        VALID_TRANSITIONS에 정의된 유효한 전이만 허용한다.

        Args:
            agent_id: 전이할 에이전트 인스턴스 ID.
            new_state: 전이할 목표 상태.
            reason: 전이 사유.
            error_message: 실패 시 에러 메시지.
            result_summary: 완료 시 결과 요약.

        Returns:
            SubAgentEvent 또는 None (인스턴스 미존재 또는 유효하지 않은 전이 시).
        """
        instance = self._instances.get(agent_id)
        if instance is None:
            return None

        # 유효한 전이인지 검사
        allowed = VALID_TRANSITIONS.get(instance.state, set())
        if new_state not in allowed:
            logger.warning(
                "[SubAgent] %s 잘못된 전이: %s → %s",
                agent_id[:8],
                instance.state.value,
                new_state.value,
            )
            return None

        from_state = instance.state

        # 인스턴스 상태 갱신
        instance.state = new_state
        instance.updated_at = datetime.now(timezone.utc)
        if error_message is not None:
            instance.error_message = error_message
        if result_summary is not None:
            instance.result_summary = result_summary

        # 이벤트 생성 및 기록
        event = SubAgentEvent(
            agent_id=agent_id,
            from_state=from_state,
            to_state=new_state,
            reason=reason,
        )
        self._events.append(event)

        logger.info(
            "[SubAgent] %s %s → %s: %s",
            agent_id[:8],
            from_state.value,
            new_state.value,
            reason or "-",
        )

        return event

    def get_instance(self, agent_id: str) -> SubAgentInstance | None:
        """인스턴스를 조회한다.

        Args:
            agent_id: 조회할 에이전트 인스턴스 ID.

        Returns:
            SubAgentInstance 또는 None (미존재 시).
        """
        return self._instances.get(agent_id)

    def list_instances(
        self,
        *,
        state: SubAgentStatus | None = None,
        parent_id: str | None = None,
    ) -> list[SubAgentInstance]:
        """인스턴스 목록을 반환한다.

        Args:
            state: 상태 필터. None이면 전체.
            parent_id: 부모 ID 필터. None이면 전체.

        Returns:
            필터 조건에 맞는 SubAgentInstance 리스트.
        """
        result = list(self._instances.values())
        if state is not None:
            result = [i for i in result if i.state == state]
        if parent_id is not None:
            result = [i for i in result if i.parent_id == parent_id]
        return result

    def destroy_instance(
        self, agent_id: str, *, reason: str = "cleanup"
    ) -> SubAgentEvent | None:
        """인스턴스를 DESTROYED로 전이하고 정리한다.

        Args:
            agent_id: 파괴할 에이전트 인스턴스 ID.
            reason: 파괴 사유.

        Returns:
            SubAgentEvent 또는 None (전이 불가 시).
        """
        return self.transition_state(
            agent_id,
            SubAgentStatus.DESTROYED,
            reason=reason,
        )

    def cleanup_completed(self, *, max_age_seconds: float = 300) -> int:
        """완료/실패/취소된 인스턴스 중 max_age_seconds 지난 것을 정리한다.

        Args:
            max_age_seconds: 정리 대상 최소 경과 시간(초).

        Returns:
            정리된 인스턴스 수.
        """
        terminal_states = {
            SubAgentStatus.COMPLETED,
            SubAgentStatus.FAILED,
            SubAgentStatus.CANCELLED,
            SubAgentStatus.DESTROYED,
        }
        now = datetime.now(timezone.utc)
        to_remove: list[str] = []

        for agent_id, inst in self._instances.items():
            if inst.state not in terminal_states:
                continue
            elapsed = (now - inst.updated_at).total_seconds()
            if elapsed >= max_age_seconds:
                to_remove.append(agent_id)

        for agent_id in to_remove:
            del self._instances[agent_id]

        if to_remove:
            logger.info(
                "[SubAgent] cleanup_completed: %d개 인스턴스 정리됨",
                len(to_remove),
            )
        return len(to_remove)

    @property
    def event_log(self) -> list[SubAgentEvent]:
        """전체 상태 전이 이벤트 로그."""
        return list(self._events)

    @property
    def active_instances(self) -> list[SubAgentInstance]:
        """현재 활성(CREATED/ASSIGNED/RUNNING/BLOCKED) 인스턴스."""
        return [
            i for i in self._instances.values() if i.state in _ACTIVE_STATES
        ]

    # ── 내부 헬퍼 ──

    def _filter_candidates(
        self, required_capabilities: list[str] | None
    ) -> list[SubAgentSpec]:
        """필수 능력을 만족하는 사용 가능 에이전트 필터링."""
        available = self.list_available()
        if not required_capabilities:
            return available
        required_set = set(required_capabilities)
        return [a for a in available if required_set <= set(a.capabilities)]

    def _compute_quality(self, agent_name: str, task_type: str) -> float:
        """에이전트의 태스크 유형별 품질 점수."""
        # 특정 태스크 유형 성공률 (가중 70%) + 전체 성공률 (가중 30%)
        task_rate = self.get_success_rate(agent_name, task_type)
        overall_rate = self.get_success_rate(agent_name)
        return 0.7 * task_rate + 0.3 * overall_rate
