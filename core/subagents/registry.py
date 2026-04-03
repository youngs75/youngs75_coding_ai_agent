"""동적 SubAgent 레지스트리.

Puppeteer 패턴:
  R = r(quality) - λ·C(cost)
  quality: 해당 에이전트의 유사 태스크 성공률
  cost: cost_weight (레이턴시 + 토큰 비용 대리)

사용 통계를 추적하여 선택 품질을 지속적으로 개선한다.
"""

from __future__ import annotations

from collections import defaultdict

from youngs75_a2a.core.subagents.schemas import (
    SelectionResult,
    SubAgentSpec,
    SubAgentStatus,
    SubAgentUsageRecord,
)


class SubAgentRegistry:
    """서브에이전트 등록, 동적 선택, 사용 통계 추적."""

    def __init__(self, cost_sensitivity: float = 0.3):
        """
        Args:
            cost_sensitivity: 비용 민감도 λ (0 → 품질만, 1 → 비용 위주)
        """
        self._agents: dict[str, SubAgentSpec] = {}
        self._usage: list[SubAgentUsageRecord] = []
        self._lambda = cost_sensitivity

    def register(self, spec: SubAgentSpec) -> None:
        """서브에이전트 등록."""
        self._agents[spec.name] = spec

    def unregister(self, name: str) -> bool:
        """서브에이전트 제거."""
        return self._agents.pop(name, None) is not None

    def get(self, name: str) -> SubAgentSpec | None:
        return self._agents.get(name)

    def list_available(self) -> list[SubAgentSpec]:
        """사용 가능한 에이전트 목록."""
        return [a for a in self._agents.values() if a.status == SubAgentStatus.AVAILABLE]

    def select(self, task_type: str, required_capabilities: list[str] | None = None) -> SelectionResult | None:
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
        """사용 기록 추가."""
        self._usage.append(record)

    def get_success_rate(self, agent_name: str, task_type: str | None = None) -> float:
        """에이전트의 성공률 계산."""
        relevant = [
            u for u in self._usage
            if u.agent_name == agent_name
            and (task_type is None or u.task_type == task_type)
        ]
        if not relevant:
            return 0.5  # 기록 없으면 중립값
        return sum(1 for u in relevant if u.success) / len(relevant)

    @property
    def usage_stats(self) -> dict[str, dict[str, int]]:
        """에이전트별 사용 통계."""
        stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0, "fail": 0})
        for u in self._usage:
            stats[u.agent_name]["total"] += 1
            if u.success:
                stats[u.agent_name]["success"] += 1
            else:
                stats[u.agent_name]["fail"] += 1
        return dict(stats)

    def _filter_candidates(self, required_capabilities: list[str] | None) -> list[SubAgentSpec]:
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
