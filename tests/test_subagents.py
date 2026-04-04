"""동적 SubAgent 유닛 테스트."""

from __future__ import annotations

import pytest

from youngs75_a2a.core.subagents.registry import SubAgentRegistry
from youngs75_a2a.core.subagents.schemas import (
    SubAgentSpec,
    SubAgentStatus,
    SubAgentUsageRecord,
)


@pytest.fixture()
def registry() -> SubAgentRegistry:
    reg = SubAgentRegistry(cost_sensitivity=0.3)
    reg.register(
        SubAgentSpec(
            name="coder",
            description="코드 생성 에이전트",
            capabilities=["code_generation", "refactoring"],
            cost_weight=1.0,
        )
    )
    reg.register(
        SubAgentSpec(
            name="reviewer",
            description="코드 리뷰 에이전트",
            capabilities=["code_review", "quality"],
            cost_weight=0.5,
        )
    )
    reg.register(
        SubAgentSpec(
            name="researcher",
            description="리서치 에이전트",
            capabilities=["research", "search"],
            cost_weight=0.8,
        )
    )
    return reg


class TestSubAgentRegistry:
    def test_register_and_list(self, registry):
        available = registry.list_available()
        assert len(available) == 3

    def test_get(self, registry):
        agent = registry.get("coder")
        assert agent is not None
        assert agent.name == "coder"

    def test_unregister(self, registry):
        assert registry.unregister("coder")
        assert registry.get("coder") is None
        assert not registry.unregister("nonexistent")

    def test_select_by_task_type(self, registry):
        # 기록이 없으면 모두 0.5 품질, 비용으로 선택
        result = registry.select("code_generation")
        assert result is not None
        # 비용이 낮은 reviewer가 선택될 수 있음 (0.5 - 0.3*0.5 = 0.35 vs 0.5 - 0.3*1.0 = 0.2)
        assert result.agent.name == "reviewer"

    def test_select_with_capabilities(self, registry):
        result = registry.select(
            "code_generation", required_capabilities=["code_generation"]
        )
        assert result is not None
        assert result.agent.name == "coder"  # coder만 code_generation 능력 보유

    def test_select_no_matching_capabilities(self, registry):
        result = registry.select(
            "task", required_capabilities=["nonexistent_capability"]
        )
        assert result is None

    def test_record_usage_and_stats(self, registry):
        registry.record_usage(
            SubAgentUsageRecord(agent_name="coder", task_type="gen", success=True)
        )
        registry.record_usage(
            SubAgentUsageRecord(agent_name="coder", task_type="gen", success=True)
        )
        registry.record_usage(
            SubAgentUsageRecord(agent_name="coder", task_type="gen", success=False)
        )

        stats = registry.usage_stats
        assert stats["coder"]["total"] == 3
        assert stats["coder"]["success"] == 2
        assert stats["coder"]["fail"] == 1

    def test_success_rate(self, registry):
        registry.record_usage(
            SubAgentUsageRecord(agent_name="coder", task_type="gen", success=True)
        )
        registry.record_usage(
            SubAgentUsageRecord(agent_name="coder", task_type="gen", success=False)
        )
        assert registry.get_success_rate("coder", "gen") == 0.5
        assert registry.get_success_rate("coder") == 0.5

    def test_success_rate_no_records(self, registry):
        # 기록 없으면 중립값 0.5
        assert registry.get_success_rate("coder") == 0.5

    def test_quality_improves_selection(self, registry):
        # coder에 좋은 기록을 쌓으면 선택됨
        for _ in range(10):
            registry.record_usage(
                SubAgentUsageRecord(agent_name="coder", task_type="gen", success=True)
            )
        for _ in range(10):
            registry.record_usage(
                SubAgentUsageRecord(
                    agent_name="reviewer", task_type="gen", success=False
                )
            )

        result = registry.select("gen")
        assert result is not None
        assert result.agent.name == "coder"

    def test_disabled_agent_not_selected(self, registry):
        disabled = registry.get("researcher")
        assert disabled is not None
        registry.register(
            SubAgentSpec(
                name="researcher",
                description="disabled",
                capabilities=["research"],
                status=SubAgentStatus.DISABLED,
            )
        )
        available = registry.list_available()
        assert all(a.name != "researcher" for a in available)


class TestSelectionResult:
    def test_selection_has_reason(self, registry):
        result = registry.select("task")
        assert result is not None
        assert "quality=" in result.reason
        assert "cost=" in result.reason
