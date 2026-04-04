"""A2A 프로토콜 통합 테스트.

디스커버리, 복원력, 라우터, 스트리밍, Executor를 포괄하는 단위/통합 테스트.
외부 서버 불필요 (모두 모킹/인메모리).

실행: python -m pytest tests/test_a2a_protocol.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a.types import AgentCard, AgentCapabilities, AgentSkill

# ── 팩토리 헬퍼 ──────────────────────────────────────────


def make_agent_card(
    name: str = "test-agent",
    url: str = "http://localhost:8080",
    skills: list[AgentSkill] | None = None,
    description: str = "테스트 에이전트",
) -> AgentCard:
    """테스트용 AgentCard ��성."""
    return AgentCard(
        name=name,
        description=description,
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        skills=skills or [],
    )


def make_skill(
    name: str = "code_generation",
    skill_id: str = "code-gen",
    description: str = "코드 생성",
    tags: list[str] | None = None,
) -> AgentSkill:
    """테스트용 AgentSkill 생성."""
    return AgentSkill(
        name=name,
        id=skill_id,
        description=description,
        tags=tags or [],
    )


# ══════════════════════════════════════════════════════════
# 1. AgentCardEntry 테스트
# ═════════════════════���════════════════════════════════════


class TestAgentCardEntry:
    """AgentCardEntry 헬스 상태 관리 테스트."""

    def test_mark_healthy(self):
        """mark_healthy() 호출 시 is_healthy == True."""
        from youngs75_a2a.a2a.discovery import AgentCardEntry

        entry = AgentCardEntry(card=make_agent_card(), url="http://localhost:8080")
        entry.mark_healthy()
        assert entry.is_healthy
        assert entry.consecutive_failures == 0

    def test_mark_failed_increments(self):
        """mark_failed() 호출 시 연속 실패 횟수 증가."""
        from youngs75_a2a.a2a.discovery import AgentCardEntry

        entry = AgentCardEntry(card=make_agent_card(), url="http://localhost:8080")
        entry.mark_failed()
        entry.mark_failed()
        assert entry.consecutive_failures == 2

    def test_unhealthy_after_timeout(self):
        """60초 경과 후 is_healthy == False."""
        from youngs75_a2a.a2a.discovery import AgentCardEntry

        entry = AgentCardEntry(card=make_agent_card(), url="http://localhost:8080")
        entry.last_healthy = time.time() - 61
        assert not entry.is_healthy


# ═════��═══════════════════════════���════════════════════��═══
# 2. AgentCardRegistry 테스트
# ════════���═════════════════════════════════════════════════


class TestAgentCardRegistry:
    """AgentCardRegistry 등록/검색/헬스체크 테스트."""

    def test_register_and_get(self):
        """에이전트 등록 후 URL로 조회."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(name="agent-1", url="http://localhost:8081")
        registry.register(card, url="http://localhost:8081")

        entry = registry.get("http://localhost:8081")
        assert entry is not None
        assert entry.card.name == "agent-1"

    def test_register_and_unregister(self):
        """등록 후 해제 시 조회 불가."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(url="http://localhost:8081")
        registry.register(card, url="http://localhost:8081")
        assert registry.unregister("http://localhost:8081")
        assert registry.get("http://localhost:8081") is None

    def test_get_by_name_exact(self):
        """이름 정확 매칭 조회."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(name="code-agent", url="http://localhost:8081")
        registry.register(card)
        assert registry.get_by_name("code-agent") is not None

    def test_get_by_name_partial(self):
        """이름 부분 매칭 조회."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(name="my-code-agent", url="http://localhost:8081")
        registry.register(card)
        assert registry.get_by_name("code") is not None

    def test_list_all(self):
        """모든 에이전트 목록."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        registry.register(make_agent_card(name="a1", url="http://localhost:8081"))
        registry.register(make_agent_card(name="a2", url="http://localhost:8082"))
        assert len(registry.list_all()) == 2

    def test_list_healthy(self):
        """건강한 에이전트만 반환."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        registry.register(make_agent_card(name="healthy", url="http://localhost:8081"))
        e2 = registry.register(
            make_agent_card(name="sick", url="http://localhost:8082")
        )
        e2.last_healthy = time.time() - 120  # 2분 전 → 불건강

        healthy = registry.list_healthy()
        assert len(healthy) == 1
        assert healthy[0].card.name == "healthy"

    def test_find_by_skill_exact(self):
        """스킬 이름 정확 매칭 검색."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(
            name="coder",
            url="http://localhost:8081",
            skills=[make_skill(name="code_generation")],
        )
        registry.register(card)

        results = registry.find_by_skill("code_generation")
        assert len(results) == 1
        assert results[0].match_score == 1.0

    def test_find_by_skill_partial(self):
        """스킬 이름 부분 매칭 검색."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(
            name="coder",
            url="http://localhost:8081",
            skills=[make_skill(name="code_generation")],
        )
        registry.register(card)

        results = registry.find_by_skill("code")
        assert len(results) == 1
        assert results[0].match_score == 0.8

    def test_find_by_skill_description(self):
        """스킬 설명 매칭 검색."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(
            name="helper",
            url="http://localhost:8081",
            skills=[make_skill(name="assist", description="코드 생성과 리뷰")],
        )
        registry.register(card)

        results = registry.find_by_skill("코드 생성")
        assert len(results) == 1

    def test_find_by_tags(self):
        """태그 기반 검��."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        card = make_agent_card(
            name="coder",
            url="http://localhost:8081",
            skills=[make_skill(tags=["python", "javascript", "review"])],
        )
        registry.register(card)

        results = registry.find_by_tags(["python", "java"])
        assert len(results) == 1
        assert "python" in results[0].matched_tags

    def test_find_by_tags_empty(self):
        """매칭되는 태그 없을 때 빈 결과."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        registry.register(
            make_agent_card(
                url="http://localhost:8081",
                skills=[make_skill(tags=["python"])],
            )
        )
        results = registry.find_by_tags(["rust"])
        assert len(results) == 0

    def test_find_by_skill_sorted_by_score(self):
        """스킬 검색 결과가 점수순 정렬."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        # 정확 매칭 에이전트 (점수 1.0)
        registry.register(
            make_agent_card(
                name="exact",
                url="http://localhost:8081",
                skills=[make_skill(name="search")],
            )
        )
        # 부분 매칭 에이전트 (점수 0.8)
        registry.register(
            make_agent_card(
                name="partial",
                url="http://localhost:8082",
                skills=[make_skill(name="web_search")],
            )
        )

        results = registry.find_by_skill("search")
        assert len(results) == 2
        assert results[0].entry.card.name == "exact"
        assert results[0].match_score > results[1].match_score

    async def test_discover_many_with_mock(self):
        """discover_many() 병렬 디스커버리 (모킹)."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry()
        mock_card = make_agent_card(name="remote-agent")

        with patch.object(registry, "discover") as mock_discover:
            from youngs75_a2a.a2a.discovery import AgentCardEntry

            mock_entry = AgentCardEntry(card=mock_card, url="http://remote:8080")
            mock_discover.return_value = mock_entry

            entries = await registry.discover_many(["http://remote:8080"])
            assert len(entries) == 1

    def test_periodic_health_check_start_stop(self):
        """주기적 헬스체크 시작/중지."""
        from youngs75_a2a.a2a.discovery import AgentCardRegistry

        registry = AgentCardRegistry(health_check_interval=0.1)

        # 이벤트 루프 없이는 시작 불가하므로 직접 확인
        assert registry._health_task is None
        registry.stop_periodic_health_check()  # 안전하게 호출 가능
        assert registry._health_task is None


# ══════════════════════════════════════════════════════════
# 3. RetryPolicy 테스트
# ══════════════════════════════════════════════════════════


class TestRetryPolicy:
    """RetryPolicy 지수 백오프 테스트."""

    def test_compute_delay_exponential(self):
        """지��� 백오프 대기 시간 계산."""
        from youngs75_a2a.a2a.resilience import RetryPolicy

        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0)
        assert policy.compute_delay(0) == 1.0
        assert policy.compute_delay(1) == 2.0
        assert policy.compute_delay(2) == 4.0
        assert policy.compute_delay(3) == 8.0

    def test_compute_delay_max_cap(self):
        """최대 대기 시간 제��."""
        from youngs75_a2a.a2a.resilience import RetryPolicy

        policy = RetryPolicy(base_delay=1.0, max_delay=5.0, exponential_base=2.0)
        assert policy.compute_delay(10) == 5.0

    def test_is_retryable(self):
        """재시도 가능한 예외 판별."""
        from youngs75_a2a.a2a.resilience import RetryPolicy

        policy = RetryPolicy()
        assert policy.is_retryable(ConnectionError())
        assert policy.is_retryable(asyncio.TimeoutError())
        assert not policy.is_retryable(ValueError())


# ════════════════════════════════��═════════════════════════
# 4. CircuitBreaker 테스트
# ══════════════════════════════════════════════════════════


class TestCircuitBreaker:
    """CircuitBreaker 상태 전이 테스트."""

    def test_initial_state_closed(self):
        """초기 상태 CLOSED."""
        from youngs75_a2a.a2a.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_open_after_threshold(self):
        """임계치 초과 후 OPEN."""
        from youngs75_a2a.a2a.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_half_open_after_recovery(self):
        """복구 시간 후 HALF_OPEN 전이."""
        from youngs75_a2a.a2a.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.0)
        cb.record_failure()
        cb.record_failure()
        # recovery_timeout=0 이므로 state 프로퍼티 접근 시 즉시 HALF_OPEN 전이
        # 내부 _state로 OPEN 확인
        assert cb._state == CircuitState.OPEN

        # state 프로퍼티가 시간 기반 자동 전이를 수행
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()

    def test_success_closes_half_open(self):
        """HALF_OPEN 상태에서 성공하면 CLOSED."""
        from youngs75_a2a.a2a.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failure_reopens_half_open(self):
        """HALF_OPEN 상태에서 실패하면 다��� OPEN."""
        from youngs75_a2a.a2a.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb._state == CircuitState.OPEN

    def test_reset(self):
        """reset() 호출 시 CLOSED로 초기화."""
        from youngs75_a2a.a2a.resilience import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0


# ═════���═══════════════════════════���════════════════════════
# 5. AgentMonitor 테스트
# ═════��═════════════════════════════════��══════════════════


class TestAgentMonitor:
    """AgentMonitor 통계 추적 테스트."""

    def test_record_success(self):
        """성공 기록 후 통계 반영."""
        from youngs75_a2a.a2a.resilience import AgentMonitor

        monitor = AgentMonitor()
        monitor.record_success("http://localhost:8080", 100.0)
        monitor.record_success("http://localhost:8080", 200.0)

        stats = monitor.get_stats("http://localhost:8080")
        assert stats.total_requests == 2
        assert stats.successful_requests == 2
        assert stats.avg_latency_ms == 150.0

    def test_record_failure(self):
        """실패 기록 후 통계 반영."""
        from youngs75_a2a.a2a.resilience import AgentMonitor

        monitor = AgentMonitor()
        monitor.record_success("http://localhost:8080", 100.0)
        monitor.record_failure("http://localhost:8080", "timeout")

        stats = monitor.get_stats("http://localhost:8080")
        assert stats.total_requests == 2
        assert stats.success_rate == 0.5
        assert stats.last_error == "timeout"

    def test_get_healthy_urls(self):
        """건강한 에이전트 URL 조회."""
        from youngs75_a2a.a2a.resilience import AgentMonitor

        monitor = AgentMonitor()
        monitor.record_success("http://a:8080", 100.0)
        monitor.record_failure("http://b:8080", "error")

        healthy = monitor.get_healthy_urls(min_success_rate=0.5)
        assert "http://a:8080" in healthy
        assert "http://b:8080" not in healthy

    def test_success_rate_no_requests(self):
        """요청 없을 때 성공률 1.0."""
        from youngs75_a2a.a2a.resilience import AgentHealthStats

        stats = AgentHealthStats(url="http://localhost:8080")
        assert stats.success_rate == 1.0


# ═══════��═════════════════════��════════════════════════════
# 6. TaskDelegator 테스트
# ════════��═════════════════════════════════════════════════


class TestTaskDelegator:
    """TaskDelegator 위임 로직 테스트."""

    async def test_delegate_success(self):
        """위임 성공 시 DelegationResult.success == True."""
        from youngs75_a2a.a2a.router import TaskDelegator

        delegator = TaskDelegator()
        mock_response = MagicMock()

        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await delegator.delegate(
                url="http://localhost:8080",
                content="hello",
                agent_name="test-agent",
            )
            assert result.success
            assert result.agent_name == "test-agent"
            assert result.latency_ms > 0

    async def test_delegate_failure(self):
        """위임 실패 시 DelegationResult.success == False."""
        from youngs75_a2a.a2a.router import TaskDelegator

        delegator = TaskDelegator()

        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            side_effect=ConnectionError("연결 실패"),
        ):
            result = await delegator.delegate(
                url="http://localhost:8080",
                content="hello",
                agent_name="test-agent",
            )
            assert not result.success
            assert "연결 실패" in result.error

    async def test_delegate_parallel(self):
        """병렬 위임 성공."""
        from youngs75_a2a.a2a.router import TaskDelegator

        delegator = TaskDelegator()
        mock_response = MagicMock()

        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            results = await delegator.delegate_parallel(
                targets=[
                    {"url": "http://a:8080", "name": "agent-a"},
                    {"url": "http://b:8080", "name": "agent-b"},
                ],
                content="hello",
            )
            assert len(results) == 2
            assert all(r.success for r in results)

    async def test_delegate_with_consensus_success(self):
        """합의 위임 — 최소 성공 충족."""
        from youngs75_a2a.a2a.router import TaskDelegator

        delegator = TaskDelegator()
        mock_response = MagicMock()

        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            results = await delegator.delegate_with_consensus(
                targets=[
                    {"url": "http://a:8080", "name": "a"},
                    {"url": "http://b:8080", "name": "b"},
                ],
                content="hello",
                min_success=1,
            )
            assert len(results) == 2

    async def test_delegate_with_consensus_failure(self):
        """합의 위임 — 최소 성공 미충족 시 RuntimeError."""
        from youngs75_a2a.a2a.router import TaskDelegator

        delegator = TaskDelegator()

        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            side_effect=ConnectionError("fail"),
        ):
            with pytest.raises(RuntimeError, match="합의 실패"):
                await delegator.delegate_with_consensus(
                    targets=[{"url": "http://a:8080", "name": "a"}],
                    content="hello",
                    min_success=1,
                )


# ════���═══════════════════════��═════════════════════════════
# 7. AgentRouter 라우팅 테스트
# ═══════════════════════════════════��══════════════════════


class TestAgentRouter:
    """AgentRouter 라우팅 의사결정 테스트."""

    def _make_router(self):
        from youngs75_a2a.a2a.router import AgentRouter, RoutingMode

        return AgentRouter(routing_mode=RoutingMode.SKILL_BASED)

    def test_route_skill_based(self):
        """스킬 기반 라우팅."""
        router = self._make_router()
        card = make_agent_card(
            name="coder",
            url="http://localhost:8081",
            skills=[make_skill(name="code_generation")],
        )
        router.register_agent(card)

        decision = router.route("code_generation")
        assert decision is not None
        assert decision.agent_name == "coder"
        assert decision.confidence > 0

    def test_route_with_tags(self):
        """태그 포함 스킬 기반 라우���."""
        router = self._make_router()
        card = make_agent_card(
            name="py-coder",
            url="http://localhost:8081",
            skills=[make_skill(name="coding", tags=["python", "testing"])],
        )
        router.register_agent(card)

        decision = router.route("coding", required_tags=["python"])
        assert decision is not None
        assert decision.agent_name == "py-coder"

    def test_route_no_match_fallback(self):
        """매칭 없을 때 기본 에이전트 폴백."""
        router = self._make_router()
        card = make_agent_card(name="generic", url="http://localhost:8081")
        router.register_agent(card)

        decision = router.route("unknown_skill")
        assert decision is not None
        assert decision.confidence == 0.1  # 폴백 점수

    def test_route_no_agents(self):
        """등록된 에이전트 없을 때 None."""
        router = self._make_router()
        decision = router.route("anything")
        assert decision is None

    def test_route_round_robin(self):
        """라운드 로빈 라우팅."""
        from youngs75_a2a.a2a.router import AgentRouter, RoutingMode

        router = AgentRouter(routing_mode=RoutingMode.ROUND_ROBIN)
        router.register_agent(make_agent_card(name="a1", url="http://a:8081"))
        router.register_agent(make_agent_card(name="a2", url="http://a:8082"))

        names = set()
        for _ in range(4):
            decision = router.route("query")
            assert decision is not None
            names.add(decision.agent_name)
        # 두 에이전트 모두 선택되어야 함
        assert len(names) == 2

    def test_route_weighted(self):
        """가중치 기반 라우팅."""
        from youngs75_a2a.a2a.router import AgentRouter, RoutingMode

        router = AgentRouter(routing_mode=RoutingMode.WEIGHTED)
        router.register_agent(make_agent_card(name="fast", url="http://fast:8080"))
        router.register_agent(make_agent_card(name="slow", url="http://slow:8080"))

        # fast 에이전트에 좋은 통계
        router.delegator.monitor.record_success("http://fast:8080", 50.0)
        router.delegator.monitor.record_success("http://fast:8080", 60.0)
        # slow 에이전트에 나쁜 통계
        router.delegator.monitor.record_failure("http://slow:8080", "timeout")

        decision = router.route("query")
        assert decision is not None
        assert decision.agent_name == "fast"

    def test_alternatives_in_decision(self):
        """라우팅 결정에 대안 에이전트 포함."""
        router = self._make_router()
        router.register_agent(
            make_agent_card(
                name="best",
                url="http://best:8080",
                skills=[make_skill(name="search")],
            )
        )
        router.register_agent(
            make_agent_card(
                name="alt",
                url="http://alt:8080",
                skills=[make_skill(name="web_search")],
            )
        )

        decision = router.route("search")
        assert decision is not None
        assert decision.agent_name == "best"
        assert "alt" in decision.alternatives

    async def test_route_and_delegate(self):
        """통합: 라우팅 + 위임."""
        router = self._make_router()
        router.register_agent(
            make_agent_card(
                name="coder",
                url="http://coder:8080",
                skills=[make_skill(name="code")],
            )
        )

        mock_response = MagicMock()
        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await router.route_and_delegate("code")
            assert result.success
            assert result.agent_name == "coder"

    async def test_route_and_delegate_no_agent(self):
        """통합: 에이전트 없을 때 RuntimeError."""
        router = self._make_router()
        with pytest.raises(RuntimeError, match="라우팅 대상"):
            await router.route_and_delegate("query")

    async def test_broadcast(self):
        """브로드캐스트 병렬 위임."""
        router = self._make_router()
        router.register_agent(
            make_agent_card(
                name="a1",
                url="http://a1:8080",
                skills=[make_skill(name="search")],
            )
        )
        router.register_agent(
            make_agent_card(
                name="a2",
                url="http://a2:8080",
                skills=[make_skill(name="deep_search")],
            )
        )

        mock_response = MagicMock()
        with patch(
            "youngs75_a2a.a2a.resilience.ResilientA2AClient.send_message",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            results = await router.broadcast("search", max_agents=3)
            assert len(results) == 2
            assert all(r.success for r in results)


# ══��═══════════════════════════════════════════════════════
# 8. Executor 타임아웃 테스트
# ═══════���══════════════════════���═══════════════════════���═══


class TestExecutorTimeout:
    """BaseAgentExecutor/LGAgentExecutor 실행 타임아웃 테스트."""

    async def test_base_executor_timeout(self):
        """BaseAgentExecutor — 타임아웃 시 failed 상태."""
        from youngs75_a2a.a2a.executor import BaseAgentExecutor

        async def slow_agent(query, ctx):
            await asyncio.sleep(10)
            return "done"

        executor = BaseAgentExecutor(slow_agent, execution_timeout=0.1)

        # RequestContext 모킹
        context = MagicMock()
        context.get_user_input.return_value = "test"
        context.current_task = None
        mock_message = MagicMock()
        mock_message.parts = []
        context.message = mock_message

        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()

        # TaskUpdater를 모킹하여 failed 호출 확인
        with patch("youngs75_a2a.a2a.executor.TaskUpdater") as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.update_status = AsyncMock()
            mock_updater.start_work = AsyncMock()
            mock_updater.failed = AsyncMock()
            MockUpdater.return_value = mock_updater

            with patch("youngs75_a2a.a2a.executor.new_task") as mock_new_task:
                mock_task = MagicMock()
                mock_task.id = "task-1"
                mock_task.context_id = "ctx-1"
                mock_new_task.return_value = mock_task

                await executor.execute(context, event_queue)

                # 타임아웃 시 failed 호출 확인
                mock_updater.failed.assert_called_once()

    async def test_base_executor_no_timeout(self):
        """BaseAgentExecutor — 타임아웃 None이면 무제한 대기."""
        from youngs75_a2a.a2a.executor import BaseAgentExecutor

        async def fast_agent(query, ctx):
            return "fast result"

        executor = BaseAgentExecutor(fast_agent, execution_timeout=None)

        context = MagicMock()
        context.get_user_input.return_value = "test"
        context.current_task = None
        mock_message = MagicMock()
        mock_message.parts = []
        context.message = mock_message

        event_queue = MagicMock()
        event_queue.enqueue_event = AsyncMock()

        with patch("youngs75_a2a.a2a.executor.TaskUpdater") as MockUpdater:
            mock_updater = MagicMock()
            mock_updater.update_status = AsyncMock()
            mock_updater.start_work = AsyncMock()
            mock_updater.add_artifact = AsyncMock()
            mock_updater.complete = AsyncMock()
            MockUpdater.return_value = mock_updater

            with patch("youngs75_a2a.a2a.executor.new_task") as mock_new_task:
                mock_task = MagicMock()
                mock_task.id = "task-1"
                mock_task.context_id = "ctx-1"
                mock_new_task.return_value = mock_task

                await executor.execute(context, event_queue)

                # 정상 완료 시 complete 호출
                mock_updater.complete.assert_called_once()


# ═════���════════════════════════════��═══════════════════════
# 9. StreamingResponseCollector 테스트
# ════���══════════════════════════════��══════════════════════


class TestStreamChunk:
    """StreamChunk 데이터 클래스 테스트."""

    def test_stream_chunk_fields(self):
        """StreamChunk 필드 초기화."""
        from youngs75_a2a.a2a.streaming import StreamChunk

        chunk = StreamChunk(
            text="hello",
            accumulated="hello",
            state="working",
            index=0,
            elapsed_ms=10.5,
        )
        assert chunk.text == "hello"
        assert chunk.accumulated == "hello"
        assert chunk.index == 0


class TestStreamingResponseCollector:
    """StreamingResponseCollector 테스트."""

    def test_initial_state(self):
        """초기 상태 확인."""
        from youngs75_a2a.a2a.streaming import StreamingResponseCollector

        collector = StreamingResponseCollector()
        assert collector.accumulated_text == ""
        assert collector.chunks == []
        assert collector.get_final_text() == ""

    def test_final_text_priority(self):
        """_final_text가 있으면 그것을 반환."""
        from youngs75_a2a.a2a.streaming import StreamingResponseCollector

        collector = StreamingResponseCollector()
        collector._accumulated_text = "partial"
        collector._final_text = "complete result"
        assert collector.get_final_text() == "complete result"


# ══════════════════════════════════════════════════════════
# 10. 통합 import 테스트
# ═════════��════════════════════════════════════════════════


class TestA2AImports:
    """모든 A2A 모듈 import 검증."""

    def test_all_exports(self):
        """__init__.py의 __all__ 항목이 모두 import 가능."""
        import youngs75_a2a.a2a as a2a_mod

        for name in a2a_mod.__all__:
            assert hasattr(a2a_mod, name), f"{name}이(가) a2a 모듈에 없음"

    def test_executor_classes(self):
        """Executor 클래스 import."""
        from youngs75_a2a.a2a import BaseAgentExecutor, LGAgentExecutor

        assert BaseAgentExecutor is not None
        assert LGAgentExecutor is not None

    def test_discovery_classes(self):
        """Discovery 클래스 import."""
        from youngs75_a2a.a2a import AgentCardRegistry, AgentCardEntry, DiscoveryResult

        assert AgentCardRegistry is not None
        assert AgentCardEntry is not None
        assert DiscoveryResult is not None

    def test_resilience_classes(self):
        """Resilience 클래스 import."""
        from youngs75_a2a.a2a import (
            RetryPolicy,
            CircuitBreaker,
        )

        assert RetryPolicy is not None
        assert CircuitBreaker is not None

    def test_router_classes(self):
        """Router 클래스 import."""
        from youngs75_a2a.a2a import (
            AgentRouter,
            RoutingMode,
        )

        assert AgentRouter is not None
        assert RoutingMode is not None

    def test_streaming_classes(self):
        """Streaming 클래스 import."""
        from youngs75_a2a.a2a import (
            StreamingResponseCollector,
            StreamChunk,
            stream_agent_response,
        )

        assert StreamingResponseCollector is not None
        assert StreamChunk is not None
        assert stream_agent_response is not None

    def test_server_functions(self):
        """서버 조립 함수 import 및 호출 가능."""
        from youngs75_a2a.a2a import create_agent_card

        card = create_agent_card(name="test", url="http://localhost:9999")
        assert card.name == "test"
        assert card.capabilities.streaming is True


# ══���═════════════════════════���═════════════════════════════
# 11. 결과 병합 유틸리티 테스트
# ═════════════════════��════════════════════════════════════


class TestMergeResults:
    """AgentRouter._merge_results 테스트."""

    def test_merge_skill_and_tag_results(self):
        """스킬 + 태그 결과 병합."""
        from youngs75_a2a.a2a.discovery import AgentCardEntry, DiscoveryResult
        from youngs75_a2a.a2a.router import AgentRouter

        card = make_agent_card(name="agent-1", url="http://a:8080")
        entry = AgentCardEntry(card=card, url="http://a:8080")
        entry.mark_healthy()

        skill_results = [
            DiscoveryResult(entry=entry, match_score=0.8, matched_skills=["code"])
        ]
        tag_results = [
            DiscoveryResult(entry=entry, match_score=0.6, matched_tags=["python"])
        ]

        merged = AgentRouter._merge_results(skill_results, tag_results)
        assert len(merged) == 1
        # 스킬 0.8 + 태그 보너스 0.6 * 0.2 = 0.92
        assert merged[0].match_score == pytest.approx(0.92, abs=0.01)

    def test_merge_different_agents(self):
        """서로 다른 에이전트 결과 병합."""
        from youngs75_a2a.a2a.discovery import AgentCardEntry, DiscoveryResult
        from youngs75_a2a.a2a.router import AgentRouter

        card_a = make_agent_card(name="a", url="http://a:8080")
        card_b = make_agent_card(name="b", url="http://b:8080")
        entry_a = AgentCardEntry(card=card_a, url="http://a:8080")
        entry_b = AgentCardEntry(card=card_b, url="http://b:8080")
        entry_a.mark_healthy()
        entry_b.mark_healthy()

        skill_results = [DiscoveryResult(entry=entry_a, match_score=0.8)]
        tag_results = [
            DiscoveryResult(entry=entry_b, match_score=0.6, matched_tags=["py"])
        ]

        merged = AgentRouter._merge_results(skill_results, tag_results)
        assert len(merged) == 2
        # 스킬만 매칭된 에이전트가 점수 높음
        assert merged[0].entry.url == "http://a:8080"


# ═════════════════════════════════���════════════════════════
# 12. CircuitOpenError 테스트
# ══��═══════════════════════════════════════════════════════


class TestCircuitOpenError:
    """CircuitOpenError 예외 테스트."""

    def test_error_message(self):
        """에러 메시지에 URL 포함."""
        from youngs75_a2a.a2a.resilience import CircuitOpenError

        err = CircuitOpenError("http://localhost:8080")
        assert "http://localhost:8080" in str(err)
        assert err.url == "http://localhost:8080"
