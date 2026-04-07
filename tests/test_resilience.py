"""core/resilience.py 테스트 -- 장애 매트릭스, 재시도, 모델 폴백."""

from __future__ import annotations

import asyncio

import pytest

from youngs75_a2a.core.resilience import (
    FailureMatrix,
    FailurePolicy,
    FailureType,
    ModelFallbackChain,
    RetryWithBackoff,
)


# ── Mock LLM ──


class MockModel:
    """테스트용 Mock LLM."""

    def __init__(self, should_fail: bool = False, delay: float = 0):
        self.should_fail = should_fail
        self.delay = delay

    async def ainvoke(self, messages, **kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise Exception("model error")
        return "mock response"


# ── FailureMatrix ──


class TestFailureMatrix:
    """장애 유형별 정책 매트릭스 테스트."""

    @pytest.fixture()
    def matrix(self) -> FailureMatrix:
        return FailureMatrix()

    def test_default_policies_all_types(self, matrix: FailureMatrix):
        """7가지 FailureType 모두 기본 정책이 존재해야 한다."""
        for ft in FailureType:
            policy = matrix.get_policy(ft)
            assert policy is not None
            assert isinstance(policy, FailurePolicy)
            assert policy.failure_type == ft

    def test_get_policy_returns_correct_type(self, matrix: FailureMatrix):
        """각 타입별 정책 속성이 올바르게 설정되어 있는지 확인."""
        # MODEL_TIMEOUT: retry 2, fallback 활성
        timeout_policy = matrix.get_policy(FailureType.MODEL_TIMEOUT)
        assert timeout_policy.max_retries == 2
        assert timeout_policy.fallback_enabled is True

        # EXTERNAL_API_ERROR: retry 3
        api_policy = matrix.get_policy(FailureType.EXTERNAL_API_ERROR)
        assert api_policy.max_retries == 3

        # BAD_TOOL_CALL: retry 1
        tool_policy = matrix.get_policy(FailureType.BAD_TOOL_CALL)
        assert tool_policy.max_retries == 1

    def test_set_policy_override(self, matrix: FailureMatrix):
        """커스텀 정책 설정 후 조회 시 오버라이드된 값이 반환되어야 한다."""
        custom = FailurePolicy(
            failure_type=FailureType.MODEL_TIMEOUT,
            max_retries=5,
            backoff_base=3.0,
            fallback_enabled=False,
        )
        matrix.set_policy(custom)

        retrieved = matrix.get_policy(FailureType.MODEL_TIMEOUT)
        assert retrieved.max_retries == 5
        assert retrieved.backoff_base == 3.0
        assert retrieved.fallback_enabled is False

    def test_model_timeout_policy(self, matrix: FailureMatrix):
        """MODEL_TIMEOUT 정책에 fallback_enabled=True 확인."""
        policy = matrix.get_policy(FailureType.MODEL_TIMEOUT)
        assert policy.fallback_enabled is True
        assert policy.backoff_base > 0

    def test_safe_stop_policy(self, matrix: FailureMatrix):
        """SAFE_STOP은 retry 0, fallback 없음이어야 한다."""
        policy = matrix.get_policy(FailureType.SAFE_STOP)
        assert policy.max_retries == 0
        assert policy.fallback_enabled is False
        assert policy.backoff_base == 0.0


# ── RetryWithBackoff ──


class TestRetryWithBackoff:
    """지수 백오프 재시도 테스트."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """성공 시 재시도 없이 즉시 반환."""
        retry = RetryWithBackoff(max_retries=3, backoff_base=0.01)

        async def success_fn():
            return "ok"

        result = await retry.execute(success_fn)
        assert result == "ok"
        assert retry.attempt_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self):
        """1회 실패 후 2회차에 성공."""
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary error")
            return "recovered"

        retry = RetryWithBackoff(max_retries=3, backoff_base=0.01, backoff_multiplier=1.0)
        result = await retry.execute(flaky_fn)
        assert result == "recovered"
        assert retry.attempt_count == 2

    @pytest.mark.asyncio
    async def test_exhaust_retries(self):
        """모든 재시도 소진 후 마지막 예외가 raise 되어야 한다."""

        async def always_fail():
            raise RuntimeError("persistent error")

        retry = RetryWithBackoff(max_retries=2, backoff_base=0.01, backoff_multiplier=1.0)
        with pytest.raises(RuntimeError, match="persistent error"):
            await retry.execute(always_fail)
        assert retry.attempt_count == 3  # 초기 1회 + 재시도 2회

    def test_backoff_delay_calculation(self):
        """calculate_delay()가 지수적으로 증가하는지 확인."""
        retry = RetryWithBackoff(backoff_base=1.0, backoff_multiplier=2.0, backoff_max=100.0)
        d0 = retry.calculate_delay(0)  # 1.0 * 2^0 = 1.0
        d1 = retry.calculate_delay(1)  # 1.0 * 2^1 = 2.0
        d2 = retry.calculate_delay(2)  # 1.0 * 2^2 = 4.0

        assert d0 == pytest.approx(1.0)
        assert d1 == pytest.approx(2.0)
        assert d2 == pytest.approx(4.0)
        assert d2 > d1 > d0  # 단조 증가

    def test_backoff_max_cap(self):
        """최대 대기시간(backoff_max)을 초과하지 않아야 한다."""
        retry = RetryWithBackoff(backoff_base=1.0, backoff_multiplier=2.0, backoff_max=10.0)
        # attempt=10 → 1.0 * 2^10 = 1024 → cap 10.0
        delay = retry.calculate_delay(10)
        assert delay == 10.0

    def test_reset(self):
        """reset() 호출 후 attempt_count가 0으로 초기화."""
        retry = RetryWithBackoff()
        # 내부 상태를 직접 설정하여 시뮬레이션
        retry._attempt_count = 5
        retry._last_error = Exception("some error")
        retry.reset()
        assert retry.attempt_count == 0
        assert retry.last_error is None


# ── ModelFallbackChain ──


class TestModelFallbackChain:
    """모델 폴백 체인 테스트."""

    @pytest.mark.asyncio
    async def test_first_model_success(self):
        """첫 모델 성공 시 즉시 반환."""
        chain = ModelFallbackChain(
            models=[MockModel(), MockModel(should_fail=True)],
            model_names=["primary", "fallback"],
            timeout_per_model=5.0,
        )
        result = await chain.invoke_with_fallback([])
        assert result == "mock response"
        assert chain.current_model_name == "primary"
        assert len(chain.fallback_history) == 0

    @pytest.mark.asyncio
    async def test_fallback_to_second(self):
        """첫 모델 실패 시 두 번째 모델로 전환."""
        chain = ModelFallbackChain(
            models=[MockModel(should_fail=True), MockModel()],
            model_names=["primary", "fallback"],
            timeout_per_model=5.0,
        )
        result = await chain.invoke_with_fallback([])
        assert result == "mock response"
        assert chain.current_model_name == "fallback"
        assert len(chain.fallback_history) == 1
        assert chain.fallback_history[0]["model"] == "primary"

    @pytest.mark.asyncio
    async def test_all_models_fail(self):
        """모든 모델 실패 시 마지막 예외가 raise."""
        chain = ModelFallbackChain(
            models=[MockModel(should_fail=True), MockModel(should_fail=True)],
            model_names=["model_a", "model_b"],
            timeout_per_model=5.0,
        )
        with pytest.raises(Exception, match="model error"):
            await chain.invoke_with_fallback([])
        assert len(chain.fallback_history) == 2

    @pytest.mark.asyncio
    async def test_fallback_history_recorded(self):
        """실패 기록이 history에 쌓이는지 확인."""
        chain = ModelFallbackChain(
            models=[
                MockModel(should_fail=True),
                MockModel(should_fail=True),
                MockModel(),
            ],
            model_names=["tier1", "tier2", "tier3"],
            timeout_per_model=5.0,
        )
        result = await chain.invoke_with_fallback([])
        assert result == "mock response"

        history = chain.fallback_history
        assert len(history) == 2
        assert history[0]["model"] == "tier1"
        assert history[1]["model"] == "tier2"
        # 각 기록에 error, error_type, timestamp 존재
        for record in history:
            assert "error" in record
            assert "error_type" in record
            assert "timestamp" in record

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self):
        """timeout 발생 시 다음 모델로 전환."""
        chain = ModelFallbackChain(
            models=[
                MockModel(delay=10.0),  # 타임아웃 유발 (10초 지연)
                MockModel(),            # 즉시 성공
            ],
            model_names=["slow_model", "fast_model"],
            timeout_per_model=0.1,  # 0.1초 타임아웃
        )
        result = await chain.invoke_with_fallback([])
        assert result == "mock response"
        assert chain.current_model_name == "fast_model"
        assert len(chain.fallback_history) == 1
        assert chain.fallback_history[0]["error_type"] == "TimeoutError"
