"""성능 최적화 모듈 유닛 테스트.

LLM 호출 없이 다음 모듈들을 검증한다:
- token_optimizer: 토큰 카운팅, 프롬프트 압축, 토큰 예산 관리
- llm_cache: LLM 응답 캐싱, TTL 만료, LRU 정책, 메트릭
- profiler: 실행 시간 프로파일링, 토큰 추적, 리포트 생성
- batch_executor: 병렬 실행, 세마포어, 에러 핸들링
- model_tiers: 비용/성능 분석, 목적별 추천
"""

from __future__ import annotations

import asyncio
import time

import pytest

# ══════════════════════════════════════════════════════════════════
#  token_optimizer 테스트
# ══════════════════════════════════════════════════════════════════

from coding_agent.utils.token_optimizer import (
    TokenBudget,
    compress_prompt,
    count_messages_tokens,
    count_tokens,
    report_prompt_tokens,
    report_prompt_tokens_text,
)


class TestCountTokens:
    """토큰 카운팅 테스트."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        n = count_tokens("Hello, world!")
        assert n > 0
        assert n < 20  # 짧은 문장은 토큰이 적어야 함

    def test_korean_text(self):
        """한국어 텍스트도 정상적으로 토큰을 카운팅한다."""
        n = count_tokens("안녕하세요, 세계!")
        assert n > 0

    def test_different_models_may_differ(self):
        """다른 모델 이름으로도 동작한다 (폴백 인코딩 사용)."""
        text = "Test prompt text"
        n1 = count_tokens(text, model="openrouter/deepseek/deepseek-v3.2")
        n2 = count_tokens(text, model="unknown-model-xyz")
        # 둘 다 양수여야 하고, 같은 인코딩(cl100k_base 폴백)을 쓸 수 있음
        assert n1 > 0
        assert n2 > 0


class TestCountMessagesTokens:
    """메시지 토큰 카운팅 테스트."""

    def test_single_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        n = count_messages_tokens(messages)
        # 메시지 오버헤드 + 내용 토큰 + 프라이밍 토큰
        assert n > 0

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        n = count_messages_tokens(messages)
        assert n > count_messages_tokens([messages[0]])


class TestCompressPrompt:
    """프롬프트 압축 테스트."""

    def test_removes_extra_blank_lines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = compress_prompt(text)
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_removes_extra_spaces(self):
        text = "Hello     world    test"
        result = compress_prompt(text)
        assert "     " not in result
        assert "Hello" in result

    def test_preserves_code_blocks(self):
        text = "Before\n```python\n    indented    code\n```\nAfter"
        result = compress_prompt(text)
        assert "    indented    code" in result

    def test_max_tokens_truncation(self):
        # 매우 긴 텍스트를 작은 예산으로 잘라냄
        text = "hello " * 500
        result = compress_prompt(text, max_tokens=50)
        tokens = count_tokens(result)
        assert tokens <= 55  # 약간의 여유 (말줄임 표시 포함)

    def test_strip_comments(self):
        text = "# 주석\n실제 내용\n# 또 다른 주석"
        result = compress_prompt(text, strip_comments=True)
        assert "# 주석" not in result
        assert "실제 내용" in result

    def test_dedent(self):
        text = "    indented text\n    more indented"
        result = compress_prompt(text)
        assert result.startswith("indented text")


class TestTokenBudget:
    """토큰 예산 관리 테스트."""

    def test_under_budget(self):
        budget = TokenBudget(parse=1000)
        is_over, info = budget.check("parse", "Short text")
        assert not is_over
        assert info["tokens"] < 1000
        assert info["over_by"] == 0

    def test_over_budget(self):
        budget = TokenBudget(parse=5)  # 매우 작은 예산
        is_over, info = budget.check(
            "parse", "This text will exceed five tokens easily"
        )
        assert is_over
        assert info["over_by"] > 0

    def test_default_budget_for_unknown_purpose(self):
        budget = TokenBudget(default=2000)
        assert budget.get_budget("unknown") == 2000

    def test_usage_tracking(self):
        budget = TokenBudget(parse=10000)
        budget.check("parse", "First call")
        budget.check("parse", "Second call")
        assert "parse" in budget.usage
        assert budget.usage["parse"] > 0

    def test_reset(self):
        budget = TokenBudget()
        budget.check("parse", "Some text")
        assert len(budget.usage) > 0
        budget.reset()
        assert len(budget.usage) == 0


class TestPromptReport:
    """프롬프트 토큰 리포트 테스트."""

    def test_report_structure(self):
        report = report_prompt_tokens()
        assert "parse" in report
        assert "execute" in report
        assert "verify" in report
        assert "_total" in report

        for name in ["parse", "execute", "verify"]:
            assert "tokens" in report[name]
            assert report[name]["tokens"] > 0

    def test_report_text(self):
        text = report_prompt_tokens_text()
        assert "프롬프트 토큰 리포트" in text
        assert "합계" in text


# ══════════════════════════════════════════════════════════════════
#  llm_cache 테스트
# ══════════════════════════════════════════════════════════════════

from coding_agent.utils.llm_cache import LLMCache, get_llm_cache, reset_llm_cache  # noqa: E402


class TestLLMCache:
    """LLM 응답 캐시 테스트."""

    def test_put_and_get(self):
        cache = LLMCache(max_size=10, ttl_seconds=60)
        cache.put(
            "openrouter/deepseek/deepseek-v3.2",
            "test prompt",
            temperature=0.0,
            response="test response",
        )
        result = cache.get("openrouter/deepseek/deepseek-v3.2", "test prompt", temperature=0.0)
        assert result == "test response"

    def test_cache_miss(self):
        cache = LLMCache(max_size=10, ttl_seconds=60)
        result = cache.get(
            "openrouter/deepseek/deepseek-v3.2", "nonexistent prompt", temperature=0.0
        )
        assert result is None

    def test_different_keys(self):
        """다른 모델/프롬프트/temperature는 다른 캐시 엔트리."""
        cache = LLMCache(max_size=10, ttl_seconds=60)
        cache.put(
            "openrouter/deepseek/deepseek-v3.2", "prompt", temperature=0.0, response="response-a"
        )
        cache.put(
            "openrouter/deepseek/deepseek-v3.2", "prompt", temperature=0.5, response="response-b"
        )
        cache.put("qwen/qwen3.5-9b", "prompt", temperature=0.0, response="response-c")

        assert (
            cache.get("openrouter/deepseek/deepseek-v3.2", "prompt", temperature=0.0)
            == "response-a"
        )
        assert (
            cache.get("openrouter/deepseek/deepseek-v3.2", "prompt", temperature=0.5)
            == "response-b"
        )
        assert cache.get("qwen/qwen3.5-9b", "prompt", temperature=0.0) == "response-c"

    def test_lru_eviction(self):
        """max_size 초과 시 가장 오래된 항목을 제거한다."""
        cache = LLMCache(max_size=2, ttl_seconds=0)  # TTL 없음
        cache.put("m", "p1", response="r1")
        cache.put("m", "p2", response="r2")
        cache.put("m", "p3", response="r3")  # p1이 제거되어야 함

        assert cache.get("m", "p1") is None  # 제거됨
        assert cache.get("m", "p2") == "r2"
        assert cache.get("m", "p3") == "r3"
        assert cache.metrics.evictions >= 1

    def test_ttl_expiration(self):
        """TTL 만료 후 캐시에서 사라진다."""
        cache = LLMCache(max_size=10, ttl_seconds=0.01)  # 10ms TTL
        cache.put("m", "p", response="r")

        # 즉시 조회 — 히트
        assert cache.get("m", "p") == "r"

        # TTL 만료 대기
        time.sleep(0.02)

        # 만료 후 조회 — 미스
        assert cache.get("m", "p") is None
        assert cache.metrics.expirations >= 1

    def test_metrics(self):
        cache = LLMCache(max_size=10, ttl_seconds=60)
        cache.put("m", "p1", response="r1")

        cache.get("m", "p1")  # 히트
        cache.get("m", "p2")  # 미스

        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 1
        assert cache.metrics.hit_rate == 0.5

    def test_invalidate(self):
        cache = LLMCache(max_size=10, ttl_seconds=60)
        cache.put("m", "p", response="r")
        assert cache.invalidate("m", "p") is True
        assert cache.get("m", "p") is None
        assert cache.invalidate("m", "p") is False  # 이미 없음

    def test_clear(self):
        cache = LLMCache(max_size=10, ttl_seconds=60)
        cache.put("m", "p1", response="r1")
        cache.put("m", "p2", response="r2")
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    def test_cleanup_expired(self):
        cache = LLMCache(max_size=10, ttl_seconds=0.01)
        cache.put("m", "p1", response="r1")
        cache.put("m", "p2", response="r2")
        time.sleep(0.02)
        removed = cache.cleanup_expired()
        assert removed == 2

    def test_update_existing_entry(self):
        """이미 존재하는 키를 업데이트하면 값이 갱신된다."""
        cache = LLMCache(max_size=10, ttl_seconds=60)
        cache.put("m", "p", response="old")
        cache.put("m", "p", response="new")
        assert cache.get("m", "p") == "new"
        assert cache.size == 1  # 중복 항목 없음


class TestGlobalCache:
    """전역 캐시 인스턴스 테스트."""

    def test_singleton(self):
        reset_llm_cache()
        c1 = get_llm_cache()
        c2 = get_llm_cache()
        assert c1 is c2
        reset_llm_cache()

    def test_reset(self):
        reset_llm_cache()
        c1 = get_llm_cache()
        reset_llm_cache()
        c2 = get_llm_cache()
        assert c1 is not c2
        reset_llm_cache()


# ══════════════════════════════════════════════════════════════════
#  profiler 테스트
# ══════════════════════════════════════════════════════════════════

from coding_agent.utils.profiler import (  # noqa: E402
    NodeProfile,
    Profiler,
    profile_async,
    profile_sync,
)


class TestNodeProfile:
    """NodeProfile 데이터 클래스 테스트."""

    def test_record_duration(self):
        node = NodeProfile(name="test")
        node.record_duration(0.1)
        node.record_duration(0.3)
        assert node.call_count == 2
        assert abs(node.total_duration_s - 0.4) < 0.01
        assert abs(node.avg_duration_s - 0.2) < 0.01
        assert abs(node.min_duration_s - 0.1) < 0.01
        assert abs(node.max_duration_s - 0.3) < 0.01

    def test_total_tokens(self):
        node = NodeProfile(name="test", input_tokens=100, output_tokens=50)
        assert node.total_tokens == 150

    def test_to_dict(self):
        node = NodeProfile(name="test")
        node.record_duration(0.5)
        d = node.to_dict()
        assert d["name"] == "test"
        assert d["call_count"] == 1


class TestProfiler:
    """Profiler 테스트."""

    def test_context_manager(self):
        profiler = Profiler(name="test")
        with profiler.measure("node_a"):
            time.sleep(0.01)

        assert "node_a" in profiler.nodes
        assert profiler.nodes["node_a"].call_count == 1
        assert profiler.nodes["node_a"].total_duration_s >= 0.005

    def test_start_end_pattern(self):
        profiler = Profiler(name="test")
        profiler.start_node("node_b")
        time.sleep(0.01)
        duration = profiler.end_node("node_b")
        assert duration >= 0.005
        assert profiler.nodes["node_b"].call_count == 1

    def test_end_node_without_start_raises(self):
        profiler = Profiler(name="test")
        with pytest.raises(KeyError):
            profiler.end_node("unstarted_node")

    def test_record_tokens(self):
        profiler = Profiler(name="test")
        profiler.record_tokens("node_c", input_tokens=100, output_tokens=50)
        profiler.record_tokens("node_c", input_tokens=200, output_tokens=100)
        assert profiler.nodes["node_c"].input_tokens == 300
        assert profiler.nodes["node_c"].output_tokens == 150
        assert profiler.total_input_tokens == 300
        assert profiler.total_output_tokens == 150
        assert profiler.total_tokens == 450

    def test_finalize(self):
        profiler = Profiler(name="test")
        time.sleep(0.01)
        profiler.finalize()
        assert profiler.total_duration_s >= 0.005

    def test_report(self):
        profiler = Profiler(name="test_report")
        with profiler.measure("node_x"):
            pass
        profiler.record_tokens("node_x", input_tokens=50, output_tokens=30)

        report = profiler.report()
        assert report["profiler_name"] == "test_report"
        assert "nodes" in report
        assert "node_x" in report["nodes"]
        assert report["total_tokens"] == 80

    def test_report_text(self):
        profiler = Profiler(name="text_test")
        with profiler.measure("my_node"):
            pass
        text = profiler.report_text()
        assert "프로파일 리포트" in text
        assert "my_node" in text

    def test_to_agent_metrics_dict(self):
        """AgentMetricsCollector.to_dict() 호환 형식."""
        profiler = Profiler(name="compat_test")
        with profiler.measure("parse"):
            pass
        profiler.record_tokens("parse", input_tokens=100, output_tokens=50)

        d = profiler.to_agent_metrics_dict()
        assert d["agent_name"] == "compat_test"
        assert "total_tokens" in d
        assert "prompt_tokens" in d
        assert "completion_tokens" in d
        assert "duration_ms" in d
        assert "error_count" in d
        assert "error_rate" in d
        assert "nodes" in d

    def test_error_tracking_in_context_manager(self):
        profiler = Profiler(name="error_test")
        with pytest.raises(ValueError):
            with profiler.measure("failing_node"):
                raise ValueError("test error")

        assert profiler.nodes["failing_node"].errors == 1
        assert profiler.nodes["failing_node"].call_count == 1


class TestProfileDecorators:
    """프로파일 데코레이터 테스트."""

    def test_profile_sync(self):
        profiler = Profiler(name="dec_test")

        @profile_sync("sync_func", profiler=profiler)
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(5)
        assert result == 10
        assert "sync_func" in profiler.nodes
        assert profiler.nodes["sync_func"].call_count == 1

    async def test_profile_async(self):
        profiler = Profiler(name="async_dec_test")

        @profile_async("async_func", profiler=profiler)
        async def my_async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3

        result = await my_async_func(5)
        assert result == 15
        assert "async_func" in profiler.nodes
        assert profiler.nodes["async_func"].call_count == 1


# ══════════════════════════════════════════════════════════════════
#  batch_executor 테스트
# ══════════════════════════════════════════════════════════════════

from coding_agent.core.batch_executor import BatchExecutor, BatchResult, TaskResult  # noqa: E402


class TestTaskResult:
    """TaskResult 데이터 클래스 테스트."""

    def test_success(self):
        r = TaskResult(index=0, success=True, value="ok", duration_s=0.1)
        assert r.success
        assert r.value == "ok"
        assert r.error is None

    def test_failure(self):
        err = ValueError("test")
        r = TaskResult(index=1, success=False, error=err, duration_s=0.2)
        assert not r.success
        assert r.error is err


class TestBatchResult:
    """BatchResult 데이터 클래스 테스트."""

    def test_properties(self):
        results = [
            TaskResult(index=0, success=True, value="a"),
            TaskResult(index=1, success=False, error=ValueError("e")),
            TaskResult(index=2, success=True, value="c"),
        ]
        br = BatchResult(results=results, total_duration_s=1.0)
        assert br.success_count == 2
        assert br.failure_count == 1
        assert br.total_count == 3
        assert not br.all_succeeded
        assert br.values == ["a", None, "c"]
        assert br.successful_values == ["a", "c"]
        assert len(br.errors) == 1

    def test_to_dict(self):
        br = BatchResult(results=[], total_duration_s=0.5)
        d = br.to_dict()
        assert d["total_count"] == 0
        assert d["all_succeeded"] is True


class TestBatchExecutor:
    """BatchExecutor 테스트."""

    async def test_empty_tasks(self):
        executor = BatchExecutor(max_concurrency=5)
        result = await executor.execute([])
        assert result.total_count == 0

    async def test_all_succeed(self):
        executor = BatchExecutor(max_concurrency=3)

        async def task_a():
            return "a"

        async def task_b():
            return "b"

        result = await executor.execute([task_a, task_b])
        assert result.all_succeeded
        assert result.values == ["a", "b"]

    async def test_partial_failure(self):
        """일부 태스크 실패 시 나머지 결과는 정상 반환."""
        executor = BatchExecutor(max_concurrency=5)

        async def success_task():
            return "ok"

        async def fail_task():
            raise ValueError("boom")

        result = await executor.execute([success_task, fail_task, success_task])
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.results[0].success
        assert not result.results[1].success
        assert result.results[2].success

    async def test_concurrency_limit(self):
        """세마포어로 동시 실행 수가 제한된다."""
        executor = BatchExecutor(max_concurrency=2)
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_task():
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.05)
            async with lock:
                current_concurrent -= 1
            return True

        tasks = [tracked_task for _ in range(5)]
        result = await executor.execute(tasks)
        assert result.all_succeeded
        assert max_concurrent <= 2

    async def test_timeout(self):
        """개별 태스크 타임아웃."""
        executor = BatchExecutor(max_concurrency=5, timeout_s=0.05)

        async def slow_task():
            await asyncio.sleep(1.0)
            return "late"

        async def fast_task():
            return "fast"

        result = await executor.execute([fast_task, slow_task])
        assert result.results[0].success
        assert not result.results[1].success  # 타임아웃

    async def test_execute_map(self):
        """execute_map으로 동일 함수를 여러 입력에 적용."""
        executor = BatchExecutor(max_concurrency=3)

        async def double(x: int) -> int:
            return x * 2

        result = await executor.execute_map(double, [1, 2, 3, 4])
        assert result.all_succeeded
        assert result.values == [2, 4, 6, 8]

    async def test_order_preserved(self):
        """결과 순서가 입력 순서와 일치한다."""
        executor = BatchExecutor(max_concurrency=2)

        async def delayed(val: int) -> int:
            # 역순 지연으로 완료 순서와 입력 순서를 다르게 함
            await asyncio.sleep(0.01 * (5 - val))
            return val

        result = await executor.execute_map(delayed, [1, 2, 3, 4])
        assert result.values == [1, 2, 3, 4]


# ══════════════════════════════════════════════════════════════════
#  model_tiers 비용/성능 분석 테스트
# ══════════════════════════════════════════════════════════════════

from coding_agent.core.model_tiers import (  # noqa: E402
    ModelCostInfo,
    ModelTier,
    TierConfig,
    analyze_tier_tradeoffs,
    build_default_tiers,
    estimate_cost,
    get_model_cost_info,
    recommend_tier_for_purpose,
    register_model_cost_info,
)


class TestModelCostInfo:
    """모델 비용 메타데이터 테스트."""

    def test_avg_cost(self):
        info = ModelCostInfo(
            model="test",
            input_cost_per_1m=2.0,
            output_cost_per_1m=8.0,
        )
        assert info.avg_cost_per_1m == 5.0

    def test_registry(self):
        """비용 데이터베이스 조회 및 등록."""
        info = get_model_cost_info("openrouter/deepseek/deepseek-v3.2")
        assert info is not None
        assert info.input_cost_per_1m > 0

        custom = ModelCostInfo(
            model="custom-model",
            input_cost_per_1m=1.0,
            output_cost_per_1m=3.0,
        )
        register_model_cost_info(custom)
        assert get_model_cost_info("custom-model") is not None

    def test_tier_config_cost_info(self):
        tc = TierConfig(model="openrouter/deepseek/deepseek-v3.2")
        assert tc.cost_info is not None
        assert tc.cost_info.model == "openrouter/deepseek/deepseek-v3.2"

    def test_tier_config_unknown_model(self):
        tc = TierConfig(model="unknown-xyz")
        assert tc.cost_info is None


class TestRecommendTierForPurpose:
    """목적별 티어 추천 테스트."""

    def test_parsing_prefers_fast(self):
        tiers = build_default_tiers()
        tier_name, config, analysis = recommend_tier_for_purpose("parsing", tiers)
        # parsing은 speed 가중치가 높으므로 FAST 티어가 추천되어야 함
        assert (
            analysis["scores"][ModelTier.FAST] >= analysis["scores"][ModelTier.STRONG]
        )

    def test_generation_prefers_strong(self):
        tiers = build_default_tiers()
        tier_name, config, analysis = recommend_tier_for_purpose("generation", tiers)
        # generation은 code 가중치가 높으므로 STRONG 티어가 추천되어야 함
        assert (
            analysis["scores"][ModelTier.STRONG] >= analysis["scores"][ModelTier.FAST]
        )

    def test_unknown_purpose_returns_result(self):
        tiers = build_default_tiers()
        tier_name, config, analysis = recommend_tier_for_purpose("unknown", tiers)
        assert config is not None
        assert "recommended_tier" in analysis

    def test_empty_tiers_fallback(self):
        tier_name, config, analysis = recommend_tier_for_purpose("generation", {})
        assert config is not None  # 폴백으로 기본 config 반환


class TestEstimateCost:
    """비용 추정 테스트."""

    def test_known_model(self):
        tc = TierConfig(model="openrouter/deepseek/deepseek-v3.2")
        cost = estimate_cost(tc, input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert cost["total_cost_usd"] > 0
        assert cost["input_cost_usd"] > 0
        assert cost["output_cost_usd"] > 0

    def test_unknown_model(self):
        tc = TierConfig(model="unknown-model")
        cost = estimate_cost(tc, input_tokens=1000, output_tokens=500)
        assert cost is None

    def test_zero_tokens(self):
        tc = TierConfig(model="openrouter/deepseek/deepseek-v3.2")
        cost = estimate_cost(tc, input_tokens=0, output_tokens=0)
        assert cost is not None
        assert cost["total_cost_usd"] == 0.0


class TestAnalyzeTierTradeoffs:
    """티어 트레이드오프 분석 테스트."""

    def test_returns_all_tiers(self):
        tiers = build_default_tiers()
        results = analyze_tier_tradeoffs(tiers)
        assert len(results) == len(tiers)

    def test_sorted_by_cost(self):
        tiers = build_default_tiers()
        results = analyze_tier_tradeoffs(tiers)
        costs = [
            r["estimated_cost"]["total_cost_usd"]
            for r in results
            if r["estimated_cost"]
        ]
        assert costs == sorted(costs)

    def test_includes_purpose_fit(self):
        tiers = build_default_tiers()
        results = analyze_tier_tradeoffs(tiers)
        for r in results:
            assert "purpose_fit" in r
            assert "parsing" in r["purpose_fit"]
            assert "generation" in r["purpose_fit"]
