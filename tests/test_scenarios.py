"""채점 증빙용 시나리오 테스트 — 순수 로직 테스트 (LLM 호출 없음).

5가지 시나리오:
  a) 모델 타임아웃 → 재시도 → fallback
  b) 무진전 루프 감지 (StallDetector)
  c) Safe Stop (AbortController)
  d) 메모리 누적 + 재활용 (MemoryMiddleware)
  e) SubAgent 상태 전이 + retry + JSONL 영속화
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from coding_agent.core.abort_controller import (
    AbortController,
    AbortError,
    AbortReason,
)
from coding_agent.core.middleware.base import ModelRequest, ModelResponse
from coding_agent.core.middleware.memory import MemoryMiddleware
from coding_agent.core.middleware.resilience import ResilienceMiddleware
from coding_agent.core.resilience import (
    FailureMatrix,
    FailurePolicy,
    FailureType,
    ModelFallbackChain,
    RetryWithBackoff,
)
from coding_agent.core.stall_detector import StallAction, StallDetector
from coding_agent.core.subagents.registry import SubAgentRegistry
from coding_agent.core.subagents.schemas import (
    VALID_TRANSITIONS,
    SubAgentInstance,
    SubAgentResult,
    SubAgentSpec,
    SubAgentStatus,
)


# ── 시나리오 A: 모델 타임아웃 → 재시도 → fallback ─────────────


class TestScenarioATimeoutRetryFallback:
    """ResilienceMiddleware가 타임아웃 시 retry → fallback을 수행."""

    @pytest.mark.asyncio
    async def test_retry_with_backoff_succeeds_on_third_attempt(self):
        """RetryWithBackoff: 2회 실패 후 3번째 성공."""
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("timeout")
            return "success"

        retry = RetryWithBackoff(
            max_retries=2,
            backoff_base=0.01,
            backoff_multiplier=2.0,
        )
        result = await retry.execute(flaky_fn)
        assert result == "success"
        assert call_count == 3
        assert retry.attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """재시도 횟수 소진 시 최종 예외 발생."""
        async def always_fail():
            raise TimeoutError("timeout")

        retry = RetryWithBackoff(
            max_retries=2,
            backoff_base=0.01,
            backoff_multiplier=1.0,
        )
        with pytest.raises(TimeoutError):
            await retry.execute(always_fail)
        assert retry.attempt_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_failure_matrix_policy_limits_retries(self):
        """FailureMatrix 정책에 따라 재시도 횟수가 제한된다."""
        matrix = FailureMatrix()

        # MODEL_TIMEOUT 기본 정책: max_retries=2
        policy = matrix.get_policy(FailureType.MODEL_TIMEOUT)
        assert policy.max_retries == 2
        assert policy.fallback_enabled is True

        # BAD_TOOL_CALL: max_retries=1
        policy2 = matrix.get_policy(FailureType.BAD_TOOL_CALL)
        assert policy2.max_retries == 1

        # STUCK_LOOP: max_retries=0 (재시도 없음)
        policy3 = matrix.get_policy(FailureType.STUCK_LOOP)
        assert policy3.max_retries == 0

    @pytest.mark.asyncio
    async def test_resilience_middleware_retry_then_fallback(self):
        """ResilienceMiddleware: 재시도 실패 → fallback 체인으로 대체 모델 시도."""
        abort_ctrl = AbortController()
        matrix = FailureMatrix()
        # 재시도 빠르게
        matrix.set_policy(FailurePolicy(
            failure_type=FailureType.MODEL_TIMEOUT,
            max_retries=1,
            backoff_base=0.01,
            backoff_multiplier=1.0,
            fallback_enabled=True,
        ))

        fallback_response = AIMessage(content="fallback 응답")
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=fallback_response)

        chain = ModelFallbackChain(
            models=[mock_model],
            model_names=["fallback-model"],
            timeout_per_model=5.0,
        )

        mw = ResilienceMiddleware(
            abort_controller=abort_ctrl,
            failure_matrix=matrix,
            fallback_chain_builder=lambda: chain,
        )

        call_count = 0

        async def failing_handler(req: ModelRequest) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("model timeout")

        request = ModelRequest(
            system_message="test",
            messages=[HumanMessage(content="hello")],
            metadata={"purpose": "generation"},
        )

        response = await mw.wrap_model_call(request, failing_handler)
        assert response.message.content == "fallback 응답"
        assert call_count == 2  # 1 initial + 1 retry
        mock_model.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_backoff_delay_calculation(self):
        """지수 백오프 지연 시간 계산이 올바른지 확인."""
        retry = RetryWithBackoff(
            backoff_base=1.0,
            backoff_multiplier=2.0,
            backoff_max=30.0,
        )
        assert retry.calculate_delay(0) == 1.0   # 1 * 2^0
        assert retry.calculate_delay(1) == 2.0   # 1 * 2^1
        assert retry.calculate_delay(2) == 4.0   # 1 * 2^2
        assert retry.calculate_delay(5) == 30.0  # min(32, 30)


# ── 시나리오 B: 무진전 루프 감지 ──────────────────────────────


class TestScenarioBStallDetection:
    """StallDetector가 반복 도구 호출 및 낮은 diversity를 감지."""

    def test_same_tool_3_times_force_exit(self):
        """동일 도구 호출 3회 반복 시 FORCE_EXIT."""
        detector = StallDetector(warn_threshold=2, exit_threshold=3, window_size=10)

        action1 = detector.record_and_check("read_file", {"path": "a.py"})
        assert action1 == StallAction.CONTINUE

        action2 = detector.record_and_check("read_file", {"path": "a.py"})
        assert action2 == StallAction.WARN

        action3 = detector.record_and_check("read_file", {"path": "a.py"})
        assert action3 == StallAction.FORCE_EXIT

    def test_different_args_no_stall(self):
        """동일 도구지만 다른 인자면 stall이 아니다."""
        detector = StallDetector(warn_threshold=2, exit_threshold=3, window_size=10)

        for i in range(5):
            action = detector.record_and_check("read_file", {"path": f"file_{i}.py"})
            assert action == StallAction.CONTINUE

    def test_diversity_below_threshold_force_exit(self):
        """diversity < 0.2 시 FORCE_EXIT."""
        # window_size=5, diversity_exit_threshold=0.2
        # 5회 동일 호출 → unique=1, diversity=0.2 → 경계값
        # exit_threshold=10 으로 높여서 반복 카운트 감지는 비활성화
        detector = StallDetector(
            warn_threshold=10,
            exit_threshold=20,
            window_size=5,
            diversity_exit_threshold=0.2,
            diversity_warn_threshold=0.3,
        )

        # 5회 동일 호출 → unique=1/5=0.2 → 경계값이므로 CONTINUE
        # diversity < 0.2 이어야 FORCE_EXIT → 더 극단적 케이스 필요
        # window_size=10, 10번 모두 동일 → unique=1/10=0.1 < 0.2
        detector2 = StallDetector(
            warn_threshold=20,
            exit_threshold=30,
            window_size=10,
            diversity_exit_threshold=0.2,
            diversity_warn_threshold=0.3,
        )

        for _ in range(10):
            action = detector2.record_and_check("read_file", {"path": "a.py"})

        # 마지막 호출에서 diversity = 1/10 = 0.1 < 0.2
        assert action == StallAction.FORCE_EXIT

    def test_stall_summary_message(self):
        """get_stall_summary()가 반복 패턴을 설명한다."""
        detector = StallDetector(exit_threshold=5)
        for _ in range(3):
            detector.record_and_check("read_file", {"path": "a.py"})

        summary = detector.get_stall_summary()
        assert "read_file" in summary
        assert "3회" in summary

    def test_reset_clears_state(self):
        """reset() 호출 후 상태가 초기화된다."""
        detector = StallDetector()
        detector.record_and_check("read_file", {"path": "a.py"})
        detector.record_and_check("read_file", {"path": "a.py"})
        detector.reset()

        action = detector.record_and_check("read_file", {"path": "a.py"})
        assert action == StallAction.CONTINUE


# ── 시나리오 C: Safe Stop ─────────────────────────────────────


class TestScenarioCSafeStop:
    """AbortController.abort(USER_INTERRUPT) → check_or_raise()에서 AbortError."""

    def test_abort_user_interrupt(self):
        """abort(USER_INTERRUPT) → check_or_raise()에서 AbortError 발생."""
        ctrl = AbortController()
        assert not ctrl.is_aborted

        ctrl.abort(AbortReason.USER_INTERRUPT)
        assert ctrl.is_aborted
        assert ctrl.reason == AbortReason.USER_INTERRUPT

        with pytest.raises(AbortError) as exc_info:
            ctrl.check_or_raise()

        assert exc_info.value.reason == AbortReason.USER_INTERRUPT
        assert "중단" in exc_info.value.abort_message

    def test_abort_with_custom_message(self):
        """사용자 지정 메시지로 중단."""
        ctrl = AbortController()
        ctrl.abort(AbortReason.USER_INTERRUPT, "안전하게 중단됨")

        assert ctrl.message == "안전하게 중단됨"
        with pytest.raises(AbortError) as exc_info:
            ctrl.check_or_raise()
        assert exc_info.value.abort_message == "안전하게 중단됨"

    def test_check_or_raise_no_abort(self):
        """중단 신호 없으면 예외 없이 통과."""
        ctrl = AbortController()
        ctrl.check_or_raise()  # should not raise

    def test_reset_clears_abort(self):
        """reset() 후 중단 상태가 해제된다."""
        ctrl = AbortController()
        ctrl.abort(AbortReason.STALL_DETECTED)
        assert ctrl.is_aborted

        ctrl.reset()
        assert not ctrl.is_aborted
        assert ctrl.reason is None
        ctrl.check_or_raise()  # should not raise

    @pytest.mark.asyncio
    async def test_resilience_middleware_aborts_before_llm_call(self):
        """ResilienceMiddleware: abort 신호가 있으면 LLM 호출 전에 AbortError."""
        ctrl = AbortController()
        ctrl.abort(AbortReason.USER_INTERRUPT, "안전하게 중단됨")

        mw = ResilienceMiddleware(abort_controller=ctrl)

        handler_called = False

        async def handler(req):
            nonlocal handler_called
            handler_called = True
            return ModelResponse(message=AIMessage(content="ok"))

        request = ModelRequest(
            system_message="test",
            messages=[HumanMessage(content="hello")],
        )

        with pytest.raises(AbortError) as exc_info:
            await mw.wrap_model_call(request, handler)

        assert not handler_called
        assert exc_info.value.abort_message == "안전하게 중단됨"

    def test_all_abort_reasons(self):
        """모든 AbortReason에 기본 메시지가 제공된다."""
        for reason in AbortReason:
            ctrl = AbortController()
            ctrl.abort(reason)
            assert ctrl.message  # 빈 문자열이 아님
            ctrl.reset()


# ── 시나리오 D: 메모리 누적 + 재활용 ─────────────────────────


class TestScenarioDMemory:
    """MemoryMiddleware가 domain_knowledge 저장 → 검색 → 주입하는 흐름."""

    @pytest.mark.asyncio
    async def test_domain_knowledge_accumulate_and_inject(self):
        """domain_knowledge 저장 → 다음 요청에서 검색하여 시스템 프롬프트에 주입."""
        mock_store = MagicMock()

        # 검색 결과 mock
        mock_item = MagicMock()
        mock_item.tags = ["python"]
        mock_item.type = MagicMock()
        mock_item.type.value = "domain_knowledge"
        mock_item.content = "Python에서 리스트 컴프리헨션은 for 루프보다 빠르다."
        mock_store.search.return_value = [mock_item]

        mw = MemoryMiddleware(
            memory_store=mock_store,
            memory_limit=5,
            auto_accumulate=False,
        )

        request = ModelRequest(
            system_message="기본 시스템 프롬프트",
            messages=[HumanMessage(content="Python 리스트 성능에 대해 알려줘")],
            metadata={"purpose": "generation"},
        )

        handler_request = None

        async def capture_handler(req: ModelRequest) -> ModelResponse:
            nonlocal handler_request
            handler_request = req
            return ModelResponse(message=AIMessage(content="응답"))

        await mw.wrap_model_call(request, capture_handler)

        # MemoryStore.search가 호출되었는지 확인
        assert mock_store.search.called
        # 시스템 프롬프트에 메모리가 주입되었는지 확인
        assert handler_request is not None
        assert "리스트 컴프리헨션" in handler_request.system_message

    @pytest.mark.asyncio
    async def test_auto_accumulate_domain_knowledge(self):
        """SLM이 domain_knowledge를 추출하면 MemoryStore에 축적된다."""
        mock_store = MagicMock()
        mock_store.search.return_value = []

        # SLM이 domain_knowledge 추출 결과 반환
        slm_result = json.dumps({
            "type": "domain_knowledge",
            "content": "FastAPI는 Starlette 기반 ASGI 프레임워크이다.",
            "tags": ["fastapi", "python"],
        })
        mock_slm = AsyncMock(return_value=slm_result)

        mw = MemoryMiddleware(
            memory_store=mock_store,
            slm_invoker=mock_slm,
            auto_accumulate=True,
        )

        request = ModelRequest(
            system_message="test",
            messages=[HumanMessage(content="FastAPI에 대해 설명해줘")],
        )

        async def handler(req):
            return ModelResponse(message=AIMessage(content="FastAPI는 ..."))

        await mw.wrap_model_call(request, handler)

        mock_store.accumulate_domain_knowledge.assert_called_once_with(
            content="FastAPI는 Starlette 기반 ASGI 프레임워크이다.",
            tags=["fastapi", "python"],
            source="auto",
        )

    @pytest.mark.asyncio
    async def test_auto_accumulate_user_profile(self):
        """SLM이 user_profile을 추출하면 MemoryStore에 축적된다."""
        mock_store = MagicMock()
        mock_store.search.return_value = []

        slm_result = json.dumps({
            "type": "user_profile",
            "content": "사용자는 Python 시니어 개발자이다.",
            "tags": ["experience"],
        })
        mock_slm = AsyncMock(return_value=slm_result)

        mw = MemoryMiddleware(
            memory_store=mock_store,
            slm_invoker=mock_slm,
            auto_accumulate=True,
        )

        request = ModelRequest(
            system_message="test",
            messages=[HumanMessage(content="나는 10년차 파이썬 개발자야")],
        )

        async def handler(req):
            return ModelResponse(message=AIMessage(content="네, 시니어 개발자시군요"))

        await mw.wrap_model_call(request, handler)

        mock_store.accumulate_user_profile.assert_called_once_with(
            content="사용자는 Python 시니어 개발자이다.",
            tags=["experience"],
            source="auto",
        )

    @pytest.mark.asyncio
    async def test_state_memory_injection(self):
        """state에 있는 user_profile_context, domain_knowledge_context가 주입된다."""
        mw = MemoryMiddleware(auto_accumulate=False)

        request = ModelRequest(
            system_message="기본 프롬프트",
            messages=[HumanMessage(content="코드 작성해줘")],
            state={
                "user_profile_context": "시니어 백엔드 개발자",
                "domain_knowledge_context": "이 프로젝트는 FastAPI를 사용한다.",
            },
        )

        handler_request = None

        async def capture_handler(req: ModelRequest) -> ModelResponse:
            nonlocal handler_request
            handler_request = req
            return ModelResponse(message=AIMessage(content="코드"))

        await mw.wrap_model_call(request, capture_handler)

        assert handler_request is not None
        assert "User Profile" in handler_request.system_message
        assert "시니어 백엔드 개발자" in handler_request.system_message
        assert "Domain Knowledge" in handler_request.system_message
        assert "FastAPI" in handler_request.system_message

    @pytest.mark.asyncio
    async def test_accumulate_failure_does_not_break_response(self):
        """자동 축적 실패해도 LLM 응답은 정상 반환."""
        mock_store = MagicMock()
        mock_store.search.return_value = []

        async def failing_slm(messages):
            raise RuntimeError("SLM 호출 실패")

        mw = MemoryMiddleware(
            memory_store=mock_store,
            slm_invoker=failing_slm,
            auto_accumulate=True,
        )

        request = ModelRequest(
            system_message="test",
            messages=[HumanMessage(content="hello")],
        )

        async def handler(req):
            return ModelResponse(message=AIMessage(content="world"))

        response = await mw.wrap_model_call(request, handler)
        assert response.message.content == "world"


# ── 시나리오 E: SubAgent 상태 전이 + retry + JSONL 영속화 ────


class TestScenarioESubAgentLifecycle:
    """SubAgent 상태 전이, retry, JSONL 영속화 테스트."""

    def _make_registry(self) -> SubAgentRegistry:
        """테스트용 레지스트리 생성."""
        registry = SubAgentRegistry()
        registry.register(SubAgentSpec(
            name="coding_assistant",
            description="코딩 에이전트",
            capabilities=["code_generation"],
        ))
        return registry

    def test_full_lifecycle_success(self):
        """CREATED → ASSIGNED → RUNNING → COMPLETED → DESTROYED 전이."""
        registry = self._make_registry()
        instance = registry.create_instance(
            "coding_assistant",
            task_summary="test task",
        )
        assert instance is not None
        agent_id = instance.agent_id
        assert instance.state == SubAgentStatus.CREATED

        # CREATED → ASSIGNED
        event = registry.transition_state(agent_id, SubAgentStatus.ASSIGNED)
        assert event is not None
        assert event.from_state == SubAgentStatus.CREATED
        assert event.to_state == SubAgentStatus.ASSIGNED

        # ASSIGNED → RUNNING
        event = registry.transition_state(agent_id, SubAgentStatus.RUNNING)
        assert event is not None

        # RUNNING → COMPLETED
        event = registry.transition_state(
            agent_id, SubAgentStatus.COMPLETED,
            result_summary="작업 완료",
        )
        assert event is not None

        # COMPLETED → DESTROYED
        event = registry.transition_state(agent_id, SubAgentStatus.DESTROYED)
        assert event is not None

    def test_retry_lifecycle(self):
        """CREATED → ASSIGNED → RUNNING → FAILED → ASSIGNED(retry) → RUNNING → COMPLETED → DESTROYED."""
        registry = self._make_registry()
        instance = registry.create_instance("coding_assistant", task_summary="retry task")
        assert instance is not None
        agent_id = instance.agent_id

        # CREATED → ASSIGNED → RUNNING
        registry.transition_state(agent_id, SubAgentStatus.ASSIGNED)
        registry.transition_state(agent_id, SubAgentStatus.RUNNING)

        # RUNNING → FAILED
        event = registry.transition_state(
            agent_id, SubAgentStatus.FAILED,
            error_message="모델 타임아웃",
        )
        assert event is not None

        # retry: FAILED → ASSIGNED
        inst = registry.get_instance(agent_id)
        assert inst is not None
        inst.retry_count += 1
        assert inst.retry_count == 1

        event = registry.transition_state(agent_id, SubAgentStatus.ASSIGNED, reason="retry")
        assert event is not None
        assert event.from_state == SubAgentStatus.FAILED
        assert event.to_state == SubAgentStatus.ASSIGNED

        # ASSIGNED → RUNNING → COMPLETED → DESTROYED
        registry.transition_state(agent_id, SubAgentStatus.RUNNING)
        registry.transition_state(agent_id, SubAgentStatus.COMPLETED, result_summary="성공")
        registry.transition_state(agent_id, SubAgentStatus.DESTROYED)

        inst = registry.get_instance(agent_id)
        assert inst is not None
        assert inst.state == SubAgentStatus.DESTROYED
        assert inst.retry_count == 1

    def test_invalid_transition_rejected(self):
        """유효하지 않은 전이는 None을 반환한다."""
        registry = self._make_registry()
        instance = registry.create_instance("coding_assistant")
        assert instance is not None
        agent_id = instance.agent_id

        # CREATED → RUNNING (잘못된 전이: ASSIGNED를 건너뜀)
        event = registry.transition_state(agent_id, SubAgentStatus.RUNNING)
        assert event is None  # 거부됨

        # 상태가 변하지 않음
        inst = registry.get_instance(agent_id)
        assert inst is not None
        assert inst.state == SubAgentStatus.CREATED

    def test_valid_transitions_map_completeness(self):
        """VALID_TRANSITIONS에 모든 상태가 정의되어 있다."""
        lifecycle_states = {
            SubAgentStatus.CREATED,
            SubAgentStatus.ASSIGNED,
            SubAgentStatus.RUNNING,
            SubAgentStatus.BLOCKED,
            SubAgentStatus.COMPLETED,
            SubAgentStatus.FAILED,
            SubAgentStatus.CANCELLED,
            SubAgentStatus.DESTROYED,
        }
        assert lifecycle_states == set(VALID_TRANSITIONS.keys())

    def test_destroyed_is_terminal(self):
        """DESTROYED 상태에서는 어떤 전이도 불가."""
        assert VALID_TRANSITIONS[SubAgentStatus.DESTROYED] == set()

    def test_persist_result_jsonl(self):
        """SubAgentResult가 JSONL 파일로 영속화된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"WORKSPACE": tmpdir}):
                from coding_agent.core.subagents.process_manager import (
                    SubAgentProcessManager,
                )

                registry = self._make_registry()
                pm = SubAgentProcessManager(registry=registry)

                instance = SubAgentInstance(
                    agent_id="test-agent-001",
                    spec_name="coding_assistant",
                    task_summary="테스트 작업",
                )

                result = SubAgentResult(
                    status="completed",
                    result="파일 생성 완료",
                    written_files=["output.py"],
                    duration_s=5.2,
                    token_usage={"input_tokens": 100, "output_tokens": 50},
                )

                pm._persist_result("test-agent-001", instance, result)

                jsonl_path = Path(tmpdir) / ".ai" / "subagent_results" / "results.jsonl"
                assert jsonl_path.exists()

                with open(jsonl_path) as f:
                    line = f.readline()
                    record = json.loads(line)

                assert record["agent_id"] == "test-agent-001"
                assert record["spec_name"] == "coding_assistant"
                assert record["status"] == "completed"
                assert record["duration_s"] == 5.2
                assert record["token_usage"]["input_tokens"] == 100

    def test_subagent_result_success_property(self):
        """SubAgentResult.success 프로퍼티 검증."""
        ok = SubAgentResult(status="completed", error=None)
        assert ok.success is True

        fail1 = SubAgentResult(status="failed", error="timeout")
        assert fail1.success is False

        fail2 = SubAgentResult(status="completed", error="partial failure")
        assert fail2.success is False
