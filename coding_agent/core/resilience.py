"""Agentic Loop 복원력 -- retry, backoff, fallback, 장애 매트릭스.

기존 StallDetector(반복감지), AbortController(safe stop), TurnBudgetTracker(예산)와
함께 다층 방어를 구성한다.

장애 유형별 처리 행렬:
  1. 모델 무응답/지연 -> timeout + retry + 대체 모델 fallback
  2. 반복 무진전 루프 -> StallDetector (기 구현)
  3. 잘못된 tool call -> ToolCallRecovery (파싱 재시도)
  4. SubAgent 실패 -> SubAgent 상태 머신에서 처리 (별도 모듈)
  5. 외부 API 오류 -> RetryWithBackoff (4xx/5xx 분류)
  6. 모델 fallback -> ModelFallbackChain
  7. safe stop -> AbortController (기 구현)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ── 장애 유형 및 정책 ──


class FailureType(str, Enum):
    """장애 유형."""

    MODEL_TIMEOUT = "model_timeout"
    STUCK_LOOP = "stuck_loop"  # -> StallDetector 연동
    BAD_TOOL_CALL = "bad_tool_call"
    SUBAGENT_FAILURE = "subagent_failure"
    EXTERNAL_API_ERROR = "external_api_error"
    MODEL_FALLBACK_NEEDED = "model_fallback_needed"
    SAFE_STOP = "safe_stop"  # -> AbortController 연동


@dataclass
class FailurePolicy:
    """장애 유형별 대응 정책."""

    failure_type: FailureType
    max_retries: int = 2
    backoff_base: float = 1.0  # 초 단위 기본 대기
    backoff_multiplier: float = 2.0  # 지수 배수
    backoff_max: float = 30.0  # 최대 대기
    fallback_enabled: bool = False
    user_visible_status: str = ""  # 사용자에게 보여줄 상태 메시지
    safe_stop_condition: str = ""  # safe stop 조건 설명


class FailureMatrix:
    """장애 유형별 정책 매트릭스.

    요구사항의 '장애 유형별 처리 행렬'을 코드로 구현한다.
    7가지 장애 유형(MODEL_TIMEOUT, STUCK_LOOP, BAD_TOOL_CALL 등)에 대해
    각각의 재시도/백오프/fallback 정책을 관리한다.

    Example:
        matrix = FailureMatrix()
        policy = matrix.get_policy(FailureType.MODEL_TIMEOUT)
        matrix.set_policy(FailurePolicy(
            failure_type=FailureType.EXTERNAL_API_ERROR,
            max_retries=5,
        ))
    """

    def __init__(self) -> None:
        self._policies: dict[FailureType, FailurePolicy] = {}
        self._init_default_policies()

    def _init_default_policies(self) -> None:
        """기본 장애 정책을 초기화한다."""
        defaults = [
            FailurePolicy(
                failure_type=FailureType.MODEL_TIMEOUT,
                max_retries=2,
                backoff_base=2.0,
                backoff_multiplier=2.0,
                backoff_max=30.0,
                fallback_enabled=True,
                user_visible_status="모델 응답 지연 -- 재시도 중...",
                safe_stop_condition="모든 재시도 및 fallback 모델 소진 시 중단",
            ),
            FailurePolicy(
                failure_type=FailureType.STUCK_LOOP,
                max_retries=0,
                backoff_base=0.0,
                backoff_multiplier=1.0,
                backoff_max=0.0,
                fallback_enabled=False,
                user_visible_status="반복 패턴 감지 -- 전략을 전환합니다.",
                safe_stop_condition="StallDetector exit_threshold 도달 시 루프 탈출",
            ),
            FailurePolicy(
                failure_type=FailureType.BAD_TOOL_CALL,
                max_retries=1,
                backoff_base=0.5,
                backoff_multiplier=1.0,
                backoff_max=2.0,
                fallback_enabled=False,
                user_visible_status="도구 호출 오류 -- 재작성 시도 중...",
                safe_stop_condition="재작성 재시도 소진 시 사용자에게 오류 보고",
            ),
            FailurePolicy(
                failure_type=FailureType.SUBAGENT_FAILURE,
                max_retries=1,
                backoff_base=1.0,
                backoff_multiplier=2.0,
                backoff_max=10.0,
                fallback_enabled=True,
                user_visible_status="하위 에이전트 실패 -- 다른 역할을 투입합니다.",
                safe_stop_condition="재시도 후에도 실패 시 상위 에이전트에 에스컬레이션",
            ),
            FailurePolicy(
                failure_type=FailureType.EXTERNAL_API_ERROR,
                max_retries=3,
                backoff_base=1.0,
                backoff_multiplier=2.0,
                backoff_max=30.0,
                fallback_enabled=False,
                user_visible_status="외부 API 오류 -- 재시도 중...",
                safe_stop_condition="4xx 클라이언트 오류는 즉시 중단, 5xx는 재시도 후 중단",
            ),
            FailurePolicy(
                failure_type=FailureType.MODEL_FALLBACK_NEEDED,
                max_retries=0,
                backoff_base=0.0,
                backoff_multiplier=1.0,
                backoff_max=0.0,
                fallback_enabled=True,
                user_visible_status="현재 모델 사용 불가 -- 대체 모델로 전환합니다.",
                safe_stop_condition="모든 fallback 모델 소진 시 중단",
            ),
            FailurePolicy(
                failure_type=FailureType.SAFE_STOP,
                max_retries=0,
                backoff_base=0.0,
                backoff_multiplier=1.0,
                backoff_max=0.0,
                fallback_enabled=False,
                user_visible_status="안전 중단이 요청되었습니다.",
                safe_stop_condition="AbortController 신호 수신 시 즉시 중단",
            ),
        ]
        for policy in defaults:
            self._policies[policy.failure_type] = policy

    def get_policy(self, failure_type: FailureType) -> FailurePolicy:
        """장애 유형에 해당하는 정책을 반환한다.

        등록되지 않은 유형이면 기본값(retry 1, backoff 1초)을 반환한다.

        Args:
            failure_type: 조회할 장애 유형.

        Returns:
            해당 장애 유형의 FailurePolicy.
        """
        return self._policies.get(
            failure_type,
            FailurePolicy(failure_type=failure_type),
        )

    def set_policy(self, policy: FailurePolicy) -> None:
        """장애 유형별 정책을 설정(오버라이드)한다.

        Args:
            policy: 설정할 FailurePolicy.
        """
        self._policies[policy.failure_type] = policy


# ── 지수 백오프 재시도 ──


@dataclass
class RetryWithBackoff:
    """지수 백오프 기반 재시도 래퍼.

    LLM 호출, 외부 API 호출 등에 사용한다.
    """

    max_retries: int = 2
    backoff_base: float = 1.0
    backoff_multiplier: float = 2.0
    backoff_max: float = 30.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)

    # 추적
    _attempt_count: int = field(default=0, repr=False)
    _last_error: Exception | None = field(default=None, repr=False)

    async def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """비동기 함수를 재시도 정책에 따라 실행한다.

        Args:
            fn: 실행할 비동기 함수
            *args, **kwargs: 함수 인자

        Returns:
            함수 반환값

        Raises:
            마지막 예외 (재시도 소진 시)
        """
        self._attempt_count = 0
        self._last_error = None

        for attempt in range(self.max_retries + 1):
            self._attempt_count = attempt + 1
            try:
                return await fn(*args, **kwargs)
            except self.retryable_exceptions as exc:
                self._last_error = exc
                if attempt >= self.max_retries:
                    logger.error(
                        "[RetryWithBackoff] 최종 실패 (시도 %d/%d): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    raise
                delay = self.calculate_delay(attempt)
                logger.warning(
                    "[RetryWithBackoff] 시도 %d/%d 실패: %s -- %.1f초 후 재시도",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        # max_retries < 0 등 비정상 케이스 방어
        raise RuntimeError("RetryWithBackoff: 재시도 루프가 비정상 종료됨")

    def calculate_delay(self, attempt: int) -> float:
        """attempt번째 재시도의 대기 시간(초)을 계산한다.

        Args:
            attempt: 현재 재시도 횟수 (0-based).

        Returns:
            대기 시간(초). backoff_max를 초과하지 않는다.
        """
        delay = self.backoff_base * (self.backoff_multiplier ** attempt)
        return min(delay, self.backoff_max)

    @property
    def attempt_count(self) -> int:
        """현재까지의 시도 횟수."""
        return self._attempt_count

    @property
    def last_error(self) -> Exception | None:
        """마지막으로 발생한 예외."""
        return self._last_error

    def reset(self) -> None:
        """상태를 초기화한다."""
        self._attempt_count = 0
        self._last_error = None


# ── 모델 Fallback 체인 ──


@dataclass
class ModelFallbackChain:
    """모델 fallback 체인.

    현재 모델이 실패하면 다음 순위 모델로 자동 전환한다.
    4-tier 체계와 연동: STRONG -> DEFAULT -> FAST.

    각 모델 호출에 asyncio.wait_for(timeout) 적용하여
    무한 대기를 방지한다. 모든 모델 실패 시 마지막 예외를 raise한다.

    Args:
        models: fallback 순서대로 정렬된 BaseChatModel 리스트.
        model_names: 로깅/추적용 모델 이름 리스트.
        timeout_per_model: 모델별 호출 타임아웃(초).
    """

    models: list[BaseChatModel] = field(default_factory=list)
    model_names: list[str] = field(default_factory=list)  # 로깅용 이름
    timeout_per_model: float = 60.0

    _current_index: int = field(default=0, repr=False)
    _fallback_history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    async def invoke_with_fallback(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> Any:
        """모델 체인을 따라 호출을 시도한다.

        첫 모델 실패 시 다음 모델로 전환, 모든 모델 실패 시 마지막 예외 raise.
        각 모델 호출에 asyncio.wait_for(timeout) 적용.
        """
        if not self.models:
            raise ValueError("ModelFallbackChain: 등록된 모델이 없습니다.")

        last_error: Exception | None = None

        for i, model in enumerate(self.models):
            self._current_index = i
            name = self.model_names[i] if i < len(self.model_names) else f"model_{i}"
            try:
                result = await asyncio.wait_for(
                    model.ainvoke(messages, **kwargs),
                    timeout=self.timeout_per_model,
                )
                logger.info("[ModelFallback] %s 호출 성공", name)
                return result
            except (asyncio.TimeoutError, Exception) as exc:
                last_error = exc
                self._fallback_history.append(
                    {
                        "model": name,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "timestamp": time.time(),
                    }
                )
                if i < len(self.models) - 1:
                    logger.warning(
                        "[ModelFallback] %s 실패, 다음 모델로 전환: %s",
                        name,
                        exc,
                    )
                else:
                    logger.error(
                        "[ModelFallback] 모든 모델 실패. 마지막 오류: %s",
                        exc,
                    )

        raise last_error  # type: ignore[misc]

    @property
    def current_model_name(self) -> str:
        """현재 사용 중인 모델의 이름."""
        if self.model_names and self._current_index < len(self.model_names):
            return self.model_names[self._current_index]
        return f"model_{self._current_index}"

    @property
    def fallback_history(self) -> list[dict[str, Any]]:
        """fallback 이력 (모델명, 오류, 타임스탬프)."""
        return list(self._fallback_history)

    def reset(self) -> None:
        """상태를 초기화한다."""
        self._current_index = 0
        self._fallback_history.clear()
