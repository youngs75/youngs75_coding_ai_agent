"""ResilienceMiddleware — 재시도, fallback, abort 체크포인트 통합.

기존 dead code(FailureMatrix, RetryWithBackoff, ModelFallbackChain,
AbortController)를 에이전트 루프에 연결하는 핵심 미들웨어.

양파 패턴:
  before (abort check) → handler (retry + fallback) → after (abort check)
"""

from __future__ import annotations

import logging
from typing import Callable


from coding_agent.core.abort_controller import AbortController
from coding_agent.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
)
from coding_agent.core.resilience import (
    FailureMatrix,
    FailurePolicy,
    FailureType,
    ModelFallbackChain,
    RetryWithBackoff,
)

logger = logging.getLogger(__name__)

# purpose → FailureType 매핑
_PURPOSE_FAILURE_MAP: dict[str, FailureType] = {
    "generation": FailureType.MODEL_TIMEOUT,
    "planning": FailureType.MODEL_TIMEOUT,
    "verification": FailureType.MODEL_TIMEOUT,
    "parsing": FailureType.MODEL_TIMEOUT,
    "tool_planning": FailureType.MODEL_TIMEOUT,
    "default": FailureType.MODEL_TIMEOUT,
}

# fallback 체인 빌더 타입: () → ModelFallbackChain
FallbackChainBuilder = Callable[[], ModelFallbackChain]


class ResilienceMiddleware(AgentMiddleware):
    """복원력 미들웨어 — 재시도 + fallback + abort 체크포인트.

    AbortController, RetryWithBackoff, ModelFallbackChain, FailureMatrix를
    MiddlewareChain에 통합하여 LLM 호출의 복원력을 보장한다.

    Args:
        abort_controller: 중단 신호 제어기.
        failure_matrix: 장애 유형별 정책 매트릭스.
        fallback_chain_builder: ModelFallbackChain을 생성하는 팩토리 함수.
            None이면 fallback 없이 retry만 수행.
    """

    def __init__(
        self,
        abort_controller: AbortController,
        failure_matrix: FailureMatrix | None = None,
        fallback_chain_builder: FallbackChainBuilder | None = None,
    ) -> None:
        self._abort = abort_controller
        self._matrix = failure_matrix or FailureMatrix()
        self._fallback_builder = fallback_chain_builder

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        """abort 체크 → retry/fallback 래핑 → abort 체크.

        1. before: AbortController.check_or_raise()
        2. FailureMatrix에서 purpose별 정책 조회
        3. RetryWithBackoff로 handler 호출 래핑
        4. 모든 재시도 실패 시 ModelFallbackChain으로 대체 모델 시도
        5. after: AbortController.check_or_raise()

        Args:
            request: LLM 호출 요청.
            handler: 다음 미들웨어 또는 최종 LLM 호출.

        Returns:
            LLM 응답.

        Raises:
            AbortError: 중단 신호가 발생한 경우.
        """
        # before: abort 체크
        self._abort.check_or_raise()

        purpose = request.metadata.get("purpose", "default")
        policy = self._resolve_policy(purpose)

        retry = RetryWithBackoff(
            max_retries=policy.max_retries,
            backoff_base=policy.backoff_base,
            backoff_multiplier=policy.backoff_multiplier,
            backoff_max=policy.backoff_max,
        )

        try:
            response = await retry.execute(handler, request)
        except Exception as exc:
            logger.warning(
                "[Resilience] 재시도 소진 (purpose=%s): %s",
                purpose,
                exc,
            )
            # fallback 시도
            if policy.fallback_enabled and self._fallback_builder:
                response = await self._try_fallback(request, exc)
            else:
                raise

        # after: abort 체크
        self._abort.check_or_raise()

        return response

    async def _try_fallback(
        self,
        request: ModelRequest,
        original_error: Exception,
    ) -> ModelResponse:
        """ModelFallbackChain으로 대체 모델을 시도한다.

        Args:
            request: 원래 LLM 호출 요청.
            original_error: 원래 발생한 예외.

        Returns:
            fallback 모델의 ModelResponse.

        Raises:
            original_error: fallback_builder가 없는 경우.
            Exception: 모든 fallback 모델 실패 시 마지막 예외.
        """
        if not self._fallback_builder:
            raise original_error

        chain = self._fallback_builder()
        logger.info(
            "[Resilience] fallback 체인 시도: %s",
            chain.model_names,
        )
        message = await chain.invoke_with_fallback(request.all_messages)
        return ModelResponse(message=message)

    def _resolve_policy(self, purpose: str) -> FailurePolicy:
        """purpose에 해당하는 FailurePolicy를 반환한다.

        Args:
            purpose: 호출 목적.

        Returns:
            해당 장애 유형의 정책.
        """
        failure_type = _PURPOSE_FAILURE_MAP.get(
            purpose, FailureType.MODEL_TIMEOUT
        )
        return self._matrix.get_policy(failure_type)
