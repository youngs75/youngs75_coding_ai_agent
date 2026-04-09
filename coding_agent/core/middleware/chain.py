"""미들웨어 체인 실행기.

미들웨어 리스트를 양파(Onion) 패턴으로 실행한다.
리스트의 첫 번째 미들웨어가 가장 바깥쪽(먼저 실행, 마지막 반환).
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel

from .base import AgentMiddleware, Handler, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

# 기본 LLM 호출 타임아웃 (초) — metadata["request_timeout"]으로 오버라이드 가능
_DEFAULT_LLM_TIMEOUT: float = 120.0


class MiddlewareChain:
    """미들웨어 리스트를 양파(Onion) 패턴으로 실행한다.

    리스트의 첫 번째 미들웨어가 가장 바깥쪽(먼저 실행, 마지막 반환)이며,
    가장 안쪽 핸들러가 실제 LLM을 호출한다.

    Args:
        middlewares: 초기 미들웨어 리스트. None이면 빈 체인.

    Example:
        chain = MiddlewareChain([
            MessageWindowMiddleware(max_turns=4),
            SummarizationMiddleware(threshold=0.85),
            SkillMiddleware(skill_registry),
        ])
        request = ModelRequest(system_message="...", messages=[...])
        response = await chain.invoke(request, llm)
    """

    def __init__(self, middlewares: list[AgentMiddleware] | None = None) -> None:
        self._middlewares: list[AgentMiddleware] = list(middlewares or [])

    def add(self, middleware: AgentMiddleware) -> "MiddlewareChain":
        """미들웨어를 체인 끝에 추가한다.

        Args:
            middleware: 추가할 미들웨어.

        Returns:
            self (메서드 체이닝 지원).
        """
        self._middlewares.append(middleware)
        return self

    @property
    def middlewares(self) -> list[AgentMiddleware]:
        return list(self._middlewares)

    async def invoke(
        self,
        request: ModelRequest,
        llm: BaseChatModel,
    ) -> ModelResponse:
        """미들웨어 체인을 통해 LLM을 호출한다.

        1. 미들웨어를 역순으로 래핑하여 양파 구조를 만든다
        2. 가장 안쪽 핸들러가 실제 LLM을 호출한다
        3. 각 미들웨어는 request를 수정하고 handler()를 호출한다

        Args:
            request: LLM 호출 요청.
            llm: LangChain 기반 LLM 모델 인스턴스.

        Returns:
            미들웨어 체인을 거친 LLM 응답.
        """

        # 가장 안쪽: 실제 LLM 호출 (타임아웃 보호)
        async def _final_handler(req: ModelRequest) -> ModelResponse:
            all_messages = req.all_messages
            timeout = req.metadata.get("request_timeout", _DEFAULT_LLM_TIMEOUT)
            logger.debug(
                "[MiddlewareChain] LLM 호출: model=%s, messages=%d, timeout=%.0fs",
                req.model_name,
                len(all_messages),
                timeout,
            )
            try:
                response = await asyncio.wait_for(
                    llm.ainvoke(all_messages),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[MiddlewareChain] LLM 타임아웃 (%s, %.0fs 초과)",
                    req.model_name,
                    timeout,
                )
                raise
            return ModelResponse(message=response)

        # 미들웨어를 역순으로 래핑 (양파 패턴)
        handler: Handler = _final_handler
        for middleware in reversed(self._middlewares):
            handler = _make_wrapper(middleware, handler)

        return await handler(request)


def _make_wrapper(middleware: AgentMiddleware, next_handler: Handler) -> Handler:
    """클로저 변수 캡처 문제를 방지하기 위한 래퍼 팩토리."""

    async def _wrapped(request: ModelRequest) -> ModelResponse:
        return await middleware.wrap_model_call(request, next_handler)

    return _wrapped
