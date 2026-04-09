"""요약 기반 컨텍스트 컴팩션 미들웨어.

Phase 1 (경량): 큰 tool args 자동 축소 — AIMessage의 tool_calls 인자가 길면 잘라냄
Phase 2 (중량): 토큰 임계치 초과 시 오래된 메시지를 규칙 기반으로 요약/제거
"""

from __future__ import annotations

import copy
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from coding_agent.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    """메시지 리스트의 토큰 수를 대략 추정한다.

    한국어 ≈ 0.4 tokens/char, 코드 ≈ 0.25 tokens/char.
    혼합 텍스트에 대해 0.35를 사용한다.
    """
    total_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
    return int(total_chars * 0.35)


class SummarizationMiddleware(AgentMiddleware):
    """2-phase 요약 미들웨어.

    Phase 1: AIMessage의 tool_calls 인자가 max_tool_arg_chars를 초과하면 잘라냄.
    Phase 2: 전체 토큰이 token_threshold를 초과하면 오래된 메시지를 제거하고 요약 삽입.
    """

    def __init__(
        self,
        token_threshold: int = 100_000,
        keep_recent_messages: int = 6,
        max_tool_arg_chars: int = 2000,
    ) -> None:
        self._token_threshold = token_threshold
        self._keep_recent = keep_recent_messages
        self._max_tool_arg_chars = max_tool_arg_chars

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        messages = list(request.messages)
        modified = False

        # Phase 1: tool_calls 인자 축소
        for i, msg in enumerate(messages):
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                args = tc.get("args", {})
                if not isinstance(args, dict):
                    continue
                for key, value in args.items():
                    if isinstance(value, str) and len(value) > self._max_tool_arg_chars:
                        if not modified:
                            # 첫 수정 시 messages를 deep copy
                            messages = copy.deepcopy(messages)
                            modified = True
                        truncated = (
                            value[:200]
                            + f"... ({len(value)}자 생략)"
                        )
                        messages[i].tool_calls[
                            msg.tool_calls.index(tc)
                        ]["args"][key] = truncated
                        logger.debug(
                            "[Summarization] Phase1: tool_call arg '%s' 축소 (%d → %d chars)",
                            key,
                            len(value),
                            len(truncated),
                        )

        # Phase 2: 토큰 임계치 초과 시 오래된 메시지 제거
        estimated = _estimate_tokens(messages)
        if estimated > self._token_threshold and len(messages) > self._keep_recent + 1:
            first_msg = messages[0]
            tail_msgs = messages[-self._keep_recent:]
            removed_count = len(messages) - 1 - self._keep_recent

            summary = HumanMessage(
                content=(
                    f"[컨텍스트 축소: {removed_count}개 메시지 제거, "
                    f"최근 {self._keep_recent}개 유지]"
                )
            )

            messages = [first_msg, summary] + tail_msgs
            modified = True
            logger.info(
                "[Summarization] Phase2: 토큰 %d (임계=%d), %d개 메시지 제거",
                estimated,
                self._token_threshold,
                removed_count,
            )

        if modified:
            return await handler(request.override(messages=messages))
        return await handler(request)
