"""토큰 기반 다단계 메시지 컴팩션 미들웨어.

3종 코딩 에이전트(Claude Code, Codex, DeepAgents) 분석을 바탕으로 설계:

1단계 — 마이크로 컴팩트: 오래된 도구 결과를 [cleared]로 교체 (Claude Code 패턴)
2단계 — 윈도우 컴팩트: 토큰 예산 초과 시 첫 메시지 + 에러 메시지 + 최근 N개 유지
3단계 — 긴급 자르기: 그래도 초과하면 최근 메시지만 유지

핵심 원칙:
- 토큰 기반 트리거 (메시지 수가 아님)
- 에러 메시지 우선 보존
- 도구 결과는 단계적으로 정리
- 첫 사용자 메시지(원본 태스크)는 항상 보존
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from coding_agent.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

# 토큰 추정: 4 문자 ≈ 1 토큰 (Claude Code, DeepAgents 동일 근사치)
_CHARS_PER_TOKEN = 4

# 도구 결과 교체 메시지 (Claude Code 패턴)
_TOOL_RESULT_CLEARED = "[이전 도구 결과 — 컨텍스트 압축으로 제거됨]"


def _estimate_tokens(msg: BaseMessage) -> int:
    """메시지의 토큰 수를 추정한다."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    tokens = len(content) // _CHARS_PER_TOKEN + 1
    # 도구 호출 인자도 토큰에 포함
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            args = tc.get("args", {})
            tokens += len(str(args)) // _CHARS_PER_TOKEN
    return tokens


def _is_error_message(msg: BaseMessage) -> bool:
    """에러 주입 메시지인지 판별한다."""
    if not isinstance(msg, HumanMessage):
        return False
    content = msg.content if isinstance(msg.content, str) else ""
    return any(kw in content for kw in (
        "테스트가 실패했습니다",
        "에러를 분석하고",
        "에러 출력",
        "FAILED",
        "Error",
        "Traceback",
    ))


class MessageWindowMiddleware(AgentMiddleware):
    """토큰 기반 다단계 메시지 컴팩션 미들웨어.

    Args:
        max_context_tokens: 메시지에 할당할 최대 토큰 (시스템 프롬프트 제외)
        tool_result_max_tokens: 개별 도구 결과의 최대 토큰 (초과 시 truncation)
        keep_recent: 긴급 자르기 시 보존할 최근 메시지 수

    호환성:
        기존 max_turns, max_messages 인자도 수용하되 무시한다.
    """

    def __init__(
        self,
        max_context_tokens: int = 60_000,
        tool_result_max_tokens: int = 5_000,
        keep_recent: int = 6,
        # 기존 API 호환 — 무시됨
        max_turns: int | None = None,
        max_messages: int | None = None,
    ) -> None:
        self._max_context_tokens = max_context_tokens
        self._tool_result_max_tokens = tool_result_max_tokens
        self._keep_recent = keep_recent

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        messages = list(request.messages)
        total_tokens = sum(_estimate_tokens(m) for m in messages)

        if total_tokens <= self._max_context_tokens:
            return await handler(request)

        # ── 1단계: 마이크로 컴팩트 — 오래된 도구 결과 정리 ──
        messages = self._micro_compact(messages)
        total_tokens = sum(_estimate_tokens(m) for m in messages)

        if total_tokens <= self._max_context_tokens:
            logger.info(
                "[MessageWindow] 마이크로 컴팩트 적용: %d tokens",
                total_tokens,
            )
            return await handler(request.override(messages=messages))

        # ── 2단계: 윈도우 컴팩트 — 첫 메시지 + 에러 + 최근 유지 ──
        messages = self._window_compact(messages)
        total_tokens = sum(_estimate_tokens(m) for m in messages)

        if total_tokens <= self._max_context_tokens:
            logger.info(
                "[MessageWindow] 윈도우 컴팩트 적용: %d tokens",
                total_tokens,
            )
            return await handler(request.override(messages=messages))

        # ── 3단계: 긴급 자르기 — 최근 메시지만 ──
        messages = self._emergency_trim(messages)
        total_tokens = sum(_estimate_tokens(m) for m in messages)
        logger.warning(
            "[MessageWindow] 긴급 자르기 적용: %d tokens, %d messages",
            total_tokens,
            len(messages),
        )
        return await handler(request.override(messages=messages))

    def _micro_compact(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """1단계: 오래된 도구 결과를 [cleared]로 교체.

        Claude Code 패턴: 최근 도구 결과는 보존, 오래된 것만 정리.
        DeepAgents 패턴: 도구 결과 20K 토큰 상한.
        """
        result: list[BaseMessage] = []
        # 최근 keep_recent 메시지는 도구 결과도 보존
        preserve_from = max(0, len(messages) - self._keep_recent)

        for idx, msg in enumerate(messages):
            if idx >= preserve_from:
                # 최근 메시지: 도구 결과 truncation만 적용
                if isinstance(msg, ToolMessage):
                    msg = self._truncate_tool_result(msg)
                result.append(msg)
                continue

            if isinstance(msg, ToolMessage):
                tokens = _estimate_tokens(msg)
                if tokens > self._tool_result_max_tokens:
                    # 큰 도구 결과 → cleared
                    result.append(
                        ToolMessage(
                            content=_TOOL_RESULT_CLEARED,
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", ""),
                        )
                    )
                    continue
            result.append(msg)

        return result

    def _window_compact(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """2단계: 첫 메시지 + 에러 메시지 + 최근 N개 유지.

        Codex 패턴: base_instructions 항상 보존.
        Claude Code 패턴: 에러 컨텍스트 우선 보존.
        """
        if len(messages) <= self._keep_recent + 2:
            return messages

        first_msg = messages[0]

        # 에러 메시지 수집 (중간에 있는 것도 보존)
        error_msgs: list[BaseMessage] = []
        middle = messages[1:-self._keep_recent]
        for msg in middle:
            if _is_error_message(msg):
                error_msgs.append(msg)

        # 최근 메시지
        tail = messages[-self._keep_recent:]

        # 에러 메시지가 이미 tail에 포함되어 있으면 중복 제거
        tail_ids = {id(m) for m in tail}
        unique_errors = [m for m in error_msgs if id(m) not in tail_ids]

        removed_count = len(messages) - 1 - len(unique_errors) - len(tail)
        if removed_count > 0:
            summary = HumanMessage(
                content=f"[이전 {removed_count}개 메시지 압축됨 — 원본 요청과 에러 컨텍스트는 보존]"
            )
            return [first_msg, summary] + unique_errors + tail

        return messages

    def _emergency_trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """3단계: 토큰 예산 내로 최근 메시지만 유지.

        Codex 패턴: FIFO — 오래된 메시지부터 제거.
        """
        first_msg = messages[0]
        first_tokens = _estimate_tokens(first_msg)
        budget = self._max_context_tokens - first_tokens - 100  # 여유분

        # 뒤에서부터 예산 내로 메시지 수집
        kept: list[BaseMessage] = []
        used = 0
        for msg in reversed(messages[1:]):
            msg_tokens = _estimate_tokens(msg)
            if used + msg_tokens > budget:
                break
            kept.append(msg)
            used += msg_tokens

        kept.reverse()
        removed = len(messages) - 1 - len(kept)
        summary = HumanMessage(
            content=f"[이전 {removed}개 메시지 제거됨 — 토큰 예산 초과]"
        )
        return [first_msg, summary] + kept

    def _truncate_tool_result(self, msg: ToolMessage) -> ToolMessage:
        """개별 도구 결과가 상한을 초과하면 truncation.

        DeepAgents 패턴: 20K 토큰 상한 + [truncated] 안내.
        """
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        max_chars = self._tool_result_max_tokens * _CHARS_PER_TOKEN
        if len(content) <= max_chars:
            return msg
        truncated = content[:max_chars] + f"\n\n... [결과 잘림: {len(content)}자 중 {max_chars}자만 유지]"
        return ToolMessage(
            content=truncated,
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", ""),
        )
