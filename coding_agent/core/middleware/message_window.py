"""메시지 슬라이딩 윈도우 미들웨어.

검증 재시도 루프에서 이전 생성 코드(22K chars)가 messages에 누적되는 문제를 해결한다.
최근 N턴만 유지하고, 그 이전 메시지는 요약으로 대체한다.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from coding_agent.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)


class MessageWindowMiddleware(AgentMiddleware):
    """최근 N턴만 유지하는 슬라이딩 윈도우 미들웨어.

    검증 재시도 루프에서 messages가 폭발적으로 늘어나는 것을 방지한다.
    첫 번째 사용자 메시지(원본 태스크)는 항상 보존한다.
    """

    def __init__(self, max_turns: int = 4, max_messages: int = 10) -> None:
        self._max_turns = max_turns
        self._max_messages = max_messages

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        messages = request.messages

        # 턴 수 계산: HumanMessage 개수 = 턴 수
        turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        msg_count = len(messages)

        if msg_count <= self._max_messages and turn_count <= self._max_turns:
            return await handler(request)

        # 윈도우 적용: 첫 메시지(원본 태스크) + 최근 max_turns*2 메시지
        keep_tail = self._max_turns * 2
        if keep_tail >= len(messages):
            return await handler(request)

        first_msg = messages[0]
        tail_msgs = messages[-keep_tail:]
        removed_count = len(messages) - 1 - keep_tail  # 첫 메시지 제외

        summary = HumanMessage(
            content=f"[이전 {removed_count}개 메시지 생략 — 최근 대화만 유지]"
        )

        trimmed = [first_msg, summary] + tail_msgs
        logger.info(
            "[MessageWindow] %d개 → %d개로 축소 (턴=%d, 제거=%d)",
            msg_count,
            len(trimmed),
            turn_count,
            removed_count,
        )

        return await handler(request.override(messages=trimmed))
