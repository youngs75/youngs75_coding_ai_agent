"""협력적 에이전트 실행 중단 제어.

Claude Code의 AbortController 패턴 포팅:
- asyncio.Event 기반 협력적 취소 신호
- Ctrl+C (SIGINT) → 우아한 중단
- 도구 실행/LLM 호출 중 체크포인트에서 중단 감지
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AbortReason(str, Enum):
    """중단 사유."""

    USER_INTERRUPT = "user_interrupt"  # Ctrl+C / SIGINT
    STALL_DETECTED = "stall_detected"  # StallDetector 강제 탈출
    BUDGET_EXCEEDED = "budget_exceeded"  # TurnBudgetTracker 한도 초과
    TURN_LIMIT = "turn_limit"  # 최대 턴 수 초과
    TIMEOUT = "timeout"  # 전체 실행 시간 초과


class AbortError(Exception):
    """중단 신호 발생 시 예외."""

    def __init__(
        self,
        reason: AbortReason | None = None,
        message: str = "",
    ) -> None:
        self.reason = reason
        self.abort_message = message
        super().__init__(message)


_DEFAULT_MESSAGES = {
    AbortReason.USER_INTERRUPT: "사용자가 실행을 중단했습니다.",
    AbortReason.STALL_DETECTED: "모델이 동일한 도구를 반복 호출하여 루프를 탈출합니다.",
    AbortReason.BUDGET_EXCEEDED: "토큰 예산을 초과하여 실행을 중단합니다.",
    AbortReason.TURN_LIMIT: "최대 턴 수에 도달하여 실행을 중단합니다.",
    AbortReason.TIMEOUT: "전체 실행 시간 제한을 초과했습니다.",
}


@dataclass
class AbortController:
    """협력적 중단 제어기.

    asyncio.Event 기반으로 외부 신호(Ctrl+C 등)를 에이전트 실행 루프에
    전달하여 우아하게 중단한다.

    사용 예:
        controller = AbortController()

        # 신호 핸들러에서:
        controller.abort(AbortReason.USER_INTERRUPT)

        # 에이전트 루프 내 체크포인트:
        controller.check_or_raise()  # AbortError 발생
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _reason: Optional[AbortReason] = field(default=None, repr=False)
    _message: str = field(default="", repr=False)

    @property
    def is_aborted(self) -> bool:
        """중단 신호가 발생했는지 확인한다."""
        return self._event.is_set()

    @property
    def reason(self) -> AbortReason | None:
        """중단 사유를 반환한다. 중단되지 않았으면 None."""
        return self._reason

    @property
    def message(self) -> str:
        """사용자에게 보여줄 중단 메시지."""
        return self._message

    def abort(self, reason: AbortReason, message: str = "") -> None:
        """중단 신호를 발생시킨다.

        Args:
            reason: 중단 사유
            message: 사용자 메시지 (비어 있으면 기본 메시지 사용)
        """
        self._reason = reason
        self._message = message or _DEFAULT_MESSAGES.get(
            reason, "알 수 없는 이유로 중단되었습니다."
        )
        self._event.set()
        logger.info(
            "[AbortController] 중단 신호: %s — %s",
            reason.value,
            self._message,
        )

    def check_or_raise(self) -> None:
        """중단 신호가 있으면 AbortError를 발생시킨다.

        에이전트 루프 내 동기 체크포인트에서 호출한다.

        Raises:
            AbortError: 중단 신호가 발생한 경우
        """
        if self._event.is_set():
            raise AbortError(self._reason, self._message)

    def reset(self) -> None:
        """새 턴을 위해 상태를 초기화한다."""
        self._event.clear()
        self._reason = None
        self._message = ""
