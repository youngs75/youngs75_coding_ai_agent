"""턴별 LLM 호출 예산 및 감소수익 감지기.

Claude Code의 tokenBudget 패턴 포팅:
- 절대 한도: 턴당 최대 LLM 호출 횟수
- 감소수익: 연속 N회 저효율 호출 시 자동 중단

LLM 호출마다 출력 토큰을 기록하고 진전이 없으면 루프를 종료한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BudgetVerdict(str, Enum):
    """예산 체크 결과."""

    OK = "ok"  # 정상 — 계속 진행
    WARN_DIMINISHING = "warn_diminishing"  # 감소수익 경고
    STOP = "stop"  # 예산 초과 또는 감소수익 한계 — 중단


@dataclass
class TurnBudgetTracker:
    """턴별 LLM 호출 예산 추적기.

    Args:
        max_llm_calls: 턴당 최대 LLM 호출 횟수 (기본: 15)
        diminishing_streak_limit: 연속 저효율 호출 N회 시 중단 (기본: 3)
        min_delta_tokens: 유의미한 진전으로 판단하는 최소 토큰 수 (기본: 500)
    """

    max_llm_calls: int = 15
    diminishing_streak_limit: int = 3
    min_delta_tokens: int = 500

    _llm_call_count: int = field(default=0, repr=False)
    _diminishing_streak: int = field(default=0, repr=False)
    _total_output_tokens: int = field(default=0, repr=False)

    def record_llm_call(self, output_tokens: int) -> BudgetVerdict:
        """LLM 호출을 기록하고 예산을 체크한다.

        Args:
            output_tokens: 이번 LLM 응답의 출력 토큰 수

        Returns:
            BudgetVerdict — OK, WARN_DIMINISHING, STOP
        """
        self._llm_call_count += 1
        self._total_output_tokens += output_tokens

        # 절대 한도 체크
        if self._llm_call_count >= self.max_llm_calls:
            logger.warning(
                "[TurnBudget] 최대 LLM 호출 도달: %d/%d",
                self._llm_call_count,
                self.max_llm_calls,
            )
            return BudgetVerdict.STOP

        # 감소수익 감지
        if output_tokens < self.min_delta_tokens:
            self._diminishing_streak += 1
        else:
            self._diminishing_streak = 0

        if self._diminishing_streak >= self.diminishing_streak_limit:
            logger.warning(
                "[TurnBudget] 감소수익 감지: 연속 %d회 저효율 호출 (<%d 토큰)",
                self._diminishing_streak,
                self.min_delta_tokens,
            )
            return BudgetVerdict.STOP

        if self._diminishing_streak >= 2:
            return BudgetVerdict.WARN_DIMINISHING

        return BudgetVerdict.OK

    @property
    def llm_call_count(self) -> int:
        """현재까지의 LLM 호출 횟수."""
        return self._llm_call_count

    @property
    def total_output_tokens(self) -> int:
        """현재까지의 총 출력 토큰 수."""
        return self._total_output_tokens

    def get_summary(self) -> str:
        """현재 예산 상태의 사용자 메시지를 반환한다."""
        return (
            f"LLM 호출: {self._llm_call_count}/{self.max_llm_calls}, "
            f"총 출력 토큰: {self._total_output_tokens:,}, "
            f"저효율 연속: {self._diminishing_streak}/{self.diminishing_streak_limit}"
        )

    def reset(self) -> None:
        """새 턴 시작 시 상태를 초기화한다."""
        self._llm_call_count = 0
        self._diminishing_streak = 0
        self._total_output_tokens = 0
