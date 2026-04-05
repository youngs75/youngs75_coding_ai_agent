"""ReAct 루프 반복 도구 호출 감지기.

LLM이 동일한 도구를 동일한 인자로 반복 호출하는 패턴을 감지하여
무한루프를 사전에 탈출한다. Claude Code의 다층 안전장치 패턴에서 영감.

Qwen 등 약한 모델이 stall 패턴에 빠지기 쉬운 문제를 해결한다.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StallAction(str, Enum):
    """반복 감지 결과."""

    CONTINUE = "continue"  # 정상 — 계속 진��
    WARN = "warn"  # 경고 — 시스템 메시지 주입 후 계속
    FORCE_EXIT = "force_exit"  # 강제 탈출 — ReAct 루프 종료


@dataclass
class _StallRecord:
    """도구 호출 기록."""

    tool_name: str
    args_hash: str


@dataclass
class StallDetector:
    """반복 도구 호출 감지기.

    (tool_name, hash(args)) 쌍의 빈도를 슬라이딩 윈도우로 추적한다.

    Args:
        warn_threshold: 동일 호출 N회 시 경고 (기본: 2)
        exit_threshold: 동일 호출 N회 시 강제 탈출 (기본: 3)
        window_size: 슬라이딩 윈도우 크기 (기본: 10)
    """

    warn_threshold: int = 2
    exit_threshold: int = 3
    window_size: int = 10

    _history: deque[_StallRecord] = field(
        default_factory=lambda: deque(maxlen=10),
        repr=False,
    )
    _call_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.window_size)

    def record_and_check(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
    ) -> StallAction:
        """도구 호출을 기록하고 반복 패턴을 체크한���.

        Args:
            tool_name: 도구 이름
            tool_args: 도구 인자

        Returns:
            StallAction — CONTINUE, WARN, FORCE_EXIT
        """
        if not tool_name:
            return StallAction.CONTINUE

        args_hash = self._hash_args(tool_args or {})
        record = _StallRecord(tool_name=tool_name, args_hash=args_hash)
        key = f"{tool_name}:{args_hash}"

        self._history.append(record)
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        count = self._call_counts[key]

        if count >= self.exit_threshold:
            logger.warning(
                "[StallDetector] FORCE_EXIT: %s — 동일 인자 %d회 반복",
                tool_name,
                count,
            )
            return StallAction.FORCE_EXIT

        if count >= self.warn_threshold:
            logger.warning(
                "[StallDetector] WARN: %s — 동일 인자 %d회 반복",
                tool_name,
                count,
            )
            return StallAction.WARN

        return StallAction.CONTINUE

    def get_stall_summary(self) -> str:
        """감지된 반복 패턴의 사용자 메시지를 반환한다."""
        if not self._call_counts:
            return ""
        worst_key = max(self._call_counts, key=lambda k: self._call_counts[k])
        count = self._call_counts[worst_key]
        tool_name = worst_key.split(":")[0]
        return (
            f"도구 '{tool_name}'이(가) 동일한 인자로 {count}회 반복 호출되었습니다. "
            f"수집된 정보를 바탕으로 응답을 생성합니다."
        )

    def reset(self) -> None:
        """턴 시작 시 상태를 초기화한다."""
        self._history.clear()
        self._call_counts.clear()

    @staticmethod
    def _hash_args(args: dict[str, Any]) -> str:
        """도구 인자의 안정적인 해시를 ���성한다."""
        try:
            serialized = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = str(args)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
