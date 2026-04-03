"""메모리 시스템 스키마 정의.

CoALA 논문 기반 4종 메모리 타입:
- Working: 현재 대화 컨텍스트 (= messages)
- Episodic: 실행 결과 이력 (세션 스코프)
- Semantic: 도메인 지식/규칙 (프로젝트 컨벤션 등)
- Procedural: 학습된 스킬 패턴 (Voyager식 누적)

Agent-as-a-Judge 교훈:
  Episodic/Procedural은 오류 전파 위험이 있으므로
  초기에는 Working + Semantic만 활성화한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """CoALA 메모리 타입."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryItem(BaseModel):
    """단일 메모리 항목."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    type: MemoryType
    content: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None
    score: float = 0.0

    def matches_tags(self, query_tags: list[str]) -> bool:
        """주어진 태그 중 하나라도 일치하면 True."""
        if not query_tags:
            return True
        return bool(set(self.tags) & set(query_tags))
