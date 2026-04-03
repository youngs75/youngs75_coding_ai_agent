"""Skills 시스템 스키마.

3-Level Progressive Loading:
  L1 — 메타데이터만 (name, description, tags). 항상 컨텍스트에 주입.
  L2 — 본문 (prompt body). 스킬 활성화 시 온디맨드 로드.
  L3 — 참조 파일 (references). 외부 파일/문서를 스킬 실행 시 로드.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SkillLevel(IntEnum):
    """스킬 로딩 레벨."""

    L1_METADATA = 1
    L2_BODY = 2
    L3_REFERENCES = 3


class SkillMetadata(BaseModel):
    """L1: 스킬 메타데이터 — 항상 컨텍스트에 주입."""

    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    enabled: bool = True


class SkillReference(BaseModel):
    """L3: 스킬 참조 파일."""

    path: str
    description: str = ""
    content: str | None = None  # 로드 후 채워짐


class Skill(BaseModel):
    """스킬 전체 정의 (L1 + L2 + L3)."""

    metadata: SkillMetadata
    body: str | None = None  # L2: 프롬프트 본문 (None이면 미로드)
    references: list[SkillReference] = Field(default_factory=list)  # L3
    source_path: Path | None = None  # 스킬 파일 원본 경로
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def loaded_level(self) -> SkillLevel:
        """현재 로드된 레벨."""
        if any(ref.content is not None for ref in self.references):
            return SkillLevel.L3_REFERENCES
        if self.body is not None:
            return SkillLevel.L2_BODY
        return SkillLevel.L1_METADATA

    def as_context_entry(self) -> str:
        """L1 메타데이터를 컨텍스트 주입용 문자열로 반환."""
        tags_str = ", ".join(self.metadata.tags) if self.metadata.tags else "none"
        return f"[{self.metadata.name}] {self.metadata.description} (tags: {tags_str})"
