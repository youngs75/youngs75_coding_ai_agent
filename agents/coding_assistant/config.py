"""Coding Assistant 에이전트 설정.

Generator-Verifier 분리 원칙 (RubricRewards 논문):
- generation → strong 티어 (코드 생성용)
- verification → default 티어 (검증용)

레거시 환경변수(CODING_GEN_MODEL, CODING_VERIFY_MODEL)가 설정되면
티어 시스템보다 우선 적용된다.

MCP 도구 연동:
- code_tools: 파일 I/O, 코드 검색, 코드 실행 MCP 서버
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from pydantic import Field

from youngs75_a2a.core.config import BaseAgentConfig
from youngs75_a2a.core.model_tiers import (
    ModelTier,
    TierConfig,
    create_chat_model,
)


class CodingConfig(BaseAgentConfig):
    """Coding Assistant 에이전트 설정."""

    # 레거시 호환: 명시적 모델 오버라이드 (설정 시 티어보다 우선)
    generation_model: str | None = Field(
        default_factory=lambda: os.getenv("CODING_GEN_MODEL"),
        description="코드 생성용 모델 (명시적 오버라이드)",
    )
    verification_model: str | None = Field(
        default_factory=lambda: os.getenv("CODING_VERIFY_MODEL"),
        description="코드 검증용 모델 (명시적 오버라이드)",
    )

    # 허용 파일 확장자
    allowed_extensions: list[str] = Field(
        default=[".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".toml"],
    )
    # 최대 삭제 허용 줄 수
    max_delete_lines: int = Field(default=100)

    # MCP 서버
    mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "code_tools": os.getenv("CODE_TOOLS_MCP_URL", "http://localhost:3003/mcp/"),
        },
        description="MCP 서버 엔드포인트",
    )

    # ReAct 루프 최대 도구 호출 횟수
    max_tool_calls: int = Field(default=10)

    # Coding 전용 purpose → tier 매핑
    purpose_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "generation": ModelTier.STRONG,
            "tool_planning": ModelTier.FAST,
            "verification": ModelTier.DEFAULT,
            "parsing": ModelTier.FAST,
            "default": ModelTier.DEFAULT,
        },
    )

    def get_model(
        self,
        purpose: str = "default",
        *,
        structured: type | None = None,
    ) -> BaseChatModel:
        """목적별 모델 반환 — 레거시 오버라이드 우선, 없으면 티어 해석."""
        explicit = self._get_explicit_override(purpose)
        if explicit is not None:
            tier_config = self.get_tier_config(purpose)
            override_config = TierConfig(
                model=explicit,
                provider=tier_config.provider,
                context_window=tier_config.context_window,
                temperature=tier_config.temperature,
            )
            return create_chat_model(
                override_config,
                temperature=self.temperature,
                structured=structured,
            )
        return super().get_model(purpose, structured=structured)

    def _get_explicit_override(self, purpose: str) -> str | None:
        """purpose에 대한 명시적 모델 오버라이드를 반환한다."""
        if purpose == "generation" and self.generation_model:
            return self.generation_model
        if purpose == "verification" and self.verification_model:
            return self.verification_model
        return None
