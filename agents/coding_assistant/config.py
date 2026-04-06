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
        default=[
            # Python
            ".py", ".pyi",
            # JavaScript / TypeScript
            ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
            # Web frameworks
            ".vue", ".svelte",
            # Web
            ".html", ".css", ".scss", ".less",
            # Go
            ".go",
            # Rust
            ".rs",
            # Java / Kotlin
            ".java", ".kt", ".kts",
            # C / C++
            ".c", ".h", ".cpp", ".hpp", ".cc",
            # C# / .NET
            ".cs",
            # Ruby
            ".rb",
            # Swift
            ".swift",
            # Shell
            ".sh", ".bash", ".zsh",
            # Config / Data
            ".json", ".yaml", ".yml", ".toml", ".xml",
            ".md", ".txt", ".cfg", ".ini", ".env",
            # Build / Project
            ".gradle", ".cmake",
            # Docker
            ".dockerfile",
        ],
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

    # 다층 안전장치 설정 (Claude Code OS 패턴)
    stall_warn_threshold: int = Field(
        default=2,
        description="동일 도구+인자 반복 N회 시 경고 메시지 주입",
    )
    stall_exit_threshold: int = Field(
        default=3,
        description="동일 도구+인자 반복 N회 시 강제 루프 ��출",
    )
    max_llm_calls_per_turn: int = Field(
        default=15,
        description="턴당 최대 LLM 호출 횟수",
    )
    diminishing_streak_limit: int = Field(
        default=3,
        description="연속 저효율 호출 N회 시 중단",
    )
    min_delta_tokens: int = Field(
        default=500,
        description="유의미한 진전으로 판단하는 최소 토큰 수",
    )

    # Coding 전용 purpose → tier 매핑
    purpose_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "generation": ModelTier.STRONG,
            "tool_planning": ModelTier.DEFAULT,  # FAST → DEFAULT: Flash는 ���구 판단이 약함
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
