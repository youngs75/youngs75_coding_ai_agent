"""Verification Agent 설정."""

from __future__ import annotations

import os

from pydantic import Field

from youngs75_a2a.core.config import BaseAgentConfig
from youngs75_a2a.core.model_tiers import ModelTier


class VerifierConfig(BaseAgentConfig):
    """Verification Agent 설정."""

    # MCP 서버 (code_tools의 run_python 등 사용)
    mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "code_tools": os.getenv("CODE_TOOLS_MCP_URL", "http://localhost:3003/mcp/"),
        },
        description="MCP 서버 엔드포인트",
    )

    # 검증 활성화 플래그
    enable_lint: bool = Field(default=True, description="lint 검증 활성화")
    enable_test: bool = Field(default=True, description="test 검증 활성화")
    enable_llm_review: bool = Field(default=True, description="LLM 리뷰 검증 활성화")

    # 타임아웃 (초)
    lint_timeout: int = Field(default=30, description="lint 실행 타임아웃")
    test_timeout: int = Field(default=60, description="test 실행 타임아웃")
    review_timeout: int = Field(default=30, description="LLM 리뷰 타임아웃")

    # LLM 리뷰 코드 최대 길이 (토큰 폭발 방지)
    max_review_chars: int = Field(default=4000, description="LLM 리뷰에 전달할 최대 코드 길이")

    # purpose → tier 매핑
    purpose_tiers: dict[str, str] = Field(
        default_factory=lambda: {
            "review": ModelTier.DEFAULT,
            "default": ModelTier.DEFAULT,
        },
    )
