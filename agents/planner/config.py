"""Planner Agent 설정."""

from __future__ import annotations

import os

from pydantic import Field

from youngs75_a2a.core.config import BaseAgentConfig
from youngs75_a2a.core.model_tiers import ModelTier


class PlannerConfig(BaseAgentConfig):
    """Planner Agent 설정.

    planning 목적에 STRONG 티어 모델을 사용한다.
    Read-only 도구만 허용 (코드 작성 금지).
    """

    mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "code_tools": os.getenv("CODE_TOOLS_MCP_URL", "http://localhost:3003/mcp/"),
        },
    )

    # Read-only 도구만 허용
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "read_file",
            "list_directory",
            "search_code",
        ],
    )

    # 복잡도 자동 판단 임계치
    simple_task_max_files: int = 2  # 이 이하면 simple로 판단

    def get_model(self, purpose: str = "default", **kwargs):
        """Planning 목적에는 REASONING 티어 사용 (Qwen3-Max)."""
        if purpose in ("planning", "default"):
            tier_config = self.model_tiers.get(ModelTier.REASONING)
            if tier_config:
                from youngs75_a2a.core.model_tiers import create_chat_model

                return create_chat_model(
                    tier_config,
                    temperature=self.temperature,
                )
        return super().get_model(purpose, **kwargs)
