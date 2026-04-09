"""Planner Agent 설정."""

from __future__ import annotations

import os

from pydantic import Field

from coding_agent.core.config import BaseAgentConfig
from coding_agent.core.model_tiers import ModelTier


class PlannerConfig(BaseAgentConfig):
    """Planner Agent 설정.

    planning 목적에 STRONG 티어 모델을 사용한다.
    Read-only 도구만 허용 (코드 작성 금지).
    웹 검색 도구로 외부 API 문서를 조사할 수 있다.
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

    # 웹 검색 MCP 서버 (통합 code_tools 서버에 검색 도구 포함)
    web_search_mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "code_tools": os.getenv("CODE_TOOLS_MCP_URL", "http://localhost:3003/mcp/"),
        },
    )

    # 웹 검색에서 허용하는 도구
    web_search_tools: list[str] = Field(
        default_factory=lambda: [
            "search_web",       # Tavily
            "google_search",    # Serper
        ],
    )

    # 리서치 당 최대 웹 검색 횟수
    max_research_searches: int = 3

    # 복잡도 자동 판단 임계치
    simple_task_max_files: int = 2  # 이 이하면 simple로 판단

    def get_model(self, purpose: str = "default", **kwargs):
        """Planning 목적에는 REASONING 티어 사용 (Qwen3-Max)."""
        if purpose in ("planning", "default"):
            tier_config = self.model_tiers.get(ModelTier.REASONING)
            if tier_config:
                from coding_agent.core.model_tiers import create_chat_model

                return create_chat_model(
                    tier_config,
                    temperature=self.temperature,
                )
        return super().get_model(purpose, **kwargs)
