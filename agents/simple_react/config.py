"""Simple ReAct 에이전트 설정."""

from __future__ import annotations

import os
from pydantic import Field
from youngs75_a2a.core.config import BaseAgentConfig


class SimpleReActConfig(BaseAgentConfig):
    """Simple MCP ReAct 에이전트 설정."""

    system_prompt: str = Field(
        default=(
            "당신은 도구를 활용하는 검색 전문가입니다. "
            "필요 시 가진 도구를 사용해 답변하세요. "
            "결과가 부족하면 모른다고 답하고 도구의 결과를 활용하여 출처를 꼭 제공하세요."
        ),
    )
    mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "code_tools": os.getenv("CODE_TOOLS_MCP_URL", "http://localhost:3003/mcp/"),
        },
    )
