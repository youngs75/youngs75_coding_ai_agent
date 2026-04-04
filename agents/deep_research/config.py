"""Deep Research 에이전트 설정."""

from __future__ import annotations

import os
from pydantic import Field
from youngs75_a2a.core.config import BaseAgentConfig


class ResearchConfig(BaseAgentConfig):
    """Deep Research 에이전트 설정.

    환경변수로 오버라이드 가능하며, LangGraph RunnableConfig에서도 추출 가능하다.
    """

    # --- 실행 제어 ---
    allow_clarification: bool = Field(
        default_factory=lambda: (
            os.getenv("ALLOW_CLARIFICATION", "true").lower() == "true"
        ),
    )
    max_concurrent_research_units: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_RESEARCH", "3")),
    )
    max_structured_output_retries: int = Field(default=3)

    # --- 연구 프로세스 ---
    max_researcher_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESEARCHER_ITERATIONS", "3")),
    )
    max_react_tool_calls: int = Field(default=5)
    researcher_min_iterations_before_compress: int = Field(
        default_factory=lambda: int(
            os.getenv("RESEARCHER_MIN_ITERATIONS_BEFORE_COMPRESS", "1")
        ),
    )
    supervisor_force_conduct_research_enabled: bool = Field(
        default_factory=lambda: (
            os.getenv("SUPERVISOR_FORCE_CONDUCT_RESEARCH_ENABLED", "true").lower()
            == "true"
        ),
    )
    supervisor_force_conduct_research_until_iteration: int = Field(
        default_factory=lambda: int(
            os.getenv("SUPERVISOR_FORCE_CONDUCT_RESEARCH_UNTIL", "1")
        ),
    )
    supervisor_research_grace_seconds: float = Field(default=0.0)

    # --- HITL ---
    enable_hitl: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_HITL", "false").lower() == "true",
    )
    max_revision_loops: int = Field(default=2)

    # --- LLM 모델 (용도별) ---
    research_model: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "deepseek/deepseek-v3.2"),
    )
    compression_model: str = Field(
        default_factory=lambda: os.getenv(
            "COMPRESSION_MODEL", "deepseek/deepseek-v3.2"
        ),
    )
    final_report_model: str = Field(
        default_factory=lambda: os.getenv(
            "FINAL_REPORT_MODEL", "deepseek/deepseek-v3.2"
        ),
    )

    # --- MCP 서버 ---
    mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "arxiv": os.getenv("ARXIV_MCP_URL", "http://localhost:3000/mcp/"),
            "tavily": os.getenv("TAVILY_MCP_URL", "http://localhost:3001/mcp/"),
            "serper": os.getenv("SERPER_MCP_URL", "http://localhost:3002/mcp/"),
        },
    )

    # --- A2A 에이전트 ---
    a2a_agent_endpoints: dict[str, str] = Field(
        default_factory=lambda: {
            "supervisor": os.getenv("SUPERVISOR_A2A_URL", "http://localhost:8092"),
        },
    )

    def _resolve_model_name(self, purpose: str) -> str:
        """용도별 모델명 반환."""
        mapping = {
            "research": self.research_model,
            "compression": self.compression_model,
            "final_report": self.final_report_model,
        }
        return mapping.get(purpose, self.default_model)
