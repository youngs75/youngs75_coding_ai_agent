"""Orchestrator 에이전트 설정."""

from __future__ import annotations

from pydantic import Field
from youngs75_a2a.core.config import BaseAgentConfig


class AgentEndpoint(BaseAgentConfig):
    """하위 에이전트 엔드포인트 정보."""

    model_config = {"extra": "allow"}

    name: str
    url: str
    description: str


class OrchestratorConfig(BaseAgentConfig):
    """Orchestrator 설정.

    agent_endpoints: 라우팅 대상 에이전트 목록
    """

    agent_endpoints: list[AgentEndpoint] = Field(default_factory=list)

    def get_agent_descriptions(self) -> str:
        """LLM 프롬프트용 에이전트 목록 문자열 생성."""
        lines = []
        for ep in self.agent_endpoints:
            lines.append(f"- {ep.name}: {ep.description}")
        return "\n".join(lines)

    def get_endpoint_url(self, agent_name: str) -> str | None:
        """에이전트 이름으로 URL 조회."""
        for ep in self.agent_endpoints:
            if ep.name == agent_name:
                return ep.url
        return None
