"""에이전트 설정 기본 클래스.

모든 에이전트 Config는 이 클래스를 상속하여 일관된 설정 관리를 제공한다.
환경변수 오버라이드, 멀티티어 모델 해석, LangGraph RunnableConfig 변환을 지원한다.

모델 해석 경로:
- 레거시: _resolve_model_name(purpose) + model_provider → init_chat_model()
- 티어:  purpose_tiers[purpose] → model_tiers[tier] → create_chat_model()

기본 get_model()은 레거시 경로를 유지한다 (ResearchConfig 등 하위 호환).
티어 기반 해석은 서브클래스에서 get_model()을 오버라이드하여 사용한다.
"""

from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from youngs75_a2a.core.model_tiers import (
    TierConfig,
    build_default_purpose_tiers,
    build_default_tiers,
    create_chat_model,
    resolve_tier_config,
)


class BaseAgentConfig(BaseModel):
    """모든 에이전트 Config의 기본 클래스."""

    # 레거시 필드 (하위 호환 — ResearchConfig 등에서 사용)
    model_provider: str = Field(
        default_factory=lambda: os.getenv("MODEL_PROVIDER", "openai"),
    )
    default_model: str = Field(
        default_factory=lambda: os.getenv("MODEL_NAME", "gpt-5.4"),
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")),
    )
    max_retries: int = Field(default=3)
    mcp_servers: dict[str, str] = Field(default_factory=dict)

    # 멀티티어 모델 설정
    model_tiers: dict[str, TierConfig] = Field(default_factory=build_default_tiers)
    purpose_tiers: dict[str, str] = Field(default_factory=build_default_purpose_tiers)

    model_config = {"extra": "allow"}

    def get_model(
        self,
        purpose: str = "default",
        *,
        structured: type | None = None,
    ) -> BaseChatModel:
        """목적별 LLM 모델을 반환한다.

        기본 구현은 _resolve_model_name + model_provider 레거시 경로를 사용한다.
        티어 시스템을 사용하려면 서브클래스에서 get_model()을 오버라이드하라.
        """
        model_name = self._resolve_model_name(purpose)
        llm = init_chat_model(
            model=model_name,
            model_provider=self.model_provider,
            temperature=self.temperature,
        )
        if structured:
            llm = llm.with_structured_output(structured, include_raw=True)
        return llm

    def _resolve_model_name(self, purpose: str) -> str:
        """목적별 모델명 결정. 서브클래스에서 오버라이드."""
        return self.default_model

    def get_tier_config(self, purpose: str = "default") -> TierConfig:
        """목적에 해당하는 TierConfig를 반환한다."""
        return resolve_tier_config(purpose, self.model_tiers, self.purpose_tiers)

    def get_mcp_endpoint(self, server_name: str) -> str | None:
        """MCP 서버 엔드포인트를 반환한다."""
        return self.mcp_servers.get(server_name)

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "BaseAgentConfig":
        """LangGraph RunnableConfig에서 설정을 추출한다."""
        configurable = (config or {}).get("configurable", {})
        return cls(**{
            k: v for k, v in configurable.items()
            if k in cls.model_fields
        })

    def to_langgraph_configurable(self) -> dict[str, Any]:
        """LangGraph configurable dict로 변환한다."""
        return self.model_dump(exclude_none=True)
