"""에이전트 설정 기본 클래스.

모든 에이전트 Config는 이 클래스를 상속하여 일관된 설정 관리를 제공한다.
환경변수 오버라이드, LangGraph RunnableConfig 변환을 지원한다.
"""

from __future__ import annotations

import os
from typing import Any
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig


class BaseAgentConfig(BaseModel):
    """모든 에이전트 Config의 기본 클래스."""

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

    model_config = {"extra": "allow"}

    def get_model(
        self,
        purpose: str = "default",
        *,
        structured: type | None = None,
    ) -> BaseChatModel:
        """목적별 LLM 모델을 반환한다.

        Args:
            purpose: 모델 용도 (서브클래스에서 오버라이드하여 용도별 모델 분기)
            structured: with_structured_output에 전달할 스키마 클래스
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
