"""멀티티어 모델 해석 + OpenRouter 지원.

ModelTier 열거형으로 모델 능력 등급을 정의하고,
purpose → tier → TierConfig → ChatModel 흐름으로 목적별 모델을 해석한다.

OpenRouter 프로바이더는 OpenAI 호환 API를 통해 오픈소스 모델을 지원한다.
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class ModelTier(str, Enum):
    """모델 능력 등급."""

    STRONG = "strong"
    DEFAULT = "default"
    FAST = "fast"


class TierConfig(BaseModel):
    """단일 티어의 모델 설정."""

    model: str
    provider: str = "openai"
    context_window: int = 128_000
    temperature: float | None = None  # None → 글로벌 기본값 사용

    @property
    def summarization_threshold(self) -> int:
        """적응형 요약 임계치 — context_window의 75%."""
        return int(self.context_window * 0.75)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_default_tiers() -> dict[str, TierConfig]:
    """환경변수에서 티어별 설정을 로드하여 기본 티어 구성을 반환한다."""
    return {
        ModelTier.STRONG: TierConfig(
            model=os.getenv("STRONG_MODEL", "gpt-5.4"),
            provider=os.getenv("STRONG_PROVIDER", "openai"),
            context_window=int(os.getenv("STRONG_CONTEXT_WINDOW", "128000")),
        ),
        ModelTier.DEFAULT: TierConfig(
            model=os.getenv("DEFAULT_MODEL", "gpt-5.4"),
            provider=os.getenv("DEFAULT_PROVIDER", "openai"),
            context_window=int(os.getenv("DEFAULT_CONTEXT_WINDOW", "128000")),
        ),
        ModelTier.FAST: TierConfig(
            model=os.getenv("FAST_MODEL", "gpt-4.1-mini"),
            provider=os.getenv("FAST_PROVIDER", "openai"),
            context_window=int(os.getenv("FAST_CONTEXT_WINDOW", "128000")),
        ),
    }


def build_default_purpose_tiers() -> dict[str, str]:
    """환경변수에서 purpose→tier 매핑을 로드한다.

    PURPOSE_TIERS 환경변수로 JSON 오버라이드 가능:
        PURPOSE_TIERS='{"generation":"strong","verification":"fast"}'
    """
    env_val = os.getenv("PURPOSE_TIERS")
    if env_val:
        return json.loads(env_val)
    return {
        "generation": ModelTier.STRONG,
        "verification": ModelTier.DEFAULT,
        "parsing": ModelTier.FAST,
        "default": ModelTier.DEFAULT,
    }


def resolve_tier_config(
    purpose: str,
    tiers: dict[str, TierConfig],
    purpose_tiers: dict[str, str],
) -> TierConfig:
    """purpose를 TierConfig로 해석한다.

    해석 순서: purpose_tiers[purpose] → purpose_tiers["default"] → ModelTier.DEFAULT
    """
    tier_name = purpose_tiers.get(purpose, purpose_tiers.get("default", ModelTier.DEFAULT))
    config = tiers.get(tier_name)
    if config is None:
        config = tiers.get(ModelTier.DEFAULT)
    if config is None:
        return TierConfig(model="gpt-5.4", provider="openai")
    return config


def create_chat_model(
    tier_config: TierConfig,
    *,
    temperature: float = 0.1,
    structured: type | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """TierConfig를 기반으로 LangChain ChatModel을 생성한다.

    OpenRouter 프로바이더는 OpenAI 호환 API(openai_api_base)를 사용한다.
    """
    effective_temp = (
        tier_config.temperature if tier_config.temperature is not None else temperature
    )
    provider = tier_config.provider
    model = tier_config.model

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=effective_temp,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            openai_api_base=OPENROUTER_BASE_URL,
            **kwargs,
        )
    else:
        llm = init_chat_model(
            model=model,
            model_provider=provider,
            temperature=effective_temp,
            **kwargs,
        )

    if structured:
        llm = llm.with_structured_output(structured, include_raw=True)

    return llm
