"""멀티티어 모델 해석 + OpenRouter 지원 + 비용/성능 분석.

ModelTier 열거형으로 모델 능력 등급을 정의하고,
purpose → tier → TierConfig → ChatModel 흐름으로 목적별 모델을 해석한다.

OpenRouter 프로바이더는 OpenAI 호환 API를 통해 오픈소스 모델을 지원한다.

목적별 최적 모델 자동 선택:
  - parse: 빠르고 저렴한 모델 (FAST 티어)
  - execute: 코드 생성 능력 우선 (STRONG 티어)
  - verify: 정확성 우선 (DEFAULT 티어)

비용/성능 트레이드오프 분석 유틸리티 포함.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """모델 능력 등급."""

    STRONG = "strong"
    DEFAULT = "default"
    FAST = "fast"


# ── 모델별 비용/성능 메타데이터 ──


class ModelCostInfo(BaseModel):
    """모델의 비용 및 성능 메타데이터.

    비용 단위: USD per 1M tokens.
    latency_category: "low" | "medium" | "high"
    capability_scores: 코드 생성, 추론, 속도 등 0.0~1.0 점수
    """

    model: str
    input_cost_per_1m: float = 0.0
    output_cost_per_1m: float = 0.0
    latency_category: str = "medium"  # low, medium, high
    capability_scores: dict[str, float] = {}

    @property
    def avg_cost_per_1m(self) -> float:
        """입출력 평균 비용 (USD per 1M tokens)."""
        return (self.input_cost_per_1m + self.output_cost_per_1m) / 2


# 알려진 모델의 비용/성능 데이터 (참고용, 환경변수로 오버라이드 가능)
_MODEL_COST_DB: dict[str, ModelCostInfo] = {
    "gpt-5.4": ModelCostInfo(
        model="gpt-5.4",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        latency_category="medium",
        capability_scores={"code": 0.95, "reasoning": 0.95, "speed": 0.7},
    ),
    "gpt-5.4-mini": ModelCostInfo(
        model="gpt-5.4-mini",
        input_cost_per_1m=0.40,
        output_cost_per_1m=1.60,
        latency_category="low",
        capability_scores={"code": 0.80, "reasoning": 0.80, "speed": 0.95},
    ),
    "gpt-4.1-mini": ModelCostInfo(
        model="gpt-4.1-mini",
        input_cost_per_1m=0.40,
        output_cost_per_1m=1.60,
        latency_category="low",
        capability_scores={"code": 0.75, "reasoning": 0.75, "speed": 0.95},
    ),
    "gpt-4.1": ModelCostInfo(
        model="gpt-4.1",
        input_cost_per_1m=2.00,
        output_cost_per_1m=8.00,
        latency_category="medium",
        capability_scores={"code": 0.90, "reasoning": 0.90, "speed": 0.75},
    ),
}


def get_model_cost_info(model: str) -> ModelCostInfo | None:
    """모델의 비용/성능 메타데이터를 반환한다."""
    return _MODEL_COST_DB.get(model)


def register_model_cost_info(info: ModelCostInfo) -> None:
    """모델 비용/성능 메타데이터를 등록한다."""
    _MODEL_COST_DB[info.model] = info


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

    @property
    def cost_info(self) -> ModelCostInfo | None:
        """이 티어 모델의 비용/성능 메타데이터."""
        return get_model_cost_info(self.model)


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
    tier_name = purpose_tiers.get(
        purpose, purpose_tiers.get("default", ModelTier.DEFAULT)
    )
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


# ── 목적별 최적 모델 자동 선택 ──

# 목적별 가중치: 어떤 capability를 중시하는지 정의
PURPOSE_CAPABILITY_WEIGHTS: dict[str, dict[str, float]] = {
    "parsing": {"speed": 0.6, "reasoning": 0.3, "code": 0.1},
    "generation": {"code": 0.6, "reasoning": 0.3, "speed": 0.1},
    "verification": {"reasoning": 0.6, "code": 0.3, "speed": 0.1},
    "default": {"code": 0.4, "reasoning": 0.4, "speed": 0.2},
}


def recommend_tier_for_purpose(
    purpose: str,
    tiers: dict[str, TierConfig],
) -> tuple[str, TierConfig, dict[str, Any]]:
    """목적에 가장 적합한 티어를 추천한다.

    각 티어의 모델 capability 점수와 목적별 가중치를 곱하여
    가중 합산 점수가 가장 높은 티어를 선택한다.

    Args:
        purpose: 목적 (parsing, generation, verification, default)
        tiers: 사용 가능한 티어 설정

    Returns:
        (추천 티어 이름, TierConfig, 분석 상세 정보)
    """
    weights = PURPOSE_CAPABILITY_WEIGHTS.get(
        purpose,
        PURPOSE_CAPABILITY_WEIGHTS["default"],
    )

    best_tier_name = None
    best_config = None
    best_score = -1.0
    analysis: dict[str, Any] = {"purpose": purpose, "weights": weights, "scores": {}}

    for tier_name, tier_config in tiers.items():
        cost_info = tier_config.cost_info
        if cost_info is None:
            # 비용 정보 없는 모델은 기본 점수 0.5로 처리
            score = 0.5
        else:
            score = sum(
                cost_info.capability_scores.get(cap, 0.5) * w
                for cap, w in weights.items()
            )

        analysis["scores"][tier_name] = round(score, 4)

        if score > best_score:
            best_score = score
            best_tier_name = tier_name
            best_config = tier_config

    analysis["recommended_tier"] = best_tier_name
    analysis["recommended_model"] = best_config.model if best_config else None

    # 폴백
    if best_config is None:
        default_config = tiers.get(ModelTier.DEFAULT, TierConfig(model="gpt-5.4"))
        return ModelTier.DEFAULT, default_config, analysis

    return best_tier_name, best_config, analysis


# ── 비용/성능 트레이드오프 분석 ──


def estimate_cost(
    tier_config: TierConfig,
    *,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, float] | None:
    """주어진 티어로 특정 토큰 수를 처리할 때의 예상 비용을 계산한다.

    Args:
        tier_config: 사용할 티어 설정
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수

    Returns:
        {"input_cost": ..., "output_cost": ..., "total_cost": ...} (USD)
        비용 정보가 없으면 None
    """
    cost_info = tier_config.cost_info
    if cost_info is None:
        return None

    input_cost = (input_tokens / 1_000_000) * cost_info.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * cost_info.output_cost_per_1m

    return {
        "model": tier_config.model,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def analyze_tier_tradeoffs(
    tiers: dict[str, TierConfig],
    *,
    input_tokens: int = 1000,
    output_tokens: int = 500,
) -> list[dict[str, Any]]:
    """모든 티어의 비용/성능 트레이드오프를 분석한다.

    Args:
        tiers: 사용 가능한 티어 설정
        input_tokens: 비교 기준 입력 토큰 수
        output_tokens: 비교 기준 출력 토큰 수

    Returns:
        티어별 분석 결과 리스트 (비용 오름차순 정렬)
    """
    results: list[dict[str, Any]] = []

    for tier_name, tier_config in tiers.items():
        cost_info = tier_config.cost_info
        cost_est = estimate_cost(
            tier_config, input_tokens=input_tokens, output_tokens=output_tokens
        )

        entry: dict[str, Any] = {
            "tier": tier_name,
            "model": tier_config.model,
            "provider": tier_config.provider,
            "context_window": tier_config.context_window,
        }

        if cost_info is not None:
            entry["latency"] = cost_info.latency_category
            entry["capability_scores"] = cost_info.capability_scores
        else:
            entry["latency"] = "unknown"
            entry["capability_scores"] = {}

        if cost_est is not None:
            entry["estimated_cost"] = cost_est
        else:
            entry["estimated_cost"] = None

        # 목적별 적합도 점수
        entry["purpose_fit"] = {}
        for purpose in PURPOSE_CAPABILITY_WEIGHTS:
            weights = PURPOSE_CAPABILITY_WEIGHTS[purpose]
            if cost_info is not None:
                score = sum(
                    cost_info.capability_scores.get(cap, 0.5) * w
                    for cap, w in weights.items()
                )
            else:
                score = 0.5
            entry["purpose_fit"][purpose] = round(score, 4)

        results.append(entry)

    # 비용 오름차순 정렬 (비용 정보 없는 항목은 뒤로)
    results.sort(
        key=lambda r: (
            r["estimated_cost"]["total_cost_usd"]
            if r["estimated_cost"]
            else float("inf")
        )
    )

    return results
