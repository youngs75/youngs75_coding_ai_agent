"""멀티티어 모델 해석 + LiteLLM Gateway + 비용/성능 분석.

ModelTier 열거형으로 모델 능력 등급을 정의하고,
purpose → tier → TierConfig → ChatModel 흐름으로 목적별 모델을 해석한다.

LiteLLM Gateway 패턴으로 모든 프로바이더를 통합 지원한다:
  - DashScope: dashscope/qwen3-max, dashscope/qwen-turbo 등
  - OpenRouter: openrouter/qwen/qwen3-coder-plus 등
  - Anthropic: claude-sonnet-4-20250514 (접두사 불필요)

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coding_agent.core.resilience import ModelFallbackChain

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """모델 능력 등급.

    4-tier 체계 (DashScope Flagship 모델 기준):
    - REASONING: 최상위 추론 (계획/아키텍처 설계) — Qwen3-Max
    - STRONG: 코딩 특화 (코드 생성/도구 호출) — Qwen3-Coder-Next
    - DEFAULT: 범용 균형 (검증/분석) — Qwen-Plus
    - FAST: 빠른 응답 (파싱/분류) — qwen-turbo
    """

    REASONING = "reasoning"
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
    # ── OpenRouter 경유 Qwen 시리즈 ──
    "openrouter/qwen/qwen3-coder-plus": ModelCostInfo(
        model="openrouter/qwen/qwen3-coder-plus",
        input_cost_per_1m=0.65,
        output_cost_per_1m=3.25,
        latency_category="medium",
        capability_scores={"code": 0.98, "reasoning": 0.95, "speed": 0.65},
    ),
    "openrouter/qwen/qwen3-coder-next": ModelCostInfo(
        model="openrouter/qwen/qwen3-coder-next",
        input_cost_per_1m=0.12,
        output_cost_per_1m=0.75,
        latency_category="medium",
        capability_scores={"code": 0.96, "reasoning": 0.92, "speed": 0.78},
    ),
    "openrouter/qwen/qwen3-coder": ModelCostInfo(
        model="openrouter/qwen/qwen3-coder",
        input_cost_per_1m=0.22,
        output_cost_per_1m=1.00,
        latency_category="medium",
        capability_scores={"code": 0.95, "reasoning": 0.90, "speed": 0.75},
    ),
    "openrouter/qwen/qwen3.5-flash-02-23": ModelCostInfo(
        model="openrouter/qwen/qwen3.5-flash-02-23",
        input_cost_per_1m=0.07,
        output_cost_per_1m=0.26,
        latency_category="low",
        capability_scores={"code": 0.78, "reasoning": 0.80, "speed": 0.92},
    ),
    "openrouter/qwen/qwen3.5-9b": ModelCostInfo(
        model="openrouter/qwen/qwen3.5-9b",
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.15,
        latency_category="low",
        capability_scores={"code": 0.70, "reasoning": 0.70, "speed": 0.95},
    ),
    "openrouter/qwen/qwen3.5-397b-a17b": ModelCostInfo(
        model="openrouter/qwen/qwen3.5-397b-a17b",
        input_cost_per_1m=0.39,
        output_cost_per_1m=2.34,
        latency_category="medium",
        capability_scores={"code": 0.94, "reasoning": 0.96, "speed": 0.60},
    ),
    # ── DashScope (Qwen 공식 API) ──
    "dashscope/qwen-plus": ModelCostInfo(
        model="dashscope/qwen-plus",
        input_cost_per_1m=0.11,
        output_cost_per_1m=0.28,
        latency_category="low",
        capability_scores={"code": 0.95, "reasoning": 0.93, "speed": 0.85},
    ),
    "dashscope/qwen-max": ModelCostInfo(
        model="dashscope/qwen-max",
        input_cost_per_1m=1.20,
        output_cost_per_1m=6.00,
        latency_category="medium",
        capability_scores={"code": 0.95, "reasoning": 0.98, "speed": 0.60},
    ),
    "dashscope/qwen-turbo": ModelCostInfo(
        model="dashscope/qwen-turbo",
        input_cost_per_1m=0.04,
        output_cost_per_1m=0.08,
        latency_category="low",
        capability_scores={"code": 0.82, "reasoning": 0.80, "speed": 0.95},
    ),
    "dashscope/qwen-coder-plus": ModelCostInfo(
        model="dashscope/qwen-coder-plus",
        input_cost_per_1m=0.46,
        output_cost_per_1m=1.38,
        latency_category="low",
        capability_scores={"code": 0.98, "reasoning": 0.94, "speed": 0.80},
    ),
    # ── 기타 참조 모델 ──
    "openrouter/deepseek/deepseek-v3.2": ModelCostInfo(
        model="openrouter/deepseek/deepseek-v3.2",
        input_cost_per_1m=0.26,
        output_cost_per_1m=0.38,
        latency_category="low",
        capability_scores={"code": 0.90, "reasoning": 0.92, "speed": 0.85},
    ),
    "openrouter/z-ai/glm-5": ModelCostInfo(
        model="openrouter/z-ai/glm-5",
        input_cost_per_1m=0.72,
        output_cost_per_1m=2.30,
        latency_category="medium",
        capability_scores={"code": 0.88, "reasoning": 0.90, "speed": 0.70},
    ),
}


def get_model_cost_info(model: str) -> ModelCostInfo | None:
    """모델의 비용/성능 메타데이터를 반환한다."""
    return _MODEL_COST_DB.get(model)


def register_model_cost_info(info: ModelCostInfo) -> None:
    """모델 비용/성능 메타데이터를 등록한다."""
    _MODEL_COST_DB[info.model] = info


# ── 모델별 컨텍스트/출력 한도 (공식 문서 기준) ──

# 모델별 총 컨텍스트 한도 (input + output)
_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Qwen 2.5 (레거시)
    "qwen-turbo": 131_072,
    "qwen-plus": 131_072,
    "qwen-max": 262_144,
    "qwen-coder-plus": 262_144,
    # Qwen 3/3.5/3.6
    "qwen3-max": 262_144,
    "qwen3-coder-next": 1_000_000,
    "qwen3-coder-plus": 1_000_000,
    "qwen3-coder-flash": 1_000_000,
    "qwen3.5-plus": 1_000_000,
    "qwen3.5-flash": 1_000_000,
    "qwen3.6-plus": 1_000_000,
}

# 모델별 최대 출력 토큰 (Non-thinking 모드 기준)
_MODEL_MAX_OUTPUTS: dict[str, int] = {
    "qwen-turbo": 16384,
    "qwen-plus": 32768,
    "qwen-max": 65536,
    "qwen-coder-plus": 65536,
    "qwen3-max": 65536,
    "qwen3-coder-next": 65536,
    "qwen3-coder-plus": 65536,
    "qwen3-coder-flash": 65536,
    "qwen3.5-plus": 32768,
    "qwen3.5-flash": 32768,
    "qwen3.6-plus": 65536,
}


def get_model_context_limit(model: str) -> int:
    """모델의 총 컨텍스트 한도 (input + output)를 반환한다."""
    for key, limit in _MODEL_CONTEXT_LIMITS.items():
        if key in model:
            return limit
    return 262_144  # 보수적 기본값


def get_model_max_output(model: str) -> int:
    """모델의 최대 출력 토큰 수를 반환한다."""
    for key, limit in _MODEL_MAX_OUTPUTS.items():
        if key in model:
            return limit
    return 32768  # 보수적 기본값


class TierConfig(BaseModel):
    """단일 티어의 모델 설정."""

    model: str
    provider: str = "openai"
    context_window: int = 128_000   # 입력 전용 컨텍스트 윈도우 (참고용)
    temperature: float | None = None  # None → 글로벌 기본값 사용
    request_timeout: float = 120.0  # LLM 요청 타임아웃 (초)

    @property
    def total_context_limit(self) -> int:
        """모델의 총 컨텍스트 한도 (input + output)."""
        return get_model_context_limit(self.model)

    @property
    def max_output_tokens(self) -> int:
        """모델의 최대 출력 토큰 수."""
        return get_model_max_output(self.model)

    @property
    def summarization_threshold(self) -> int:
        """적응형 요약 임계치 — context_window의 75%."""
        return int(self.context_window * 0.75)

    @property
    def cost_info(self) -> ModelCostInfo | None:
        """이 티어 모델의 비용/성능 메타데이터."""
        return get_model_cost_info(self.model)


# LiteLLM 프로바이더 접두사 (모델명에 내장되어 프로바이더 자동 라우팅)
_LITELLM_PROVIDER_PREFIXES = frozenset(
    {"dashscope", "openrouter", "anthropic", "openai", "azure", "bedrock", "vertex_ai"}
)


def _to_litellm_model(model: str, provider: str) -> str:
    """모델명을 LiteLLM 포맷으로 변환한다.

    이미 LiteLLM 접두사가 있으면 그대로 반환한다.
    레거시 모델명(접두사 없음)은 프로바이더에 따라 접두사를 추가한다.

    예시:
        _to_litellm_model("qwen3-max", "dashscope")    → "dashscope/qwen3-max"
        _to_litellm_model("qwen/qwen3-max", "openrouter") → "openrouter/qwen/qwen3-max"
        _to_litellm_model("dashscope/qwen3-max", "dashscope") → "dashscope/qwen3-max" (변환 없음)
        _to_litellm_model("claude-sonnet-4-20250514", "anthropic") → "claude-sonnet-4-20250514"
    """
    # 이미 LiteLLM 접두사가 있으면 그대로
    first_segment = model.split("/")[0] if "/" in model else ""
    if first_segment in _LITELLM_PROVIDER_PREFIXES:
        return model

    # 프로바이더별 접두사 추가
    if provider == "dashscope":
        return f"dashscope/{model}"
    if provider == "openrouter":
        return f"openrouter/{model}"
    # anthropic, openai 등은 LiteLLM이 모델명으로 자동 인식
    return model


def build_default_tiers() -> dict[str, TierConfig]:
    """환경변수에서 티어별 설정을 로드하여 기본 티어 구성을 반환한다.

    LLM_PROVIDER 환경변수로 전체 프로바이더를 한 번에 전환 가능:
        LLM_PROVIDER=dashscope  → 전 티어 DashScope 사용
        LLM_PROVIDER=openrouter → 전 티어 OpenRouter 사용 (기본값)

    모델명은 LiteLLM 포맷 (dashscope/모델명, openrouter/제공자/모델명).
    티어별 오버라이드: STRONG_PROVIDER, DEFAULT_PROVIDER, FAST_PROVIDER
    """
    global_provider = os.getenv("LLM_PROVIDER", "openrouter")

    # 프로바이더별 기본 모델명 (LiteLLM 포맷)
    if global_provider == "dashscope":
        default_reasoning = "dashscope/qwen3-max"
        default_strong = "dashscope/qwen3-coder-next"
        default_default = "dashscope/qwen3.5-plus"
        default_fast = "dashscope/qwen3.5-flash"
    elif global_provider == "anthropic":
        default_reasoning = "claude-sonnet-4-20250514"
        default_strong = "claude-sonnet-4-20250514"
        default_default = "claude-haiku-4-20250414"
        default_fast = "claude-haiku-4-20250414"
    else:
        default_reasoning = "openrouter/qwen/qwen3-max"
        default_strong = "openrouter/qwen/qwen3-coder-plus"
        default_default = "openrouter/qwen/qwen3-coder-next"
        default_fast = "openrouter/qwen/qwen3.5-flash-02-23"

    return {
        ModelTier.REASONING: TierConfig(
            model=os.getenv("REASONING_MODEL", default_reasoning),
            provider=os.getenv("REASONING_PROVIDER", global_provider),
            context_window=int(os.getenv("REASONING_CONTEXT_WINDOW", "1000000")),
            request_timeout=float(os.getenv("REASONING_TIMEOUT", "180")),
        ),
        ModelTier.STRONG: TierConfig(
            model=os.getenv("STRONG_MODEL", default_strong),
            provider=os.getenv("STRONG_PROVIDER", global_provider),
            context_window=int(os.getenv("STRONG_CONTEXT_WINDOW", "1000000")),
            request_timeout=float(os.getenv("STRONG_TIMEOUT", "180")),
        ),
        ModelTier.DEFAULT: TierConfig(
            model=os.getenv("DEFAULT_MODEL", default_default),
            provider=os.getenv("DEFAULT_PROVIDER", global_provider),
            context_window=int(os.getenv("DEFAULT_CONTEXT_WINDOW", "262144")),
            request_timeout=float(os.getenv("DEFAULT_TIMEOUT", "120")),
        ),
        ModelTier.FAST: TierConfig(
            model=os.getenv("FAST_MODEL", default_fast),
            provider=os.getenv("FAST_PROVIDER", global_provider),
            context_window=int(os.getenv("FAST_CONTEXT_WINDOW", "1000000")),
            request_timeout=float(os.getenv("FAST_TIMEOUT", "60")),
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
        "planning": ModelTier.REASONING,
        "generation": ModelTier.STRONG,
        "tool_planning": ModelTier.FAST,
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
        return TierConfig(model="openrouter/qwen/qwen3-coder-next", provider="openrouter")
    return config


def create_chat_model(
    tier_config: TierConfig,
    *,
    temperature: float = 0.1,
    structured: type | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """TierConfig를 기반으로 LangChain ChatModel을 생성한다.

    두 가지 모드를 자동 전환:
    - 프록시 모드 (LITELLM_PROXY_URL 설정 시): ChatOpenAI → LiteLLM Proxy → 프로바이더
    - SDK 모드 (기본): ChatLiteLLM SDK로 직접 프로바이더 호출

    모델명은 LiteLLM 포맷 (dashscope/모델명, openrouter/제공자/모델명).
    """
    effective_temp = (
        tier_config.temperature if tier_config.temperature is not None else temperature
    )
    provider = tier_config.provider
    model = tier_config.model
    timeout = tier_config.request_timeout

    # LiteLLM 포맷으로 변환 (레거시 모델명 호환)
    litellm_model = _to_litellm_model(model, provider)

    # 호출자가 max_tokens를 지정하지 않으면 모델별 최대값 설정
    # 공식 문서 기준: https://help.aliyun.com/zh/model-studio/
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = get_model_max_output(litellm_model)

    # Thinking 모드 설정: 호출자가 명시하지 않으면 모델 유형별 기본값 적용
    # - REASONING(planner): thinking ON → CoT로 더 깊은 추론
    # - STRONG(coder): thinking OFF → max output 전체를 코드 생성에 사용
    # - DEFAULT/FAST: thinking OFF → 출력 토큰 절약
    # Qwen3 시리즈는 기본 CoT가 활성화되어 있어 명시적으로 설정해야 한다.
    # Thinking 모드: 전 모델 비활성화
    # qwen3-max도 thinking 없이도 충분히 좋은 품질을 제공하며,
    # thinking ON 시 Planner가 5분+ 소요되는 UX 문제가 있음
    extra_body = kwargs.get("extra_body", {})
    if "enable_thinking" not in extra_body:
        extra_body["enable_thinking"] = False
    kwargs["extra_body"] = extra_body

    # extra_body를 kwargs에서 분리 (ChatOpenAI는 명시적 파라미터로 받아야 함)
    extra_body_param = kwargs.pop("extra_body", {})

    proxy_url = os.getenv("LITELLM_PROXY_URL")

    if proxy_url:
        # 프록시 모드: ChatOpenAI로 LiteLLM Proxy 경유
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=litellm_model,
            temperature=effective_temp,
            openai_api_key=os.getenv("LITELLM_MASTER_KEY", "sk-harness-local-dev"),
            openai_api_base=proxy_url,
            request_timeout=timeout,
            extra_body=extra_body_param,
            **kwargs,
        )
        logger.debug(
            "LiteLLM Proxy 모델 생성: %s → %s (timeout=%.0fs, extra_body=%s)",
            litellm_model, proxy_url, timeout, extra_body_param,
        )
    else:
        # SDK 모드: ChatLiteLLM으로 직접 프로바이더 호출
        from langchain_litellm import ChatLiteLLM

        # ChatLiteLLM은 model_kwargs로 extra_body 전달
        model_kwargs = kwargs.pop("model_kwargs", {})
        model_kwargs["extra_body"] = extra_body_param

        # DashScope: LiteLLM이 DASHSCOPE_BASE_URL 환경변수를 무시하므로
        # api_base를 명시적으로 전달 (워크스페이스 URL 지원)
        api_base = None
        if provider == "dashscope" and os.getenv("DASHSCOPE_BASE_URL"):
            api_base = os.getenv("DASHSCOPE_BASE_URL")

        llm = ChatLiteLLM(
            model=litellm_model,
            model_name=litellm_model,
            temperature=effective_temp,
            request_timeout=timeout,
            model_kwargs=model_kwargs,
            api_base=api_base,
            **kwargs,
        )
        logger.debug(
            "LiteLLM SDK 모델 생성: %s (provider=%s, timeout=%.0fs)",
            litellm_model, provider, timeout,
        )

    if structured:
        llm = llm.with_structured_output(structured, include_raw=True)

    return llm


# ── 목적별 최적 모델 자동 선택 ──

# 목적별 가중치: 어떤 capability를 중시하는지 정의
PURPOSE_CAPABILITY_WEIGHTS: dict[str, dict[str, float]] = {
    "planning": {"reasoning": 0.8, "code": 0.1, "speed": 0.1},
    "generation": {"code": 0.6, "reasoning": 0.3, "speed": 0.1},
    "verification": {"reasoning": 0.6, "code": 0.3, "speed": 0.1},
    "tool_planning": {"speed": 0.5, "reasoning": 0.3, "code": 0.2},
    "parsing": {"speed": 0.6, "reasoning": 0.3, "code": 0.1},
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
        default_config = tiers.get(
            ModelTier.DEFAULT,
            TierConfig(model="qwen/qwen3-coder-next", provider="openrouter"),
        )
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


# ── 모델 Fallback 체인 빌더 ──


def build_fallback_chain(
    tiers: dict[str, TierConfig],
    *,
    primary_tier: str = ModelTier.STRONG,
    temperature: float = 0.1,
    timeout_per_model: float = 60.0,
) -> "ModelFallbackChain":
    """티어 설정에서 ModelFallbackChain을 구성한다.

    기본 순서: primary_tier -> DEFAULT -> FAST

    Args:
        tiers: 티어별 TierConfig 딕셔너리
        primary_tier: 최우선 시도할 티어 (기본: STRONG)
        temperature: 모델 생성 시 temperature (기본: 0.1)
        timeout_per_model: 모델별 타임아웃 초 (기본: 60.0)

    Returns:
        ModelFallbackChain 인스턴스
    """
    from coding_agent.core.resilience import ModelFallbackChain

    # fallback 순서 결정
    tier_order = [ModelTier.STRONG, ModelTier.DEFAULT, ModelTier.FAST]
    # primary_tier를 맨 앞으로
    if primary_tier in tier_order:
        tier_order.remove(primary_tier)
        tier_order.insert(0, primary_tier)

    models: list[BaseChatModel] = []
    names: list[str] = []
    for tier_name in tier_order:
        tc = tiers.get(tier_name)
        if tc:
            model = create_chat_model(tc, temperature=temperature)
            models.append(model)
            names.append(f"{tier_name}({tc.model})")

    return ModelFallbackChain(
        models=models,
        model_names=names,
        timeout_per_model=timeout_per_model,
    )
