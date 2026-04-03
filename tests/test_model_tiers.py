"""멀티티어 모델 해석 유닛 테스트.

LLM 호출 없이 티어 해석 로직, 폴백, 환경변수 오버라이드를 검증한다.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from youngs75_a2a.core.model_tiers import (
    ModelTier,
    TierConfig,
    build_default_purpose_tiers,
    build_default_tiers,
    resolve_tier_config,
)


# ── TierConfig ──


class TestTierConfig:
    def test_defaults(self):
        tc = TierConfig(model="gpt-5.4")
        assert tc.provider == "openai"
        assert tc.context_window == 128_000
        assert tc.temperature is None

    def test_summarization_threshold(self):
        tc = TierConfig(model="m", context_window=100_000)
        assert tc.summarization_threshold == 75_000

    def test_custom_temperature(self):
        tc = TierConfig(model="m", temperature=0.7)
        assert tc.temperature == 0.7


# ── build_default_tiers ──


class TestBuildDefaultTiers:
    def test_returns_three_tiers(self):
        tiers = build_default_tiers()
        assert set(tiers.keys()) == {ModelTier.STRONG, ModelTier.DEFAULT, ModelTier.FAST}

    def test_default_models(self):
        tiers = build_default_tiers()
        assert tiers[ModelTier.STRONG].model == "gpt-5.4"
        assert tiers[ModelTier.FAST].model == "gpt-4.1-mini"

    def test_env_override(self):
        env = {
            "STRONG_MODEL": "o3",
            "STRONG_PROVIDER": "openrouter",
            "STRONG_CONTEXT_WINDOW": "200000",
        }
        with patch.dict(os.environ, env, clear=False):
            tiers = build_default_tiers()
            strong = tiers[ModelTier.STRONG]
            assert strong.model == "o3"
            assert strong.provider == "openrouter"
            assert strong.context_window == 200_000


# ── build_default_purpose_tiers ──


class TestBuildDefaultPurposeTiers:
    def test_default_mapping(self):
        pt = build_default_purpose_tiers()
        assert pt["generation"] == ModelTier.STRONG
        assert pt["verification"] == ModelTier.DEFAULT
        assert pt["parsing"] == ModelTier.FAST
        assert pt["default"] == ModelTier.DEFAULT

    def test_env_json_override(self):
        custom = '{"generation":"fast","default":"fast"}'
        with patch.dict(os.environ, {"PURPOSE_TIERS": custom}, clear=False):
            pt = build_default_purpose_tiers()
            assert pt["generation"] == "fast"
            assert pt["default"] == "fast"
            assert "verification" not in pt


# ── resolve_tier_config ──


class TestResolveTierConfig:
    @pytest.fixture()
    def tiers(self):
        return {
            ModelTier.STRONG: TierConfig(model="strong-model"),
            ModelTier.DEFAULT: TierConfig(model="default-model"),
            ModelTier.FAST: TierConfig(model="fast-model"),
        }

    @pytest.fixture()
    def purpose_tiers(self):
        return {
            "generation": ModelTier.STRONG,
            "verification": ModelTier.DEFAULT,
            "parsing": ModelTier.FAST,
            "default": ModelTier.DEFAULT,
        }

    def test_known_purpose(self, tiers, purpose_tiers):
        cfg = resolve_tier_config("generation", tiers, purpose_tiers)
        assert cfg.model == "strong-model"

    def test_unknown_purpose_falls_to_default(self, tiers, purpose_tiers):
        cfg = resolve_tier_config("unknown_purpose", tiers, purpose_tiers)
        assert cfg.model == "default-model"

    def test_missing_tier_falls_to_default(self, tiers):
        pt = {"generation": "nonexistent", "default": ModelTier.DEFAULT}
        cfg = resolve_tier_config("generation", tiers, pt)
        assert cfg.model == "default-model"

    def test_all_missing_returns_hardcoded_fallback(self):
        cfg = resolve_tier_config("x", {}, {})
        assert cfg.model == "gpt-5.4"
        assert cfg.provider == "openai"


# ── BaseAgentConfig 티어 연동 ──


class TestBaseAgentConfigTiers:
    def test_get_tier_config(self):
        from youngs75_a2a.core.config import BaseAgentConfig

        config = BaseAgentConfig()
        tc = config.get_tier_config("generation")
        assert tc.model == build_default_tiers()[ModelTier.STRONG].model

    def test_legacy_get_model_still_uses_resolve_model_name(self):
        """BaseAgentConfig.get_model()은 레거시 경로(_resolve_model_name)를 유지한다."""
        from youngs75_a2a.core.config import BaseAgentConfig

        config = BaseAgentConfig(default_model="test-model", model_provider="openai")
        assert config._resolve_model_name("any") == "test-model"

    def test_legacy_fields_preserved(self):
        from youngs75_a2a.core.config import BaseAgentConfig

        config = BaseAgentConfig(
            model_provider="openai",
            default_model="gpt-5.4",
            temperature=0.1,
            mcp_servers={"tavily": "http://localhost:3001/mcp/"},
        )
        assert config.model_provider == "openai"
        assert config.default_model == "gpt-5.4"
        assert config.get_mcp_endpoint("tavily") == "http://localhost:3001/mcp/"

    def test_to_langgraph_configurable_includes_tiers(self):
        from youngs75_a2a.core.config import BaseAgentConfig

        config = BaseAgentConfig()
        configurable = config.to_langgraph_configurable()
        assert "model_tiers" in configurable
        assert "purpose_tiers" in configurable
        assert configurable["default_model"] == config.default_model


# ── CodingConfig 티어 연동 ──


class TestCodingConfigTiers:
    def test_purpose_tiers_default(self):
        from youngs75_a2a.agents.coding_assistant.config import CodingConfig

        config = CodingConfig()
        assert config.purpose_tiers["generation"] == ModelTier.STRONG
        assert config.purpose_tiers["verification"] == ModelTier.DEFAULT

    def test_tier_config_for_generation(self):
        from youngs75_a2a.agents.coding_assistant.config import CodingConfig

        config = CodingConfig()
        tc = config.get_tier_config("generation")
        assert tc.model == build_default_tiers()[ModelTier.STRONG].model

    def test_explicit_override_takes_precedence(self):
        """CODING_GEN_MODEL 환경변수가 티어보다 우선한다."""
        from youngs75_a2a.agents.coding_assistant.config import CodingConfig

        config = CodingConfig(generation_model="custom-gen-model")
        assert config._get_explicit_override("generation") == "custom-gen-model"
        assert config._get_explicit_override("verification") is None
        assert config._get_explicit_override("default") is None

    def test_no_override_when_none(self):
        from youngs75_a2a.agents.coding_assistant.config import CodingConfig

        config = CodingConfig(generation_model=None, verification_model=None)
        assert config._get_explicit_override("generation") is None
        assert config._get_explicit_override("verification") is None


# ── ResearchConfig 하위 호환 ──


class TestResearchConfigBackwardCompat:
    def test_resolve_model_name_still_works(self):
        from youngs75_a2a.agents.deep_research.config import ResearchConfig

        rc = ResearchConfig(
            research_model="research-llm",
            compression_model="compress-llm",
            final_report_model="report-llm",
        )
        assert rc._resolve_model_name("research") == "research-llm"
        assert rc._resolve_model_name("compression") == "compress-llm"
        assert rc._resolve_model_name("final_report") == "report-llm"
        assert rc._resolve_model_name("unknown") == rc.default_model


# ── OpenRouter TierConfig ──


class TestOpenRouterConfig:
    def test_openrouter_tier_config(self):
        tc = TierConfig(
            model="meta-llama/llama-4-scout",
            provider="openrouter",
            context_window=128_000,
        )
        assert tc.provider == "openrouter"
        assert tc.model == "meta-llama/llama-4-scout"

    def test_openrouter_via_env(self):
        env = {
            "FAST_MODEL": "meta-llama/llama-4-scout",
            "FAST_PROVIDER": "openrouter",
        }
        with patch.dict(os.environ, env, clear=False):
            tiers = build_default_tiers()
            fast = tiers[ModelTier.FAST]
            assert fast.provider == "openrouter"
            assert fast.model == "meta-llama/llama-4-scout"
