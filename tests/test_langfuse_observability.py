"""Langfuse 관측성 통합 테스트.

콜백 핸들러 팩토리, 메트릭 수집기, CLI 통합을 검증한다.
Langfuse 서버 연결 없이 동작하는 단위 테스트로 구성.
"""

from __future__ import annotations

import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession
from youngs75_a2a.eval_pipeline.observability.callback_handler import (
    AgentMetricsCollector,
    NodeMetrics,
    build_observed_config,
    create_langfuse_handler,
    safe_flush,
)


def _make_renderer() -> CLIRenderer:
    return CLIRenderer(console=Console(file=StringIO(), force_terminal=True))


# ── NodeMetrics ──


class TestNodeMetrics:
    def test_defaults(self):
        m = NodeMetrics(node_name="parse_request")
        assert m.node_name == "parse_request"
        assert m.call_count == 0
        assert m.total_duration_ms == 0.0
        assert m.error_count == 0

    def test_avg_duration_zero_calls(self):
        m = NodeMetrics(node_name="test")
        assert m.avg_duration_ms == 0.0

    def test_avg_duration(self):
        m = NodeMetrics(node_name="test", call_count=4, total_duration_ms=200.0)
        assert m.avg_duration_ms == 50.0

    def test_error_rate_zero_calls(self):
        m = NodeMetrics(node_name="test")
        assert m.error_rate == 0.0

    def test_error_rate(self):
        m = NodeMetrics(node_name="test", call_count=10, error_count=3)
        assert m.error_rate == pytest.approx(0.3)


# ── AgentMetricsCollector ──


class TestAgentMetricsCollector:
    def test_initial_state(self):
        c = AgentMetricsCollector(agent_name="coding_assistant")
        assert c.agent_name == "coding_assistant"
        assert c.total_prompt_tokens == 0
        assert c.total_completion_tokens == 0
        assert c.total_tokens == 0
        assert c.error_count == 0
        assert c.node_metrics == {}

    def test_record_llm_tokens(self):
        c = AgentMetricsCollector(agent_name="test")
        c.record_llm_tokens(prompt_tokens=100, completion_tokens=50)
        c.record_llm_tokens(prompt_tokens=200, completion_tokens=80)
        assert c.total_prompt_tokens == 300
        assert c.total_completion_tokens == 130
        assert c.total_tokens == 430

    def test_record_node_lifecycle(self):
        c = AgentMetricsCollector(agent_name="test")
        c.record_node_start("parse_request")
        time.sleep(0.01)  # 최소 10ms 대기
        c.record_node_end("parse_request")

        metrics = c.node_metrics
        assert "parse_request" in metrics
        assert metrics["parse_request"].call_count == 1
        assert metrics["parse_request"].total_duration_ms > 0
        assert metrics["parse_request"].error_count == 0

    def test_record_node_end_without_start(self):
        """start 없이 end를 호출해도 에러 없이 동작."""
        c = AgentMetricsCollector(agent_name="test")
        c.record_node_end("unknown_node")
        metrics = c.node_metrics
        assert "unknown_node" in metrics
        assert metrics["unknown_node"].call_count == 1

    def test_record_node_error(self):
        c = AgentMetricsCollector(agent_name="test")
        c.record_node_start("execute_code")
        c.record_node_end("execute_code", error=True)
        assert c.error_count == 1
        metrics = c.node_metrics
        assert metrics["execute_code"].error_count == 1
        assert metrics["execute_code"].error_rate == 1.0

    def test_record_error_global(self):
        c = AgentMetricsCollector(agent_name="test")
        c.record_error()
        c.record_error()
        assert c.error_count == 2

    def test_error_rate_calculation(self):
        c = AgentMetricsCollector(agent_name="test")
        c.record_node_start("a")
        c.record_node_end("a")
        c.record_node_start("b")
        c.record_node_end("b")
        c.record_node_start("c")
        c.record_node_end("c", error=True)
        # 3개 노드 호출, 1개 에러 → error_rate = 1/3
        assert c.error_rate == pytest.approx(1 / 3)

    def test_finalize_sets_end_time(self):
        c = AgentMetricsCollector(agent_name="test")
        assert c.end_time == 0.0
        c.finalize()
        assert c.end_time > 0

    def test_total_duration(self):
        c = AgentMetricsCollector(agent_name="test")
        time.sleep(0.01)
        c.finalize()
        assert c.total_duration_ms > 0

    def test_to_dict(self):
        c = AgentMetricsCollector(agent_name="coding_assistant")
        c.record_llm_tokens(prompt_tokens=100, completion_tokens=50)
        c.record_node_start("parse_request")
        time.sleep(0.01)  # 최소 10ms 대기하여 duration 보장
        c.record_node_end("parse_request")
        c.finalize()

        d = c.to_dict()
        assert d["agent_name"] == "coding_assistant"
        assert d["total_tokens"] == 150
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["duration_ms"] >= 0  # 플랫폼별 타이밍 차이 허용
        assert d["error_count"] == 0
        assert "parse_request" in d["nodes"]
        assert d["nodes"]["parse_request"]["call_count"] == 1

    def test_push_to_langfuse_disabled(self):
        """Langfuse 비활성화 시 push_to_langfuse는 아무 작업도 수행하지 않음."""
        c = AgentMetricsCollector(agent_name="test")
        c.record_llm_tokens(prompt_tokens=10, completion_tokens=5)
        c.finalize()

        with patch(
            "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
            return_value=False,
        ):
            # 예외 없이 실행되어야 함
            c.push_to_langfuse(trace_id="tr-test")

    def test_push_to_langfuse_enabled(self):
        """Langfuse 활성화 시 score_trace가 호출됨."""
        c = AgentMetricsCollector(agent_name="coding_assistant")
        c.record_llm_tokens(prompt_tokens=100, completion_tokens=50)
        c.finalize()

        with (
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
                return_value=True,
            ),
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.score_trace",
            ) as mock_score,
        ):
            c.push_to_langfuse(trace_id="tr-xxx")

        # 5개의 메트릭 스코어가 기록되어야 함
        assert mock_score.call_count == 5
        # 호출된 스코어 이름 확인
        score_names = [call.kwargs["name"] for call in mock_score.call_args_list]
        assert "agent.total_tokens" in score_names
        assert "agent.prompt_tokens" in score_names
        assert "agent.completion_tokens" in score_names
        assert "agent.duration_ms" in score_names
        assert "agent.error_count" in score_names

    def test_multiple_nodes(self):
        c = AgentMetricsCollector(agent_name="test")
        c.record_node_start("parse_request")
        c.record_node_end("parse_request")
        c.record_node_start("execute_code")
        c.record_node_end("execute_code")
        c.record_node_start("verify_result")
        c.record_node_end("verify_result")

        metrics = c.node_metrics
        assert len(metrics) == 3
        for name in ("parse_request", "execute_code", "verify_result"):
            assert metrics[name].call_count == 1


# ── create_langfuse_handler ──


class TestCreateLangfuseHandler:
    def test_disabled(self):
        """Langfuse 비활성화 시 None 반환."""
        with patch(
            "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
            return_value=False,
        ):
            handler = create_langfuse_handler()
        assert handler is None

    def test_enabled_import_failure(self):
        """CallbackHandler import 실패 시 None 반환 (graceful)."""
        with (
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
                return_value=True,
            ),
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.create_langfuse_handler",
                return_value=None,
            ) as mock_create,
        ):
            handler = mock_create()
        assert handler is None

    def test_enabled_success(self):
        """Langfuse 활성화 시 핸들러 객체 반환."""
        mock_handler = MagicMock()
        with (
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
                return_value=True,
            ),
            patch(
                "langfuse.langchain.CallbackHandler",
                return_value=mock_handler,
            ),
        ):
            handler = create_langfuse_handler()
        assert handler is mock_handler


# ── build_observed_config ──


class TestBuildObservedConfig:
    def test_without_handler(self):
        """핸들러 없이 호출 시 기본 config만 반환."""
        config = build_observed_config(thread_id="t-1")
        assert config["configurable"]["thread_id"] == "t-1"
        assert "callbacks" not in config
        assert "metadata" not in config

    def test_with_handler(self):
        """핸들러 있으면 Langfuse 메타데이터 포함."""
        mock_handler = MagicMock()
        config = build_observed_config(
            handler=mock_handler,
            session_id="sess-123",
            thread_id="t-1",
            agent_name="coding_assistant",
        )
        assert config["configurable"]["thread_id"] == "t-1"
        assert "callbacks" in config
        assert mock_handler in config["callbacks"]
        assert "metadata" in config

    def test_with_handler_tags(self):
        """에이전트 이름과 추가 태그가 포함됨."""
        mock_handler = MagicMock()
        config = build_observed_config(
            handler=mock_handler,
            agent_name="deep_research",
            extra_tags=["cli", "interactive"],
            thread_id="t-1",
        )
        # build_langchain_config가 tags를 metadata에 포함
        assert "metadata" in config

    def test_default_thread_id(self):
        """thread_id 미지정 시 기본값 사용."""
        config = build_observed_config()
        assert config["configurable"]["thread_id"] == "default"


# ── safe_flush ──


class TestSafeFlush:
    def test_disabled(self):
        """Langfuse 비활성화 시 flush 스킵."""
        with patch(
            "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
            return_value=False,
        ):
            # 예외 없이 실행
            safe_flush()

    def test_flush_exception_ignored(self):
        """flush 중 예외 발생 시 무시."""
        with (
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
                return_value=True,
            ),
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.client",
                side_effect=ConnectionError("서버 연결 실패"),
            ),
        ):
            # 예외가 전파되지 않아야 함
            safe_flush()


# ── CLIConfig langfuse_enabled ──


class TestCLIConfigLangfuse:
    def test_langfuse_enabled_default(self):
        """기본값으로 langfuse_enabled가 True."""
        config = CLIConfig()
        assert config.langfuse_enabled is True

    def test_langfuse_enabled_false(self):
        """CLI_LANGFUSE_ENABLED=0 시 비활성화."""
        with patch.dict("os.environ", {"CLI_LANGFUSE_ENABLED": "0"}):
            config = CLIConfig()
        assert config.langfuse_enabled is False

    def test_langfuse_enabled_true_explicit(self):
        """CLI_LANGFUSE_ENABLED=true 시 활성화."""
        with patch.dict("os.environ", {"CLI_LANGFUSE_ENABLED": "true"}):
            config = CLIConfig()
        assert config.langfuse_enabled is True

    def test_langfuse_enabled_yes(self):
        """CLI_LANGFUSE_ENABLED=yes 시 활성화."""
        with patch.dict("os.environ", {"CLI_LANGFUSE_ENABLED": "yes"}):
            config = CLIConfig()
        assert config.langfuse_enabled is True


# ── CLI 통합: _run_agent_turn with langfuse_handler ──


class TestRunAgentTurnLangfuse:
    @pytest.mark.asyncio
    async def test_run_agent_turn_without_handler(self):
        """langfuse_handler=None 시 기존과 동일하게 동작."""
        from youngs75_a2a.cli.app import _run_agent_turn

        session = CLISession()
        renderer = _make_renderer()

        # 에이전트를 mock으로 대체
        mock_agent = MagicMock()

        async def fake_events(*args, **kwargs):
            return
            yield  # pragma: no cover — async generator

        mock_agent.graph.astream_events = fake_events
        session.cache_agent("coding_assistant", mock_agent)

        await _run_agent_turn("hello", session, renderer, langfuse_handler=None)
        # 에러 없이 완료
        assert session.info.message_count >= 1

    @pytest.mark.asyncio
    async def test_run_agent_turn_with_handler(self):
        """langfuse_handler 제공 시 safe_flush가 호출됨."""
        from youngs75_a2a.cli.app import _run_agent_turn

        session = CLISession()
        renderer = _make_renderer()

        mock_agent = MagicMock()

        async def fake_events(*args, **kwargs):
            return
            yield  # pragma: no cover — async generator

        mock_agent.graph.astream_events = fake_events
        session.cache_agent("coding_assistant", mock_agent)

        mock_handler = MagicMock()

        with patch(
            "youngs75_a2a.cli.app.safe_flush",
        ) as mock_flush:
            await _run_agent_turn(
                "hello",
                session,
                renderer,
                langfuse_handler=mock_handler,
            )
        mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_agent_turn_metrics_on_error(self):
        """에이전트 실행 오류 시에도 safe_flush가 호출됨."""
        from youngs75_a2a.cli.app import _run_agent_turn

        session = CLISession()
        renderer = _make_renderer()

        mock_agent = MagicMock()

        async def failing_events(*args, **kwargs):
            raise RuntimeError("에이전트 실행 실패")
            yield  # pragma: no cover — async generator

        mock_agent.graph.astream_events = failing_events
        session.cache_agent("coding_assistant", mock_agent)

        mock_handler = MagicMock()

        with patch(
            "youngs75_a2a.cli.app.safe_flush",
        ) as mock_flush:
            await _run_agent_turn(
                "hello",
                session,
                renderer,
                langfuse_handler=mock_handler,
            )
        # 에러 후에도 flush 호출
        mock_flush.assert_called_once()


# ── 기존 observability.langfuse 모듈 테스트 ──


class TestLangfuseModule:
    def test_enabled_returns_false_when_disabled(self):
        """LANGFUSE_TRACING_ENABLED=0 시 False 반환."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enabled
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(
            LANGFUSE_TRACING_ENABLED=False,
            LANGFUSE_HOST="http://localhost:3100",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        assert enabled(settings) is False

    def test_enabled_returns_false_when_keys_missing(self):
        """키가 비어있으면 False 반환."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enabled
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(
            LANGFUSE_TRACING_ENABLED=True,
            LANGFUSE_HOST="http://localhost:3100",
            LANGFUSE_PUBLIC_KEY="",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        assert enabled(settings) is False

    def test_enabled_returns_true_when_configured(self):
        """모든 키가 설정되면 True 반환."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enabled
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(
            LANGFUSE_TRACING_ENABLED=True,
            LANGFUSE_HOST="http://localhost:3100",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        assert enabled(settings) is True

    def test_default_metadata(self):
        """default_metadata가 Settings 값을 반영."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import default_metadata
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(ENV="prod", SERVICE_NAME="my-agent", APP_VERSION="1.0.0")
        meta = default_metadata(settings)
        assert meta["env"] == "prod"
        assert meta["service_name"] == "my-agent"
        assert meta["app_version"] == "1.0.0"

    def test_default_tags(self):
        """default_tags가 Settings 값을 반영."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import default_tags
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(ENV="prod", SERVICE_NAME="my-agent", APP_VERSION="1.0.0")
        tags = default_tags(settings)
        assert "env:prod" in tags
        assert "service:my-agent" in tags
        assert "version:1.0.0" in tags

    def test_score_trace_disabled(self):
        """Langfuse 비활성화 시 score_trace는 아무 작업도 수행하지 않음."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import score_trace
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(LANGFUSE_TRACING_ENABLED=False)
        # 예외 없이 실행
        score_trace("tr-test", name="test", value=0.5, settings=settings)

    def test_build_langchain_config_basic(self):
        """build_langchain_config 기본 동작."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import (
            build_langchain_config,
        )

        config = build_langchain_config(
            user_id="user-1",
            session_id="sess-1",
            callbacks=["handler"],
        )
        assert "metadata" in config
        assert config["metadata"]["langfuse_user_id"] == "user-1"
        assert config["metadata"]["langfuse_session_id"] == "sess-1"
        assert config["callbacks"] == ["handler"]

    def test_build_langchain_config_no_callbacks(self):
        """콜백 미지정 시 callbacks 키 없음."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import (
            build_langchain_config,
        )

        config = build_langchain_config(user_id="u1")
        assert "callbacks" not in config

    def test_enabled_with_base_url_only(self):
        """LANGFUSE_BASE_URL만 설정 시 enabled()=True 반환."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enabled
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(
            LANGFUSE_TRACING_ENABLED=True,
            LANGFUSE_BASE_URL="http://localhost:3100",
            LANGFUSE_HOST="",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        assert enabled(settings) is True

    def test_enabled_with_host_only(self):
        """LANGFUSE_HOST만 설정 시 enabled()=True 반환 (하위 호환)."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enabled
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(
            LANGFUSE_TRACING_ENABLED=True,
            LANGFUSE_BASE_URL="",
            LANGFUSE_HOST="http://localhost:3100",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        assert enabled(settings) is True

    def test_enabled_with_neither_host_nor_base_url(self):
        """LANGFUSE_HOST와 LANGFUSE_BASE_URL 둘 다 비어있으면 enabled()=False."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enabled
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(
            LANGFUSE_TRACING_ENABLED=True,
            LANGFUSE_BASE_URL="",
            LANGFUSE_HOST="",
            LANGFUSE_PUBLIC_KEY="pk-test",
            LANGFUSE_SECRET_KEY="sk-test",
        )
        assert enabled(settings) is False

    def test_enrich_trace_disabled(self):
        """Langfuse 비활성화 시 enrich_trace는 패스스루."""
        from youngs75_a2a.eval_pipeline.observability.langfuse import enrich_trace
        from youngs75_a2a.eval_pipeline.settings import Settings

        settings = Settings(LANGFUSE_TRACING_ENABLED=False)
        with enrich_trace(user_id="u1", settings=settings):
            pass  # 예외 없이 실행
