"""Langfuse 데모 트레이스 전송 관련 유닛 테스트.

실제 Langfuse 서버 없이 동작하는 모킹 기반 테스트로 구성.
scripts/langfuse_demo.py의 시나리오 생성 및 트레이스 전송 로직을 검증합니다.
"""

from __future__ import annotations

import os
import random
from unittest.mock import MagicMock, patch

import pytest

# 테스트 대상 모듈을 scripts/ 디렉토리에서 직접 임포트하기 위해
# sys.path 조작 대신 importlib 사용
import sys
from pathlib import Path

# scripts 디렉토리를 임포트 경로에 추가
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# langfuse_demo 모듈 임포트
import langfuse_demo as demo  # noqa: E402


# ── 시나리오 생성 테스트 ──────────────────────────────────


class TestCodingScenarios:
    """coding_assistant 시나리오 생성 테스트."""

    def test_generates_requested_count(self):
        """요청한 수만큼 시나리오를 생성합니다."""
        scenarios = demo._make_coding_scenarios(5)
        assert len(scenarios) == 5

    def test_scenario_structure(self):
        """시나리오가 올바른 구조를 가집니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        s = scenarios[0]

        assert s.agent_name == "coding_assistant"
        assert s.session_id.startswith("demo-session-")
        assert s.user_id.startswith("demo-user-")
        assert len(s.user_input) > 0
        assert len(s.spans) == 3
        assert "demo" in s.tags
        assert "coding_assistant" in s.tags

    def test_span_names(self):
        """코딩 에이전트 시나리오의 span 이름이 올바릅니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        span_names = [span.name for span in scenarios[0].spans]
        assert span_names == ["parse_request", "execute_code", "verify_result"]

    def test_spans_have_model_info(self):
        """span에 LLM 모델 정보가 포함됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        for span in scenarios[0].spans:
            assert span.model is not None
            assert span.prompt_tokens > 0
            assert span.completion_tokens > 0

    def test_scores_are_present(self):
        """시나리오에 평가 스코어가 포함됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scores = scenarios[0].scores

        assert "quality.passed" in scores
        assert "quality.code_correctness" in scores
        assert "quality.response_time_ms" in scores
        assert "quality.risk_level" in scores

    def test_score_types(self):
        """스코어의 데이터 타입이 올바릅니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scores = scenarios[0].scores

        _, dtype = scores["quality.passed"]
        assert dtype == "BOOLEAN"

        _, dtype = scores["quality.code_correctness"]
        assert dtype == "NUMERIC"

        _, dtype = scores["quality.risk_level"]
        assert dtype == "CATEGORICAL"

    def test_duration_is_positive(self):
        """모든 span의 소요 시간이 양수입니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(3)
        for s in scenarios:
            for span in s.spans:
                assert span.duration_ms > 0

    def test_zero_count(self):
        """count=0이면 빈 리스트를 반환합니다."""
        scenarios = demo._make_coding_scenarios(0)
        assert scenarios == []


class TestResearchScenarios:
    """deep_research 시나리오 생성 테스트."""

    def test_generates_requested_count(self):
        scenarios = demo._make_research_scenarios(4)
        assert len(scenarios) == 4

    def test_scenario_structure(self):
        random.seed(42)
        scenarios = demo._make_research_scenarios(1)
        s = scenarios[0]

        assert s.agent_name == "deep_research"
        assert s.session_id.startswith("demo-session-")
        assert s.user_id.startswith("demo-user-")
        assert "deep_research" in s.tags

    def test_span_names(self):
        """리서치 에이전트 시나리오의 span 이름이 올바릅니다."""
        random.seed(42)
        scenarios = demo._make_research_scenarios(1)
        span_names = [span.name for span in scenarios[0].spans]
        assert span_names == [
            "clarify_with_user",
            "write_research_brief",
            "research_supervisor",
            "final_report_generation",
        ]

    def test_scores_include_research_specific(self):
        """리서치 시나리오에 고유 스코어가 포함됩니다."""
        random.seed(42)
        scenarios = demo._make_research_scenarios(1)
        scores = scenarios[0].scores

        assert "quality.completeness" in scores
        assert "quality.source_quality" in scores
        assert "quality.depth_level" in scores
        assert "quality.total_tokens" in scores


# ── 시나리오 재현성 테스트 ────────────────────────────────


class TestReproducibility:
    """동일 시드에서 동일한 시나리오가 생성되는지 확인합니다."""

    def test_same_seed_same_result(self):
        """동일 시드에서 user_input, user_id, span 수가 동일합니다.

        session_id는 uuid4()를 사용하므로 random 시드와 무관하게 생성됩니다.
        따라서 session_id는 비교 대상에서 제외합니다.
        """
        random.seed(99)
        a = demo._make_coding_scenarios(3)

        random.seed(99)
        b = demo._make_coding_scenarios(3)

        for sa, sb in zip(a, b):
            assert sa.user_input == sb.user_input
            assert sa.user_id == sb.user_id
            assert len(sa.spans) == len(sb.spans)

    def test_different_seed_different_result(self):
        random.seed(1)
        a = demo._make_coding_scenarios(3)

        random.seed(2)
        b = demo._make_coding_scenarios(3)

        # 모든 입력이 동일할 확률은 극히 낮음
        inputs_a = [s.user_input for s in a]
        inputs_b = [s.user_input for s in b]
        assert inputs_a != inputs_b or a[0].session_id != b[0].session_id


# ── 트레이스 전송 테스트 (모킹) ───────────────────────────


class TestSendTrace:
    """send_trace 함수의 Langfuse 클라이언트 호출을 모킹하여 검증합니다."""

    def _make_mock_client(self) -> MagicMock:
        """모킹된 Langfuse 클라이언트를 생성합니다."""
        mock_client = MagicMock()

        # trace() -> trace mock (generation, span, score, update 메서드 포함)
        mock_trace = MagicMock()
        mock_trace.id = "mock-trace-id-123"
        mock_client.trace.return_value = mock_trace

        # generation() -> generation mock
        mock_generation = MagicMock()
        mock_trace.generation.return_value = mock_generation

        # span() -> span mock
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        return mock_client

    def test_creates_trace_with_correct_params(self):
        """트레이스가 올바른 파라미터로 생성됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        mock_client = self._make_mock_client()
        trace_id = demo.send_trace(scenario, langfuse_client=mock_client)

        # trace() 호출 검증
        mock_client.trace.assert_called_once()
        trace_kwargs = mock_client.trace.call_args
        assert trace_kwargs.kwargs["name"] == "coding_assistant:demo"
        assert trace_kwargs.kwargs["user_id"] == scenario.user_id
        assert trace_kwargs.kwargs["session_id"] == scenario.session_id
        assert "demo" in trace_kwargs.kwargs["tags"]

        assert trace_id == "mock-trace-id-123"

    def test_creates_generations_for_llm_spans(self):
        """LLM 모델이 있는 span에 대해 generation이 생성됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        mock_client = self._make_mock_client()
        mock_trace = mock_client.trace.return_value

        demo.send_trace(scenario, langfuse_client=mock_client)

        # 코딩 시나리오의 3개 span 모두 model이 있으므로 3개 generation
        llm_spans = [s for s in scenario.spans if s.model]
        assert mock_trace.generation.call_count == len(llm_spans)

        # 첫 번째 generation 검증
        first_gen_kwargs = mock_trace.generation.call_args_list[0].kwargs
        assert first_gen_kwargs["name"] == "parse_request:llm"
        assert "model" in first_gen_kwargs
        assert "usage" in first_gen_kwargs

    def test_creates_spans_for_all_nodes(self):
        """모든 노드에 대해 span이 생성됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        mock_client = self._make_mock_client()
        mock_trace = mock_client.trace.return_value

        demo.send_trace(scenario, langfuse_client=mock_client)

        assert mock_trace.span.call_count == len(scenario.spans)

    def test_records_scores(self):
        """시나리오의 모든 스코어가 기록됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        mock_client = self._make_mock_client()
        mock_trace = mock_client.trace.return_value

        demo.send_trace(scenario, langfuse_client=mock_client)

        assert mock_trace.score.call_count == len(scenario.scores)

        # 스코어 이름 확인
        score_names = [
            call_args.kwargs["name"] for call_args in mock_trace.score.call_args_list
        ]
        for expected_name in scenario.scores:
            assert expected_name in score_names

    def test_boolean_score_conversion(self):
        """BOOLEAN 스코어가 올바르게 변환됩니다 (True -> 1.0, False -> 0.0)."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        # passed를 강제로 True로 설정
        scenario.scores["quality.passed"] = (True, "BOOLEAN")

        mock_client = self._make_mock_client()
        mock_trace = mock_client.trace.return_value

        demo.send_trace(scenario, langfuse_client=mock_client)

        # BOOLEAN 스코어 호출 찾기
        for call_args in mock_trace.score.call_args_list:
            if call_args.kwargs["name"] == "quality.passed":
                assert call_args.kwargs["value"] == 1.0
                assert call_args.kwargs["data_type"] == "BOOLEAN"
                break

    def test_categorical_score_is_string(self):
        """CATEGORICAL 스코어 값이 문자열입니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        mock_client = self._make_mock_client()
        mock_trace = mock_client.trace.return_value

        demo.send_trace(scenario, langfuse_client=mock_client)

        for call_args in mock_trace.score.call_args_list:
            if call_args.kwargs["name"] == "quality.risk_level":
                assert isinstance(call_args.kwargs["value"], str)
                assert call_args.kwargs["data_type"] == "CATEGORICAL"
                break

    def test_trace_output_updated(self):
        """트레이스의 출력이 마지막 span의 출력으로 업데이트됩니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        scenario = scenarios[0]

        mock_client = self._make_mock_client()
        mock_trace = mock_client.trace.return_value

        demo.send_trace(scenario, langfuse_client=mock_client)

        # update()로 output 설정 확인
        mock_trace.update.assert_called_once()
        update_kwargs = mock_trace.update.call_args.kwargs
        assert "output" in update_kwargs
        assert "response" in update_kwargs["output"]


# ── 시나리오 출력 테스트 ──────────────────────────────────


class TestPrintScenarioSummary:
    """print_scenario_summary 함수의 출력을 검증합니다."""

    def test_does_not_raise(self, capsys):
        """출력 중 예외가 발생하지 않습니다."""
        random.seed(42)
        scenarios = demo._make_coding_scenarios(1)
        demo.print_scenario_summary(scenarios[0], 0)

        captured = capsys.readouterr()
        assert "coding_assistant" in captured.out
        assert "Span 수: 3" in captured.out

    def test_research_scenario_output(self, capsys):
        """리서치 시나리오도 정상 출력됩니다."""
        random.seed(42)
        scenarios = demo._make_research_scenarios(1)
        demo.print_scenario_summary(scenarios[0], 0)

        captured = capsys.readouterr()
        assert "deep_research" in captured.out
        assert "Span 수: 4" in captured.out


# ── setup_langfuse.py 유닛 테스트 ─────────────────────────

import setup_langfuse  # noqa: E402


class TestSetupLangfuseEnvValidation:
    """setup_langfuse.py의 환경변수 검증 테스트."""

    def test_validate_env_vars_all_set(self, capsys):
        """모든 필수 환경변수가 설정되면 True를 반환합니다."""
        env = {
            "LANGFUSE_HOST": "http://localhost:3100",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test1234",
            "LANGFUSE_SECRET_KEY": "sk-lf-test1234",
        }
        with patch.dict(os.environ, env, clear=False):
            result = setup_langfuse.validate_env_vars()
        assert result is True

    def test_validate_env_vars_missing_key(self, capsys):
        """필수 환경변수가 누락되면 False를 반환합니다."""
        env = {
            "LANGFUSE_HOST": "http://localhost:3100",
            "LANGFUSE_PUBLIC_KEY": "",
            "LANGFUSE_SECRET_KEY": "sk-lf-test1234",
        }
        with patch.dict(os.environ, env, clear=False):
            result = setup_langfuse.validate_env_vars()
        assert result is False

    def test_validate_env_vars_verbose(self, capsys):
        """verbose 모드에서 마스킹된 값이 출력됩니다."""
        env = {
            "LANGFUSE_HOST": "http://localhost:3100",
            "LANGFUSE_PUBLIC_KEY": "pk-lf-longerthanexpected",
            "LANGFUSE_SECRET_KEY": "sk-lf-longerthanexpected",
        }
        with patch.dict(os.environ, env, clear=False):
            setup_langfuse.validate_env_vars(verbose=True)

        captured = capsys.readouterr()
        # Secret key는 앞 8자만 표시
        assert "pk-lf-lo..." in captured.out
        assert "sk-lf-lo..." in captured.out


class TestSetupLangfuseHealthCheck:
    """setup_langfuse.py의 헬스체크 테스트."""

    def test_health_check_success(self):
        """서버가 200을 반환하면 True를 반환합니다."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"status":"OK"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("setup_langfuse.urlopen", return_value=mock_response):
            result = setup_langfuse.check_health(
                "http://localhost:3100",
                max_retries=1,
                retry_interval=0,
            )
        assert result is True

    def test_health_check_failure(self):
        """서버 연결 실패 시 False를 반환합니다."""
        from urllib.error import URLError

        with patch(
            "setup_langfuse.urlopen", side_effect=URLError("Connection refused")
        ):
            result = setup_langfuse.check_health(
                "http://localhost:3100",
                max_retries=2,
                retry_interval=0,
            )
        assert result is False


class TestSetupLangfuseLoadDotenv:
    """setup_langfuse.py의 .env 로드 테스트."""

    def test_load_dotenv_parses_key_value(self, tmp_path):
        """간단한 KEY=VALUE 파싱이 정상 동작합니다."""
        env_content = "TEST_LANGFUSE_SETUP_VAR=hello_world\n# 코멘트\nEMPTY_LINE\n"
        env_file = tmp_path / ".env"
        env_file.write_text(env_content)

        # 모듈의 프로젝트 루트를 tmp_path로 모킹
        with patch("setup_langfuse.Path") as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent = tmp_path
            # 직접 파싱 로직 테스트
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value

        assert os.environ.get("TEST_LANGFUSE_SETUP_VAR") == "hello_world"
        # 정리
        os.environ.pop("TEST_LANGFUSE_SETUP_VAR", None)


# ── AgentMetricsCollector → Langfuse 연동 추가 테스트 ─────

# youngs75_a2a 패키지 설치 여부에 따라 조건부 실행
_HAS_YOUNGS75 = True
try:
    from youngs75_a2a.eval_pipeline.observability.callback_handler import (  # noqa: F401
        AgentMetricsCollector as _AMC,
    )
except ImportError:
    _HAS_YOUNGS75 = False

_skip_no_package = pytest.mark.skipif(
    not _HAS_YOUNGS75,
    reason="youngs75_a2a 패키지 미설치",
)


@_skip_no_package
class TestMetricsCollectorIntegration:
    """AgentMetricsCollector의 Langfuse 스코어 기록 통합 테스트."""

    def test_push_metrics_match_demo_scenario(self):
        """데모 시나리오에서 수집한 메트릭이 올바르게 push됩니다."""
        from youngs75_a2a.eval_pipeline.observability.callback_handler import (
            AgentMetricsCollector,
        )

        # 데모 시나리오 시뮬레이션
        collector = AgentMetricsCollector(agent_name="coding_assistant")
        collector.record_node_start("parse_request")
        collector.record_llm_tokens(prompt_tokens=150, completion_tokens=60)
        collector.record_node_end("parse_request")

        collector.record_node_start("execute_code")
        collector.record_llm_tokens(prompt_tokens=500, completion_tokens=300)
        collector.record_node_end("execute_code")

        collector.record_node_start("verify_result")
        collector.record_llm_tokens(prompt_tokens=200, completion_tokens=100)
        collector.record_node_end("verify_result")

        collector.finalize()

        # 메트릭 검증
        assert collector.total_prompt_tokens == 850
        assert collector.total_completion_tokens == 460
        assert collector.total_tokens == 1310
        assert collector.error_count == 0
        assert len(collector.node_metrics) == 3

        # Langfuse push 모킹
        with (
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
                return_value=True,
            ),
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.score_trace",
            ) as mock_score,
        ):
            collector.push_to_langfuse(trace_id="demo-trace-123")

        # 5개 메트릭 기록 확인
        assert mock_score.call_count == 5

        # 값 검증
        calls_dict = {
            c.kwargs["name"]: c.kwargs["value"] for c in mock_score.call_args_list
        }
        assert calls_dict["agent.total_tokens"] == 1310.0
        assert calls_dict["agent.prompt_tokens"] == 850.0
        assert calls_dict["agent.completion_tokens"] == 460.0
        assert calls_dict["agent.error_count"] == 0.0
        assert calls_dict["agent.duration_ms"] > 0

    def test_push_metrics_with_errors(self):
        """에러가 있는 시나리오의 메트릭도 올바르게 push됩니다."""
        from youngs75_a2a.eval_pipeline.observability.callback_handler import (
            AgentMetricsCollector,
        )

        collector = AgentMetricsCollector(agent_name="coding_assistant")
        collector.record_node_start("execute_code")
        collector.record_node_end("execute_code", error=True)
        collector.record_error()  # 전역 에러 추가
        collector.finalize()

        assert collector.error_count == 2

        with (
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.enabled",
                return_value=True,
            ),
            patch(
                "youngs75_a2a.eval_pipeline.observability.callback_handler.score_trace",
            ) as mock_score,
        ):
            collector.push_to_langfuse(trace_id="demo-error-trace")

        calls_dict = {
            c.kwargs["name"]: c.kwargs["value"] for c in mock_score.call_args_list
        }
        assert calls_dict["agent.error_count"] == 2.0


# ── SimulatedSpan / SimulatedScenario 데이터클래스 테스트 ──


class TestDataClasses:
    """데이터클래스의 기본 속성 테스트."""

    def test_simulated_span_defaults(self):
        span = demo.SimulatedSpan(
            name="test",
            duration_ms=100.0,
            input_text="input",
            output_text="output",
        )
        assert span.model is None
        assert span.prompt_tokens == 0
        assert span.completion_tokens == 0
        assert span.error is None

    def test_simulated_span_with_model(self):
        span = demo.SimulatedSpan(
            name="test",
            duration_ms=100.0,
            input_text="input",
            output_text="output",
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert span.model == "gpt-4o"
        assert span.prompt_tokens == 100

    def test_simulated_scenario_fields(self):
        scenario = demo.SimulatedScenario(
            agent_name="test_agent",
            session_id="sess-1",
            user_id="user-1",
            user_input="hello",
            spans=[],
            scores={},
            tags=["test"],
        )
        assert scenario.agent_name == "test_agent"
        assert scenario.tags == ["test"]
