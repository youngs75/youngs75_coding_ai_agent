from __future__ import annotations

from unittest.mock import MagicMock, patch

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


class TestGEvalMetrics:
    """GEval 기반 커스텀 메트릭 생성 테스트.

    각 메트릭이 올바른 이름과 설정으로 생성되는지 검증합니다.
    DeepEval 설계 원칙(한 메트릭 = 한 가지 관점)을 따르는지 확인합니다.
    """

    def test_response_completeness_creation(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_response_completeness_metric,
            )

            metric = create_response_completeness_metric(threshold=0.7)
            assert metric.name == "Response Completeness"

    def test_response_completeness_override(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_response_completeness_metric,
            )

            metric = create_response_completeness_metric(
                threshold=0.7,
                criteria_override="override criteria",
            )
            assert metric.criteria == "override criteria"

    def test_citation_quality_creation(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_citation_quality_metric,
            )

            metric = create_citation_quality_metric(threshold=0.5)
            assert metric.name == "Citation Quality"

    def test_citation_quality_override(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_citation_quality_metric,
            )

            metric = create_citation_quality_metric(
                threshold=0.5,
                criteria_override="citation override",
            )
            assert metric.criteria == "citation override"

    def test_bias_creation(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_bias_metric,
            )

            metric = create_bias_metric(threshold=0.8)
            assert metric.name == "Bias"

    def test_toxicity_creation(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_toxicity_metric,
            )

            metric = create_toxicity_metric(threshold=0.8)
            assert metric.name == "Toxicity"

    def test_pii_creation(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_pii_metric,
            )

            metric = create_pii_metric(threshold=0.8)
            assert metric.name == "PII"

    def test_disclaimer_creation(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model_fn.return_value = MagicMock(spec=DeepEvalBaseLLM)
            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                create_disclaimer_metric,
            )

            metric = create_disclaimer_metric(threshold=0.7)
            assert metric.name == "Disclaimer"


class TestSafetyMetric:
    def test_safe_response(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model = MagicMock()
            mock_model.generate.return_value = (
                '{"score": 0.95, "reason": "Safe response"}'
            )
            mock_model_fn.return_value = mock_model

            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                SafetyMetric,
            )

            metric = SafetyMetric(threshold=0.8)
            score = metric.measure(
                LLMTestCase(
                    input="What is SLA?",
                    actual_output="SLA is a service-level agreement.",
                )
            )

            assert score == 0.95
            assert metric.is_successful()

    def test_json_code_fence_response(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model = MagicMock()
            mock_model.generate.return_value = (
                '```json\n{"score": 0.72, "reason": "Minor issue"}\n```'
            )
            mock_model_fn.return_value = mock_model

            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                SafetyMetric,
            )

            metric = SafetyMetric(threshold=0.8)
            score = metric.measure(
                LLMTestCase(
                    input="test",
                    actual_output="test output",
                )
            )

            assert score == 0.72
            assert not metric.is_successful()

    def test_evaluation_error(self):
        with patch(
            "coding_agent.eval_pipeline.loop2_evaluation.custom_metrics.get_deepeval_model"
        ) as mock_model_fn:
            mock_model = MagicMock()
            mock_model.generate.side_effect = Exception("API error")
            mock_model_fn.return_value = mock_model

            from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
                SafetyMetric,
            )

            metric = SafetyMetric(threshold=0.8)
            score = metric.measure(
                LLMTestCase(
                    input="test",
                    actual_output="test output",
                )
            )

            assert score == 0.0
            assert not metric.is_successful()
