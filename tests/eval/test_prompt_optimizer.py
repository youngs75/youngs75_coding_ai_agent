from __future__ import annotations

from pathlib import Path

from youngs75_a2a.eval_pipeline.loop2_evaluation.calibration_cases import (
    CalibrationCase,
)
from youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer import (
    CaseScore,
    PromptEvaluation,
    apply_best_prompts_to_file,
    evaluate_prompt,
    load_langfuse_failure_hints,
    optimize_all_prompts,
    optimize_metric_prompt,
)


def test_evaluate_prompt_with_mocked_metric(monkeypatch):
    class FakeMetric:
        def __init__(self):
            self.reason = ""

        def measure(self, test_case):
            if "good" in test_case.actual_output:
                self.reason = "good case"
                return 0.9
            self.reason = "bad case"
            return 0.6

    sample_cases = [
        CalibrationCase(
            case_id="c1",
            metric="safety",
            input="",
            actual_output="good answer",
            expected_output=None,
            context=None,
            retrieval_context=None,
            expected_min=0.8,
            expected_max=1.0,
            note="pass",
        ),
        CalibrationCase(
            case_id="c2",
            metric="safety",
            input="",
            actual_output="bad answer",
            expected_output=None,
            context=None,
            retrieval_context=None,
            expected_min=0.0,
            expected_max=0.3,
            note="fail",
        ),
    ]

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer.cases_for_metric",
        lambda metric, max_cases=None: sample_cases,
    )
    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer._build_metric",
        lambda metric, prompt_text: FakeMetric(),
    )

    evaluation = evaluate_prompt("safety", "prompt")
    assert evaluation.total_cases == 2
    assert evaluation.passed_cases == 1
    assert evaluation.fit_score == 0.5
    assert evaluation.case_scores[1].in_range is False


def test_optimize_metric_prompt_accepts_better_candidate(monkeypatch):
    baseline_eval = PromptEvaluation(
        metric="safety",
        fit_score=0.5,
        passed_cases=1,
        total_cases=2,
        case_scores=[
            CaseScore(
                case_id="c1",
                note="fail",
                score=0.6,
                expected_min=0.0,
                expected_max=0.3,
                in_range=False,
                deviation=0.3,
                reason="too high",
            ),
            CaseScore(
                case_id="c2",
                note="pass",
                score=0.9,
                expected_min=0.8,
                expected_max=1.0,
                in_range=True,
                deviation=0.0,
                reason="ok",
            ),
        ],
    )
    improved_eval = PromptEvaluation(
        metric="safety",
        fit_score=1.0,
        passed_cases=2,
        total_cases=2,
        case_scores=[
            CaseScore(
                case_id="c1",
                note="fixed",
                score=0.2,
                expected_min=0.0,
                expected_max=0.3,
                in_range=True,
                deviation=0.0,
                reason="ok",
            ),
            CaseScore(
                case_id="c2",
                note="pass",
                score=0.9,
                expected_min=0.8,
                expected_max=1.0,
                in_range=True,
                deviation=0.0,
                reason="ok",
            ),
        ],
    )

    evaluations = [baseline_eval, improved_eval]

    def fake_evaluate_prompt(metric, prompt_text, max_cases=None):
        return evaluations.pop(0)

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer.evaluate_prompt",
        fake_evaluate_prompt,
    )
    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer._request_prompt_update",
        lambda **kwargs: {
            "updated_prompt": "improved prompt",
            "change_log": ["clarified rubric"],
            "risk_notes": [],
        },
    )

    result = optimize_metric_prompt(
        "safety",
        baseline_prompt="baseline prompt",
        iterations=1,
    )
    assert result["best_prompt"] == "improved prompt"
    assert result["best_evaluation"]["fit_score"] == 1.0
    assert result["history"][0]["accepted"] is True


def test_apply_best_prompts_to_file(tmp_path: Path):
    prompt_file = tmp_path / "prompts.py"
    prompt_file.write_text(
        'RESPONSE_COMPLETENESS_PROMPT = """\\\nold completeness\n"""\n'
        'CITATION_QUALITY_PROMPT = """\\\nold citation\n"""\n'
        'SAFETY_PROMPT = """\\\nold safety\n"""\n',
        encoding="utf-8",
    )

    apply_best_prompts_to_file(
        {
            "response_completeness": "new completeness",
            "citation_quality": "new citation",
            "safety": "new safety",
        },
        prompts_path=prompt_file,
    )

    content = prompt_file.read_text(encoding="utf-8")
    assert "new completeness" in content
    assert "new citation" in content
    assert "new safety" in content


def test_load_langfuse_failure_hints_maps_metrics(tmp_path: Path):
    failed_path = tmp_path / "langfuse_failed_samples.json"
    failed_path.write_text(
        """
[
  {
    "trace_id": "t1",
    "input": "q1",
    "actual_output": "a1",
    "failed_metrics": ["eval.answer_completeness", "eval.citation_quality", "eval.safety"],
    "scores": {
      "eval.answer_completeness": 0.4,
      "eval.citation_quality": 0.2,
      "eval.safety": 0.5
    },
    "thresholds": {
      "eval.answer_completeness": 0.7,
      "eval.citation_quality": 0.7,
      "eval.safety": 0.8
    }
  }
]
""",
        encoding="utf-8",
    )

    hints = load_langfuse_failure_hints(failed_path, max_per_metric=3)
    assert len(hints["response_completeness"]) == 1
    assert len(hints["citation_quality"]) == 1
    assert len(hints["safety"]) == 1
    assert (
        hints["response_completeness"][0]["source_metric"] == "eval.answer_completeness"
    )


def test_optimize_all_prompts_forwards_langfuse_hints(monkeypatch):
    captured = {}

    def fake_optimize_metric_prompt(
        metric,
        *,
        baseline_prompt,
        iterations,
        max_cases,
        external_failure_hints=None,
    ):
        captured[metric] = external_failure_hints or []
        return {
            "metric": metric,
            "baseline_prompt": baseline_prompt,
            "baseline_evaluation": {"fit_score": 0.1},
            "best_prompt": baseline_prompt,
            "best_evaluation": {"fit_score": 0.2},
            "history": [],
        }

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer.optimize_metric_prompt",
        fake_optimize_metric_prompt,
    )

    hints = {
        "response_completeness": [{"source_metric": "eval.answer_completeness"}],
        "citation_quality": [{"source_metric": "eval.citation_quality"}],
        "safety": [{"source_metric": "eval.safety"}],
    }
    optimize_all_prompts(
        iterations=0,
        langfuse_failure_hints=hints,
    )

    assert captured["response_completeness"] == hints["response_completeness"]
    assert captured["citation_quality"] == hints["citation_quality"]
    assert captured["safety"] == hints["safety"]
