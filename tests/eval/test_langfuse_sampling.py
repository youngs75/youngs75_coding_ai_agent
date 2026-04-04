from __future__ import annotations

from types import SimpleNamespace

from deepeval.test_case import LLMTestCase

from youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator import (
    _run_metrics_on_testcase,
    batch_evaluate_langfuse,
    monitor_langfuse_scores,
)
from youngs75_a2a.eval_pipeline.loop2_evaluation.langfuse_bridge import fetch_traces


class _FakeScore:
    def __init__(
        self,
        name: str,
        value=None,
        data_type: str = "NUMERIC",
        comment: str | None = None,
    ):
        self.name = name
        self.value = value
        self.data_type = data_type
        self.comment = comment


class _FakeTrace:
    def __init__(
        self, trace_id: str, scores=None, *, input=None, output=None, metadata=None
    ):
        self.id = trace_id
        self.scores = scores or []
        self.input = input
        self.output = output
        self.metadata = metadata or {}


class _FakeTraceAPI:
    """Langfuse SDK v3의 lf.api.trace 를 모킹하는 클래스."""

    def __init__(self, traces):
        self._traces = {t.id: t for t in traces}
        self._traces_list = list(traces)

    def list(self, **_kwargs):
        return SimpleNamespace(data=list(self._traces_list))

    def get(self, trace_id: str):
        return self._traces[trace_id]


class _FakeLangfuseClient:
    def __init__(self, traces):
        self._traces = traces
        self.api = SimpleNamespace(trace=_FakeTraceAPI(traces))


def test_fetch_traces_applies_sampling_and_skip_prefix(monkeypatch):
    traces = [
        _FakeTrace("t0", scores=[_FakeScore("deepeval.answer_relevancy")]),
        *[_FakeTrace(f"t{i}") for i in range(1, 10)],
    ]

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.langfuse_bridge.enabled",
        lambda _settings: True,
    )
    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.langfuse_bridge.client",
        lambda: _FakeLangfuseClient(traces),
    )

    sample_a = fetch_traces(
        limit=10,
        sample_ratio=0.5,
        sample_seed=7,
        exclude_scored_prefix="deepeval",
    )
    sample_b = fetch_traces(
        limit=10,
        sample_ratio=0.5,
        sample_seed=7,
        exclude_scored_prefix="deepeval",
    )
    sample_c = fetch_traces(
        limit=10,
        sample_ratio=0.5,
        sample_seed=8,
        exclude_scored_prefix="deepeval",
    )

    ids_a = [trace.id for trace in sample_a]
    ids_b = [trace.id for trace in sample_b]
    ids_c = [trace.id for trace in sample_c]

    assert "t0" not in ids_a
    assert ids_a == ids_b
    assert len(ids_a) == 4  # 9개(스킵 1개)에서 50% 샘플링
    assert ids_a != ids_c


def test_batch_evaluate_langfuse_forwards_sampling_options(monkeypatch, tmp_path):
    captured = {}

    def fake_fetch_traces(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator.fetch_traces",
        fake_fetch_traces,
    )

    results = batch_evaluate_langfuse(
        from_hours_ago=12,
        limit=300,
        sample_ratio=0.25,
        max_sample_size=20,
        sample_seed=99,
        skip_scored_prefix="deepeval",
        output_path=tmp_path / "langfuse_batch_results.json",
    )

    assert results == []
    assert captured["from_hours_ago"] == 12
    assert captured["limit"] == 300
    assert captured["sample_ratio"] == 0.25
    assert captured["max_sample_size"] == 20
    assert captured["sample_seed"] == 99
    assert captured["exclude_scored_prefix"] == "deepeval"


class _FakeMetric:
    def __init__(self, metric_name: str, score: float = 0.9):
        self.__name__ = metric_name
        self.score = score
        self.calls = 0

    def measure(self, _test_case):
        self.calls += 1


def test_run_metrics_on_testcase_skips_missing_required_fields():
    faithfulness = _FakeMetric("Faithfulness")
    test_case = LLMTestCase(
        input="q",
        actual_output="a",
        retrieval_context=None,
    )

    scores, skipped = _run_metrics_on_testcase(test_case, [faithfulness])

    assert scores == {}
    assert skipped == ["faithfulness"]
    assert faithfulness.calls == 0


def test_run_metrics_on_testcase_runs_when_required_fields_exist():
    faithfulness = _FakeMetric("Faithfulness")
    test_case = LLMTestCase(
        input="q",
        actual_output="a",
        retrieval_context=["ctx"],
    )

    scores, skipped = _run_metrics_on_testcase(test_case, [faithfulness])

    assert skipped == []
    assert scores["faithfulness"] == 0.9
    assert faithfulness.calls == 1


def test_monitor_langfuse_scores_extracts_failed_samples(monkeypatch, tmp_path):
    traces = [
        _FakeTrace(
            "ok1",
            scores=[_FakeScore("eval.faithfulness", 0.91, "NUMERIC")],
            input={"query": "What is SLA?"},
            output={"answer": "SLA is a service-level agreement."},
        ),
        _FakeTrace(
            "fail1",
            scores=[
                _FakeScore("eval.faithfulness", 0.4, "NUMERIC", "Not grounded"),
                _FakeScore("eval.safety", 0.9, "NUMERIC"),
            ],
            input={"query": "Give medical advice"},
            output={"answer": "stop meds now"},
        ),
        _FakeTrace(
            "missing1",
            scores=[],
            input={"query": "No eval score trace"},
            output={"answer": "output"},
        ),
    ]

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator.fetch_traces",
        lambda **_kwargs: traces,
    )

    snapshot = monitor_langfuse_scores(
        score_prefix="eval",
        default_threshold=0.7,
        output_path=tmp_path / "snapshot.json",
        failed_output_path=tmp_path / "failed.json",
    )

    assert snapshot["summary"]["total_sampled"] == 3
    assert snapshot["summary"]["total_collected"] == 2
    assert snapshot["summary"]["traces_failed"] == 1
    assert snapshot["failed_samples"][0]["trace_id"] == "fail1"
    assert snapshot["failed_samples"][0]["failed_metrics"] == ["faithfulness"]


def test_monitor_langfuse_scores_metric_threshold_override(monkeypatch, tmp_path):
    traces = [
        _FakeTrace(
            "t1",
            scores=[_FakeScore("eval.faithfulness", 0.75, "NUMERIC")],
            input={"query": "q"},
            output={"answer": "a"},
        ),
    ]
    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator.fetch_traces",
        lambda **_kwargs: traces,
    )

    snapshot = monitor_langfuse_scores(
        score_prefix="eval",
        default_threshold=0.7,
        metric_thresholds={"faithfulness": 0.8},
        output_path=tmp_path / "snapshot.json",
        failed_output_path=tmp_path / "failed.json",
    )

    assert snapshot["summary"]["traces_failed"] == 1
