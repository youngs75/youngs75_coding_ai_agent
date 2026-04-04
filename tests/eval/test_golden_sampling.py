from __future__ import annotations

import json

from youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator import (
    evaluate_golden_dataset,
    sample_golden_items,
)


def _build_grouped_golden_items() -> list[dict]:
    grouped_sizes = {
        "faq_pricing.md": 6,
        "faq_security.md": 3,
        "faq_sla.md": 1,
    }
    items: list[dict] = []
    for source_file, size in grouped_sizes.items():
        for index in range(size):
            items.append(
                {
                    "id": f"{source_file}-{index}",
                    "input": f"{source_file} 질문 {index}",
                    "expected_output": f"{source_file} 정답 {index}",
                    "source_file": source_file,
                    "context": [f"{source_file} context {index}"],
                    "retrieval_context": [f"{source_file} retrieval {index}"],
                }
            )
    return items


def test_sample_golden_items_is_deterministic_with_seed():
    items = [
        {
            "id": f"item-{index}",
            "input": f"q-{index}",
            "expected_output": f"a-{index}",
        }
        for index in range(80)
    ]

    sampled_a = sample_golden_items(items, sample_ratio=0.25, sample_seed=11)
    sampled_b = sample_golden_items(items, sample_ratio=0.25, sample_seed=11)
    sampled_c = sample_golden_items(items, sample_ratio=0.25, sample_seed=12)

    ids_a = [item["id"] for item in sampled_a]
    ids_b = [item["id"] for item in sampled_b]
    ids_c = [item["id"] for item in sampled_c]

    assert len(ids_a) == 20
    assert ids_a == ids_b
    assert ids_a != ids_c


def test_sample_golden_items_stratify_keeps_small_group():
    items = _build_grouped_golden_items()

    sampled = sample_golden_items(
        items,
        max_sample_size=5,
        sample_seed=42,
        stratify_by=["source_file"],
    )

    sampled_groups = {item["source_file"] for item in sampled}
    assert len(sampled) == 5
    assert sampled_groups == {"faq_pricing.md", "faq_security.md", "faq_sla.md"}


def test_evaluate_golden_dataset_applies_sampling_options(monkeypatch, tmp_path):
    golden_items = _build_grouped_golden_items()
    golden_path = tmp_path / "golden_dataset.json"
    golden_path.write_text(
        json.dumps(golden_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_path = tmp_path / "eval_results.json"

    class _FakeRegistry:
        def get_metrics_by_category(self, _category: str):
            return ["mock_metric"]

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator.get_registry",
        lambda: _FakeRegistry(),
    )
    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator._run_metrics_on_testcase",
        lambda _test_case, _metrics: ({"mock_metric": 0.9}, []),
    )

    results = evaluate_golden_dataset(
        golden_path=golden_path,
        metric_categories=["custom"],
        sample_ratio=0.8,
        max_sample_size=4,
        sample_seed=99,
        stratify_by=["source_file"],
        output_path=output_path,
    )

    assert len(results) == 4
    assert output_path.exists()
    stored = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(stored) == 4
    assert all(result["scores"]["mock_metric"] == 0.9 for result in stored)


def test_step5_run_evaluation_forwards_sampling_options(monkeypatch):
    from youngs75_a2a.scripts.run_pipeline import step5_run_evaluation

    captured: dict = {}

    def fake_evaluate_golden_dataset(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator.evaluate_golden_dataset",
        fake_evaluate_golden_dataset,
    )

    step5_run_evaluation(
        categories=["custom"],
        sample_ratio=0.3,
        sample_size=9,
        sample_seed=7,
        stratify_by=["source_file", "difficulty"],
    )

    assert captured["metric_categories"] == ["custom"]
    assert captured["sample_ratio"] == 0.3
    assert captured["max_sample_size"] == 9
    assert captured["sample_seed"] == 7
    assert captured["stratify_by"] == ["source_file", "difficulty"]
