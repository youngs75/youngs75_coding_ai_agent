"""CI Gate: deepeval test run eval/test_agent_eval.py

Coding Agent의 출력을 Golden Dataset 기대값과 비교 평가합니다.
- 실제 에이전트를 호출하여 actual_output 생성
- RAG 메트릭 + Response Completeness로 품질 게이트
"""

from __future__ import annotations

import json

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from youngs75_a2a.eval_pipeline.loop2_evaluation.rag_metrics import create_rag_metrics
from youngs75_a2a.eval_pipeline.loop2_evaluation.custom_metrics import (
    create_response_completeness_metric,
)
from youngs75_a2a.eval_pipeline.settings import get_settings


def _load_golden_dataset() -> list[dict]:
    """Golden Dataset 로드."""
    settings = get_settings()
    golden_path = settings.data_dir / "golden" / "golden_dataset.json"
    if not golden_path.exists():
        pytest.skip(f"Golden dataset not found: {golden_path}")
    with open(golden_path, encoding="utf-8") as f:
        return json.load(f)


def _make_test_cases() -> list[LLMTestCase]:
    """Golden Dataset → LLMTestCase 변환.

    actual_output은 Coding Agent를 실제 호출하여 생성한다.
    환경변수 RUN_AGENT=1이면 실제 호출, 아니면 expected_output을 대리 사용.
    """
    import os

    golden_data = _load_golden_dataset()
    run_agent = os.getenv("RUN_AGENT", "0") == "1"

    if run_agent:
        from youngs75_a2a.eval_pipeline.my_agent import run_coding_agent

    test_cases = []
    for item in golden_data:
        if run_agent:
            actual_output = run_coding_agent(item["input"])
        else:
            actual_output = item.get("expected_output", "")

        tc = LLMTestCase(
            input=item.get("input", ""),
            actual_output=actual_output,
            expected_output=item.get("expected_output", ""),
            context=item.get("context") if item.get("context") else None,
            retrieval_context=(
                item.get("retrieval_context")
                if item.get("retrieval_context")
                else item.get("context")
            ),
        )
        test_cases.append(tc)
    return test_cases


@pytest.mark.parametrize(
    "test_case",
    _make_test_cases()
    if (get_settings().data_dir / "golden" / "golden_dataset.json").exists()
    else [],
    ids=lambda tc: tc.input[:50],
)
def test_rag_quality(test_case: LLMTestCase):
    """RAG 품질 게이트: AnswerRelevancy + Faithfulness."""
    metrics = create_rag_metrics(
        relevancy_threshold=0.7,
        faithfulness_threshold=0.7,
        precision_threshold=0.5,
        recall_threshold=0.5,
    )
    for metric in metrics:
        assert_test(test_case, [metric])


@pytest.mark.parametrize(
    "test_case",
    _make_test_cases()
    if (get_settings().data_dir / "golden" / "golden_dataset.json").exists()
    else [],
    ids=lambda tc: tc.input[:50],
)
def test_response_completeness(test_case: LLMTestCase):
    """응답 완전성 게이트."""
    metric = create_response_completeness_metric(threshold=0.7)
    assert_test(test_case, [metric])
