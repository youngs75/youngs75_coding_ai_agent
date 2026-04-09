from __future__ import annotations

from coding_agent.eval_pipeline.loop2_evaluation.calibration_cases import (
    all_calibration_cases,
    cases_for_metric,
)


def test_all_cases_loaded():
    cases = all_calibration_cases()
    assert len(cases) >= 18


def test_cases_for_metric_filter():
    completeness_cases = cases_for_metric("response_completeness")
    citation_cases = cases_for_metric("citation_quality")
    safety_cases = cases_for_metric("safety")

    assert len(completeness_cases) >= 6
    assert len(citation_cases) >= 6
    assert len(safety_cases) >= 6


def test_max_cases_limit():
    limited = cases_for_metric("safety", max_cases=2)
    assert len(limited) == 2


def test_case_to_testcase():
    sample = cases_for_metric("response_completeness", max_cases=1)[0]
    test_case = sample.to_test_case()

    assert test_case.input == sample.input
    assert test_case.actual_output == sample.actual_output
    assert test_case.expected_output == sample.expected_output
