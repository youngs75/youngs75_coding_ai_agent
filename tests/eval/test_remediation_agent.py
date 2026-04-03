from __future__ import annotations

import json

from youngs75_a2a.eval_pipeline.loop3_remediation.recommendation import (
    FailureAnalysis,
    FailureCategory,
    PromptOptimization,
    RecommendationReport,
    WorkflowRecommendation,
)


class TestRecommendationModels:
    """Pydantic 구조화 출력 모델 테스트."""

    def test_failure_category(self):
        cat = FailureCategory(
            name="retrieval_failures",
            count=5,
            severity="high",
            patterns=["Missing relevant context"],
            root_causes=["Low retriever_top_k"],
        )
        assert cat.count == 5
        assert cat.severity == "high"

    def test_failure_analysis(self):
        analysis = FailureAnalysis(
            total_evaluated=100,
            total_failed=20,
            failure_rate=0.2,
            categories=[
                FailureCategory(name="retrieval", count=10, severity="high"),
                FailureCategory(name="generation", count=10, severity="medium"),
            ],
        )
        assert analysis.failure_rate == 0.2
        assert len(analysis.categories) == 2

    def test_prompt_optimization(self):
        opt = PromptOptimization(
            target_prompt="researcher",
            current_issue="Too broad search queries",
            suggested_change="Add specificity constraints to search",
            expected_metric_improvement="contextual_precision +0.15",
        )
        assert opt.target_prompt == "researcher"

    def test_workflow_recommendation(self):
        rec = WorkflowRecommendation(
            title="Increase retriever_top_k",
            category="parameter",
            priority="high",
            description="Current top_k=3 misses relevant documents",
            expected_impact="Contextual recall +20%",
            implementation_complexity="easy",
            specific_changes=["Set RETRIEVER_TOP_K=5"],
        )
        assert rec.priority == "high"

    def test_full_report(self):
        report = RecommendationReport(
            summary="Agent performance below threshold",
            failure_analysis=FailureAnalysis(
                total_evaluated=50,
                total_failed=15,
                failure_rate=0.3,
            ),
            prompt_optimizations=[
                PromptOptimization(
                    target_prompt="supervisor",
                    current_issue="Vague delegation",
                    suggested_change="Add explicit task descriptions",
                    expected_metric_improvement="task_completion +0.1",
                ),
            ],
            recommendations=[
                WorkflowRecommendation(
                    title="Add retry logic",
                    category="workflow",
                    priority="medium",
                    description="Retry failed tool calls",
                    expected_impact="Tool correctness +10%",
                    implementation_complexity="medium",
                ),
            ],
            next_steps=["Re-run evaluation after changes"],
        )

        data = report.model_dump()
        assert data["failure_analysis"]["failure_rate"] == 0.3
        assert len(data["recommendations"]) == 1

        # JSON 직렬화/역직렬화
        json_str = json.dumps(data, ensure_ascii=False)
        restored = RecommendationReport(**json.loads(json_str))
        assert restored.summary == report.summary


class TestAnalysisTools:
    def test_read_eval_results_empty_dir(self, tmp_path):
        from youngs75_a2a.eval_pipeline.loop3_remediation.analysis_tools import read_eval_results

        result = read_eval_results.invoke({"results_dir": str(tmp_path)})
        assert result == []

    def test_read_eval_results_with_data(self, tmp_path):
        from youngs75_a2a.eval_pipeline.loop3_remediation.analysis_tools import read_eval_results

        eval_data = [{"id": "t1", "scores": {"faithfulness": 0.8}, "passed": True}]
        with open(tmp_path / "eval_results.json", "w") as f:
            json.dump(eval_data, f)

        result = read_eval_results.invoke({"results_dir": str(tmp_path)})
        assert len(result) == 1
        assert result[0]["id"] == "t1"
