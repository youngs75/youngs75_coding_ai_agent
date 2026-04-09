from __future__ import annotations

import json

from coding_agent.eval_pipeline.loop3_remediation.recommendation import (
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


class TestReportFormatAndChanges:
    """리포트 포맷팅 및 프롬프트 변경 추출 테스트."""

    def _make_report(self):
        return RecommendationReport(
            summary="에이전트 품질이 임계값 이하입니다",
            failure_analysis=FailureAnalysis(
                total_evaluated=100,
                total_failed=30,
                failure_rate=0.3,
                categories=[
                    FailureCategory(
                        name="generation_failures",
                        count=20,
                        severity="high",
                        patterns=["불완전한 응답"],
                        root_causes=["프롬프트 지시 부족"],
                    ),
                ],
            ),
            prompt_optimizations=[
                PromptOptimization(
                    target_prompt="execute",
                    current_issue="코드 생성 시 타입 힌트 누락",
                    suggested_change="타입 힌트 필수 규칙 추가",
                    expected_metric_improvement="faithfulness +0.1",
                ),
            ],
            recommendations=[
                WorkflowRecommendation(
                    title="verify 프롬프트 강화",
                    category="prompt",
                    priority="high",
                    description="검증 기준이 너무 느슨함",
                    expected_impact="정확성 +15%",
                    implementation_complexity="easy",
                    specific_changes=["보안 검사 항목 추가"],
                ),
                WorkflowRecommendation(
                    title="Retry 로직 추가",
                    category="workflow",
                    priority="medium",
                    description="도구 호출 실패 시 재시도",
                    expected_impact="안정성 +10%",
                    implementation_complexity="medium",
                ),
            ],
            next_steps=["프롬프트 개선 후 재평가"],
        )

    def test_format_report(self):
        report = self._make_report()
        text = report.format_report()
        assert "REMEDIATION REPORT" in text
        assert "에이전트 품질이 임계값 이하입니다" in text
        assert "30.0%" in text
        assert "generation_failures" in text
        assert "타입 힌트 필수 규칙 추가" in text
        assert "verify 프롬프트 강화" in text
        assert "프롬프트 개선 후 재평가" in text

    def test_get_prompt_changes(self):
        report = self._make_report()
        changes = report.get_prompt_changes()
        # prompt_optimizations 1건 + recommendations 중 prompt 카테고리 1건
        assert len(changes) == 2
        assert changes[0]["target_prompt"] == "execute"
        assert changes[0]["change"] == "타입 힌트 필수 규칙 추가"
        # workflow 카테고리 추천은 포함되지 않음
        assert changes[1]["target_prompt"] == "verify 프롬프트 강화"

    def test_get_prompt_changes_empty(self):
        report = RecommendationReport(
            summary="문제 없음",
            failure_analysis=FailureAnalysis(
                total_evaluated=50,
                total_failed=0,
                failure_rate=0.0,
            ),
        )
        assert report.get_prompt_changes() == []

    def test_version_field_default(self):
        report = RecommendationReport(
            summary="테스트",
            failure_analysis=FailureAnalysis(
                total_evaluated=10,
                total_failed=1,
                failure_rate=0.1,
            ),
        )
        assert report.version == "1"

    def test_version_field_custom(self):
        report = RecommendationReport(
            summary="테스트",
            failure_analysis=FailureAnalysis(
                total_evaluated=10,
                total_failed=1,
                failure_rate=0.1,
            ),
            version="2",
        )
        assert report.version == "2"


class TestSaveLoadReport:
    """리포트 저장/로드 테스트."""

    def test_save_and_load(self, tmp_path):
        from coding_agent.eval_pipeline.loop3_remediation.recommendation import (
            load_remediation_report,
            save_remediation_report,
        )

        report = RecommendationReport(
            summary="테스트 리포트",
            failure_analysis=FailureAnalysis(
                total_evaluated=50,
                total_failed=10,
                failure_rate=0.2,
            ),
            next_steps=["다음 단계"],
        )

        path = save_remediation_report(report, output_dir=tmp_path)
        assert path.exists()

        loaded = load_remediation_report(path)
        assert loaded is not None
        assert loaded.summary == "테스트 리포트"
        assert loaded.failure_analysis.total_evaluated == 50

    def test_load_nonexistent(self, tmp_path):
        from coding_agent.eval_pipeline.loop3_remediation.recommendation import (
            load_remediation_report,
        )

        result = load_remediation_report(tmp_path / "nonexistent.json")
        assert result is None


class TestAnalysisTools:
    def test_read_eval_results_empty_dir(self, tmp_path):
        from coding_agent.eval_pipeline.loop3_remediation.analysis_tools import (
            read_eval_results,
        )

        result = read_eval_results.invoke({"results_dir": str(tmp_path)})
        assert result == []

    def test_read_eval_results_with_data(self, tmp_path):
        from coding_agent.eval_pipeline.loop3_remediation.analysis_tools import (
            read_eval_results,
        )

        eval_data = [{"id": "t1", "scores": {"faithfulness": 0.8}, "passed": True}]
        with open(tmp_path / "eval_results.json", "w") as f:
            json.dump(eval_data, f)

        result = read_eval_results.invoke({"results_dir": str(tmp_path)})
        assert len(result) == 1
        assert result[0]["id"] == "t1"
