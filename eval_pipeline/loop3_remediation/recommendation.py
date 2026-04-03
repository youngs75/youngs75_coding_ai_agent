"""Remediation 구조화 출력 모델 (Loop 3).

Remediation Agent의 출력을 Pydantic 모델로 정의합니다.
비정형 LLM 출력을 구조화된 데이터로 변환하여
프로그래밍적으로 처리할 수 있게 합니다.

모델 계층 구조:
    RecommendationReport (최상위 리포트)
    ├── summary: str
    ├── failure_analysis: FailureAnalysis
    │   └── categories: list[FailureCategory]
    ├── prompt_optimizations: list[PromptOptimization]
    ├── recommendations: list[WorkflowRecommendation]
    └── next_steps: list[str]

사용 예시:
    from youngs75_a2a.eval_pipeline.loop3_remediation.recommendation import RecommendationReport
    report = RecommendationReport(
        summary="에이전트 품질이 임계값 이하입니다",
        failure_analysis=FailureAnalysis(total_evaluated=100, total_failed=30, failure_rate=0.3),
        recommendations=[...],
    )
    print(report.model_dump_json(indent=2))
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FailureCategory(BaseModel):
    """실패 카테고리 모델.

    평가 실패를 유형별로 분류합니다.

    Attributes:
        name: 카테고리명
              (retrieval_failures, generation_failures, agent_failures, safety_failures)
        count: 해당 카테고리 실패 건수
        severity: 심각도 (critical, high, medium, low)
        patterns: 발견된 반복 실패 패턴 (예: "Missing relevant context in top-3 results")
        root_causes: 추정 원인 (예: "retriever_top_k 값이 너무 낮음")
    """

    name: str = Field(
        description="카테고리명 (retrieval_failures, generation_failures, agent_failures, safety_failures)"
    )
    count: int = Field(description="해당 카테고리 실패 건수")
    severity: str = Field(description="심각도 (critical, high, medium, low)")
    patterns: list[str] = Field(default_factory=list, description="발견된 실패 패턴 목록")
    root_causes: list[str] = Field(default_factory=list, description="추정 원인 목록")


class FailureAnalysis(BaseModel):
    """실패 분석 요약 모델.

    전체 평가 결과의 통계와 카테고리별 분류를 포함합니다.

    Attributes:
        total_evaluated: 총 평가 건수
        total_failed: 실패 건수 (하나 이상의 메트릭 미달)
        failure_rate: 실패율 (0.0~1.0)
        categories: 카테고리별 실패 분석 리스트
    """

    total_evaluated: int = Field(description="총 평가 건수")
    total_failed: int = Field(description="실패 건수")
    failure_rate: float = Field(description="실패율 (0.0~1.0)")
    categories: list[FailureCategory] = Field(default_factory=list)


class PromptOptimization(BaseModel):
    """프롬프트 최적화 제안 모델.

    특정 프롬프트에 대한 구체적인 개선 제안을 구조화합니다.

    Attributes:
        target_prompt: 최적화 대상 프롬프트 식별자 (예: "researcher", "supervisor")
        current_issue: 현재 발견된 문제점
        suggested_change: 구체적인 변경 제안 내용
        expected_metric_improvement: 개선 시 예상되는 메트릭 향상 (예: "contextual_precision +0.15")
    """

    target_prompt: str = Field(description="최적화 대상 프롬프트 식별자")
    current_issue: str = Field(description="현재 문제점")
    suggested_change: str = Field(description="제안 변경 내용")
    expected_metric_improvement: str = Field(description="개선 예상 메트릭")


class WorkflowRecommendation(BaseModel):
    """워크플로우 개선 추천 모델.

    에이전트 아키텍처, 파라미터, 정책 등의 변경을 구조화합니다.

    Attributes:
        title: 추천 제목 (간결하게)
        category: 추천 유형 (prompt, workflow, parameter, architecture)
        priority: 우선순위 (high, medium, low)
        description: 상세 설명
        expected_impact: 예상 영향 (어떤 메트릭이 얼마나 개선되는지)
        implementation_complexity: 구현 복잡도 (easy, medium, hard)
        specific_changes: 구체적인 변경 목록 (실행 가능한 수준)
    """

    title: str = Field(description="추천 제목")
    category: str = Field(description="카테고리 (prompt, workflow, parameter, architecture)")
    priority: str = Field(description="우선순위 (high, medium, low)")
    description: str = Field(description="상세 설명")
    expected_impact: str = Field(description="예상 영향")
    implementation_complexity: str = Field(description="구현 복잡도 (easy, medium, hard)")
    specific_changes: list[str] = Field(default_factory=list, description="구체적 변경 목록")


class RecommendationReport(BaseModel):
    """최종 Remediation 추천 리포트 모델.

    Remediation Agent의 최종 출력을 구조화한 최상위 모델입니다.
    JSON 직렬화/역직렬화가 가능하여 파일 저장, API 응답 등에 사용됩니다.

    Attributes:
        summary: 전체 요약 (1~2문장으로 핵심 발견사항)
        failure_analysis: 실패 분석 결과
        prompt_optimizations: 프롬프트 최적화 제안 리스트
        recommendations: 워크플로우 개선 추천 리스트
        next_steps: 다음 단계 액션 아이템 (우선순위 순)
    """

    summary: str = Field(description="전체 요약")
    failure_analysis: FailureAnalysis = Field(description="실패 분석")
    prompt_optimizations: list[PromptOptimization] = Field(default_factory=list)
    recommendations: list[WorkflowRecommendation] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list, description="다음 단계 액션 아이템")
