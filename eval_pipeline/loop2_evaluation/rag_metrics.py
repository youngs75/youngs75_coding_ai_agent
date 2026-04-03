"""RAG 평가 메트릭 모듈 (Loop 2).

Retrieval-Augmented Generation(RAG) 시스템의 품질을 측정하는
4가지 핵심 메트릭을 DeepEval로 구성합니다.

메트릭 설명:
    1. AnswerRelevancyMetric (답변 관련성)
       - 답변이 질문에 얼마나 관련있는지 평가
       - "질문: SLA란?" → 답변이 SLA에 대해 다루고 있는지

    2. FaithfulnessMetric (충실도)
       - 답변이 주어진 컨텍스트에 충실한지 평가 (환각 탐지)
       - 컨텍스트에 없는 내용을 지어냈는지 검사

    3. ContextualPrecisionMetric (컨텍스트 정밀도)
       - 검색된 컨텍스트 중 실제로 답변에 도움이 되는 비율
       - 불필요한 컨텍스트가 많으면 정밀도가 낮아짐

    4. ContextualRecallMetric (컨텍스트 재현율)
       - 답변에 필요한 정보가 검색된 컨텍스트에 얼마나 포함되어 있는지
       - 핵심 정보가 누락되면 재현율이 낮아짐

사용 예시:
    from youngs75_a2a.eval_pipeline.loop2_evaluation.rag_metrics import create_rag_metrics
    metrics = create_rag_metrics(relevancy_threshold=0.7)
    # → [AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric]
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

from youngs75_a2a.eval_pipeline.llm.deepeval_model import get_deepeval_model


def create_rag_metrics(
    *,
    relevancy_threshold: float = 0.7,
    faithfulness_threshold: float = 0.7,
    precision_threshold: float = 0.5,
    recall_threshold: float = 0.5,
) -> list:
    """RAG 평가 메트릭 인스턴스 4개를 생성합니다.

    모든 메트릭은 OpenRouter 모델을 사용하여 LLM 기반 평가를 수행합니다.
    threshold는 "통과" 기준 점수이며, 이 값 이상이면 해당 메트릭을 통과합니다.

    Args:
        relevancy_threshold: 답변 관련성 통과 기준 (기본 0.7)
        faithfulness_threshold: 충실도 통과 기준 (기본 0.7)
        precision_threshold: 컨텍스트 정밀도 통과 기준 (기본 0.5)
        recall_threshold: 컨텍스트 재현율 통과 기준 (기본 0.5)

    Returns:
        DeepEval 메트릭 인스턴스 리스트 (4개)
    """
    # OpenRouter 모델을 모든 메트릭에 공통으로 전달
    model = get_deepeval_model()

    return [
        AnswerRelevancyMetric(model=model, threshold=relevancy_threshold),
        FaithfulnessMetric(model=model, threshold=faithfulness_threshold),
        ContextualPrecisionMetric(model=model, threshold=precision_threshold),
        ContextualRecallMetric(model=model, threshold=recall_threshold),
    ]
