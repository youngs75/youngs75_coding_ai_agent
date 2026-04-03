"""Agent Trajectory 평가 메트릭 모듈 (Loop 2).

AI 에이전트의 실행 궤적(trajectory)을 평가하는 메트릭을 구성합니다.
RAG 메트릭이 "답변의 품질"을 평가한다면, Agent 메트릭은
"에이전트의 행동 과정"을 평가합니다.

메트릭 설명:
    1. TaskCompletionMetric (작업 완료도)
       - 에이전트가 주어진 작업을 성공적으로 완료했는지 평가
       - 부분 완료, 오류 발생, 미완료 등을 구분

    2. ToolCorrectnessMetric (도구 사용 정확도)
       - 에이전트가 올바른 도구를 올바른 인자로 호출했는지 평가
       - 불필요한 도구 호출이나 잘못된 인자를 감지

참고:
    Agent 메트릭은 @observe 트레이싱과 evals_iterator에서 추출한
    trajectory 데이터가 필요합니다. Golden Dataset만으로는
    평가할 수 없으며, 실제 에이전트 실행 기록이 있어야 합니다.

사용 예시:
    from youngs75_a2a.eval_pipeline.loop2_evaluation.agent_metrics import create_agent_metrics
    metrics = create_agent_metrics(task_completion_threshold=0.7)
"""

from __future__ import annotations

from deepeval.metrics import (
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)

from youngs75_a2a.eval_pipeline.llm.deepeval_model import get_deepeval_model


def create_agent_metrics(
    *,
    task_completion_threshold: float = 0.7,
    tool_correctness_threshold: float = 0.5,
) -> list:
    """Agent Trajectory 평가 메트릭 인스턴스를 생성합니다.

    Args:
        task_completion_threshold: 작업 완료도 통과 기준 (기본 0.7)
        tool_correctness_threshold: 도구 사용 정확도 통과 기준 (기본 0.5)

    Returns:
        DeepEval Agent 메트릭 인스턴스 리스트 (2개)
    """
    model = get_deepeval_model()

    return [
        TaskCompletionMetric(model=model, threshold=task_completion_threshold),
        ToolCorrectnessMetric(model=model, threshold=tool_correctness_threshold),
    ]
