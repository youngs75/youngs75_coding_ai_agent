"""메트릭 인스턴스 중앙 관리 레지스트리 (Loop 2).

RAG, Agent, Custom 메트릭을 한곳에서 생성하고 관리합니다.
지연 초기화(lazy initialization) 패턴을 사용하여,
실제로 메트릭이 필요할 때만 인스턴스를 생성합니다.

레지스트리 구조:
    ┌─────────────────────┐
    │  MetricsRegistry    │
    ├─────────────────────┤
    │  rag_metrics (4)    │ → Answer Relevancy, Faithfulness,
    │                     │   Contextual Precision, Contextual Recall
    ├─────────────────────┤
    │  agent_metrics (2)  │ → Task Completion, Tool Correctness
    ├─────────────────────┤
    │  custom_metrics (7) │ → Response Completeness, Citation Quality,
    │                     │   Bias, Toxicity, PII, Disclaimer, Safety
    └─────────────────────┘

사용 예시:
    from youngs75_a2a.eval_pipeline.loop2_evaluation.metrics_registry import get_registry

    registry = get_registry()
    rag = registry.rag_metrics            # RAG 메트릭 4개
    all_metrics = registry.all_metrics    # 전체 13개
    names = registry.metric_names("rag")  # ['Answer Relevancy', ...]
"""

from __future__ import annotations

from youngs75_a2a.eval_pipeline.loop2_evaluation.agent_metrics import (
    create_agent_metrics,
)
from youngs75_a2a.eval_pipeline.loop2_evaluation.custom_metrics import (
    SafetyMetric,
    create_bias_metric,
    create_citation_quality_metric,
    create_disclaimer_metric,
    create_pii_metric,
    create_response_completeness_metric,
    create_toxicity_metric,
)
from youngs75_a2a.eval_pipeline.loop2_evaluation.rag_metrics import create_rag_metrics


class MetricsRegistry:
    """메트릭 인스턴스를 카테고리별로 관리하는 레지스트리.

    각 카테고리의 메트릭은 최초 접근 시 생성되고 캐싱됩니다.
    (Lazy initialization: 불필요한 메트릭 생성을 방지)
    """

    def __init__(self):
        # None: 아직 초기화되지 않음을 표시
        self._rag_metrics: list | None = None
        self._agent_metrics: list | None = None
        self._custom_metrics: list | None = None

    @property
    def rag_metrics(self) -> list:
        """RAG 메트릭 4개를 반환합니다 (최초 호출 시 생성)."""
        if self._rag_metrics is None:
            self._rag_metrics = create_rag_metrics()
        return self._rag_metrics

    @property
    def agent_metrics(self) -> list:
        """Agent 메트릭 2개를 반환합니다 (최초 호출 시 생성)."""
        if self._agent_metrics is None:
            self._agent_metrics = create_agent_metrics()
        return self._agent_metrics

    @property
    def custom_metrics(self) -> list:
        """Custom 메트릭 7개를 반환합니다 (최초 호출 시 생성)."""
        if self._custom_metrics is None:
            self._custom_metrics = [
                create_response_completeness_metric(),
                create_citation_quality_metric(),
                create_bias_metric(),
                create_toxicity_metric(),
                create_pii_metric(),
                create_disclaimer_metric(),
                SafetyMetric(),
            ]
        return self._custom_metrics

    @property
    def all_metrics(self) -> list:
        """전체 13개 메트릭을 반환합니다."""
        return self.rag_metrics + self.agent_metrics + self.custom_metrics

    def get_metrics_by_category(self, category: str) -> list:
        """카테고리명으로 메트릭을 조회합니다.

        Args:
            category: "rag", "agent", "custom", 또는 "all"

        Returns:
            해당 카테고리의 메트릭 인스턴스 리스트

        Raises:
            ValueError: 알 수 없는 카테고리명
        """
        match category:
            case "rag":
                return self.rag_metrics
            case "agent":
                return self.agent_metrics
            case "custom":
                return self.custom_metrics
            case "all":
                return self.all_metrics
            case _:
                raise ValueError(
                    f"알 수 없는 카테고리: {category}. rag, agent, custom, all 중 하나를 사용하세요."
                )

    def metric_names(self, category: str = "all") -> list[str]:
        """카테고리별 메트릭 이름 리스트를 반환합니다.

        Args:
            category: "rag", "agent", "custom", 또는 "all"

        Returns:
            메트릭 이름 문자열 리스트
        """
        metrics = self.get_metrics_by_category(category)
        return [getattr(m, "__name__", m.__class__.__name__) for m in metrics]


# 싱글턴: 메트릭 인스턴스를 한 번만 생성하여 재사용
_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    """MetricsRegistry 싱글턴 인스턴스를 반환합니다."""
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry
