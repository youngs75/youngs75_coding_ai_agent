"""관측성 모듈 — Langfuse 통합 및 메트릭 수집.

이 패키지는 Langfuse 기반 관측성을 에이전트 실행 파이프라인에 통합합니다.

주요 컴포넌트:
    - langfuse: 코어 Langfuse SDK 유틸리티 (enabled, client, score_trace, enrich_trace 등)
    - callback_handler: LangChain 콜백 핸들러 팩토리 및 메트릭 수집
"""

from coding_agent.eval_pipeline.observability.callback_handler import (
    AgentMetricsCollector,
    NodeMetrics,
    build_observed_config,
    create_langfuse_handler,
    safe_flush,
)
from coding_agent.eval_pipeline.observability.langfuse import (
    build_langchain_config,
    client,
    default_metadata,
    default_tags,
    enabled,
    enrich_trace,
    score_trace,
)

__all__ = [
    # langfuse.py
    "enabled",
    "client",
    "default_metadata",
    "default_tags",
    "score_trace",
    "enrich_trace",
    "build_langchain_config",
    # callback_handler.py
    "NodeMetrics",
    "AgentMetricsCollector",
    "create_langfuse_handler",
    "build_observed_config",
    "safe_flush",
]
