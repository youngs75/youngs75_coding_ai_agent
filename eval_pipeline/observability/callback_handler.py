"""Langfuse 콜백 핸들러 팩토리 및 메트릭 수집 모듈.

LangChain/LangGraph 에이전트 실행 파이프라인에 Langfuse 콜백 핸들러를
통합하여 트레이스, 메트릭, 성능 데이터를 자동 수집합니다.

핵심 기능:
    1. 콜백 핸들러 생성 — create_langfuse_handler()
    2. 에이전트 실행 config 구성 — build_observed_config()
    3. 메트릭 수집 및 기록 — AgentMetricsCollector
    4. 안전한 flush — safe_flush()

사용 예시 (CLI 통합):
    from youngs75_a2a.eval_pipeline.observability.callback_handler import (
        create_langfuse_handler,
        build_observed_config,
        safe_flush,
    )

    handler = create_langfuse_handler()
    if handler:
        config = build_observed_config(
            handler=handler,
            session_id="sess-123",
            user_id="user-abc",
            thread_id="thread-1",
        )
        result = await agent.graph.astream_events(input_state, config=config, version="v2")
        safe_flush()

설계 원칙:
    - Langfuse 서버 연결 실패 시 graceful 처리 (None 반환, 로그 경고)
    - enabled() 가드 패턴을 일관되게 사용
    - 기존 observability.langfuse 모듈의 유틸리티 재사용
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from youngs75_a2a.eval_pipeline.observability.langfuse import (
    build_langchain_config,
    client,
    enabled,
    score_trace,
)
from youngs75_a2a.eval_pipeline.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """개별 노드(에이전트 그래프 노드)의 성능 메트릭."""

    node_name: str
    call_count: int = 0
    total_duration_ms: float = 0.0
    error_count: int = 0

    @property
    def avg_duration_ms(self) -> float:
        """평균 소요 시간 (밀리초)."""
        if self.call_count == 0:
            return 0.0
        return self.total_duration_ms / self.call_count

    @property
    def error_rate(self) -> float:
        """에러 비율 (0.0 ~ 1.0)."""
        if self.call_count == 0:
            return 0.0
        return self.error_count / self.call_count


@dataclass
class AgentMetricsCollector:
    """에이전트 실행 중 메트릭을 수집하는 컬렉터.

    CLI의 _run_agent_turn() 루프에서 이벤트를 관찰하며
    토큰 사용량, 응답 시간, 에러율 등을 집계합니다.

    사용 예시:
        collector = AgentMetricsCollector(agent_name="coding_assistant")
        collector.record_llm_tokens(prompt_tokens=100, completion_tokens=50)
        collector.record_node_start("parse_request")
        collector.record_node_end("parse_request")
        collector.finalize()

        # Langfuse에 스코어로 기록
        collector.push_to_langfuse(trace_id="tr-xxx")
    """

    agent_name: str
    # 토큰 사용량
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    # 타이밍
    start_time: float = field(default_factory=time.monotonic)
    end_time: float = 0.0
    # 에러
    error_count: int = 0
    # 노드별 메트릭
    _node_metrics: dict[str, NodeMetrics] = field(default_factory=dict)
    _node_start_times: dict[str, float] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """총 토큰 사용량."""
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_duration_ms(self) -> float:
        """전체 실행 소요 시간 (밀리초)."""
        end = self.end_time if self.end_time > 0 else time.monotonic()
        return (end - self.start_time) * 1000

    @property
    def error_rate(self) -> float:
        """전체 에러 비율."""
        total_calls = sum(m.call_count for m in self._node_metrics.values())
        if total_calls == 0:
            return 0.0
        return self.error_count / total_calls

    @property
    def node_metrics(self) -> dict[str, NodeMetrics]:
        """노드별 메트릭 복사본 반환."""
        return dict(self._node_metrics)

    def record_llm_tokens(
        self, *, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """LLM 호출의 토큰 사용량을 기록합니다."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    def record_node_start(self, node_name: str) -> None:
        """노드 실행 시작을 기록합니다."""
        self._node_start_times[node_name] = time.monotonic()
        if node_name not in self._node_metrics:
            self._node_metrics[node_name] = NodeMetrics(node_name=node_name)

    def record_node_end(self, node_name: str, *, error: bool = False) -> None:
        """노드 실행 종료를 기록합니다."""
        metrics = self._node_metrics.get(node_name)
        if metrics is None:
            metrics = NodeMetrics(node_name=node_name)
            self._node_metrics[node_name] = metrics

        metrics.call_count += 1

        start = self._node_start_times.pop(node_name, None)
        if start is not None:
            duration_ms = (time.monotonic() - start) * 1000
            metrics.total_duration_ms += duration_ms

        if error:
            metrics.error_count += 1
            self.error_count += 1

    def record_error(self) -> None:
        """전역 에러를 기록합니다 (노드 무관)."""
        self.error_count += 1

    def finalize(self) -> None:
        """메트릭 수집을 완료합니다."""
        self.end_time = time.monotonic()

    def push_to_langfuse(
        self,
        trace_id: str,
        *,
        settings: Settings | None = None,
    ) -> None:
        """수집된 메트릭을 Langfuse 스코어로 기록합니다.

        NUMERIC 스코어로 다음 값들을 기록:
            - agent.total_tokens: 총 토큰 사용량
            - agent.prompt_tokens: 프롬프트 토큰
            - agent.completion_tokens: 완료 토큰
            - agent.duration_ms: 전체 소요 시간
            - agent.error_count: 에러 수

        Args:
            trace_id: Langfuse trace ID
            settings: Settings 인스턴스
        """
        s = settings or get_settings()
        if not enabled(s):
            return

        metrics_to_push = [
            ("agent.total_tokens", float(self.total_tokens)),
            ("agent.prompt_tokens", float(self.total_prompt_tokens)),
            ("agent.completion_tokens", float(self.total_completion_tokens)),
            ("agent.duration_ms", self.total_duration_ms),
            ("agent.error_count", float(self.error_count)),
        ]

        for name, value in metrics_to_push:
            try:
                score_trace(
                    trace_id,
                    name=name,
                    value=value,
                    data_type="NUMERIC",
                    comment=f"{self.agent_name}",
                    settings=s,
                )
            except Exception:
                logger.debug("Langfuse 스코어 기록 실패: %s", name, exc_info=True)

    def to_dict(self) -> dict[str, Any]:
        """메트릭을 딕셔너리로 변환합니다 (로깅/디버깅용)."""
        return {
            "agent_name": self.agent_name,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "duration_ms": round(self.total_duration_ms, 2),
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "nodes": {
                name: {
                    "call_count": m.call_count,
                    "avg_duration_ms": round(m.avg_duration_ms, 2),
                    "error_count": m.error_count,
                    "error_rate": round(m.error_rate, 4),
                }
                for name, m in self._node_metrics.items()
            },
        }


def create_langfuse_handler(
    *,
    settings: Settings | None = None,
) -> Any | None:
    """Langfuse LangChain 콜백 핸들러를 생성합니다.

    Langfuse가 비활성화되었거나 연결에 실패하면 None을 반환합니다.
    호출측은 None 여부를 확인하여 Langfuse 없이도 정상 동작해야 합니다.

    Returns:
        CallbackHandler | None: Langfuse 콜백 핸들러 또는 None
    """
    s = settings or get_settings()
    if not enabled(s):
        logger.debug("Langfuse 비활성화 상태 — 콜백 핸들러를 생성하지 않습니다.")
        return None

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        logger.debug("Langfuse 콜백 핸들러 생성 완료")
        return handler
    except Exception:
        logger.warning(
            "Langfuse 콜백 핸들러 생성 실패 — 관측성 없이 진행합니다.",
            exc_info=True,
        )
        return None


def build_observed_config(
    *,
    handler: Any | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    agent_name: str | None = None,
    extra_tags: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Langfuse 관측성이 적용된 에이전트 실행 config를 구성합니다.

    Langfuse 핸들러가 제공되면 build_langchain_config()를 사용하여
    트레이스 속성을 포함한 config를 생성합니다.
    핸들러가 없으면 기본 config(thread_id만 포함)를 반환합니다.

    Args:
        handler: Langfuse 콜백 핸들러 (None이면 관측성 비적용)
        session_id: 세션 ID (Langfuse 세션 그룹핑)
        user_id: 사용자 ID
        thread_id: LangGraph thread ID (멀티턴 체크포인팅)
        agent_name: 에이전트 이름 (태그에 추가)
        extra_tags: 추가 태그
        settings: Settings 인스턴스

    Returns:
        dict: agent.graph.astream_events()에 전달할 config 딕셔너리
    """
    # 기본 config: LangGraph 체크포인터용 thread_id
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id or "default"}}

    if handler is None:
        return config

    # Langfuse 태그 구성
    tags: list[str] = []
    if agent_name:
        tags.append(f"agent:{agent_name}")
    if extra_tags:
        tags.extend(extra_tags)

    # build_langchain_config()로 Langfuse 메타데이터 생성
    lf_config = build_langchain_config(
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        callbacks=[handler],
        settings=settings,
    )

    # 기존 config에 Langfuse 설정 병합
    if "metadata" in lf_config:
        config["metadata"] = lf_config["metadata"]
    if "callbacks" in lf_config:
        config["callbacks"] = lf_config["callbacks"]

    return config


def safe_flush(*, timeout_ms: int = 3000) -> None:
    """Langfuse 클라이언트의 버퍼를 안전하게 flush합니다.

    서버 연결 실패 등의 예외를 무시하여 에이전트 실행에 영향을 주지 않습니다.

    Args:
        timeout_ms: flush 타임아웃 (밀리초). 기본 3초.
    """
    try:
        if not enabled():
            return
        lf = client()
        lf.flush()
    except Exception:
        logger.debug("Langfuse flush 실패 (무시됨)", exc_info=True)
