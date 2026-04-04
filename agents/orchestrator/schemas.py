"""Orchestrator 에이전트 상태 스키마."""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """Orchestrator 상태.

    messages: 사용자와의 대화 이력
    selected_agent: 라우팅 결정된 에이전트 이름
    agent_response: 하위 에이전트의 응답
    """

    messages: Annotated[list[BaseMessage], add_messages]
    selected_agent: Optional[str]
    agent_response: Optional[str]


# ── Coordinator Mode 스키마 ──────────────────────────────────


class SubTask(TypedDict):
    """분해된 서브태스크.

    코디네이터가 복합 작업을 분해할 때 생성되는 단위.
    dependencies를 통해 DAG를 구성하여 실행 순서를 제어한다.
    """

    id: str
    description: str
    agent_type: str  # 할당할 에이전트 타입
    dependencies: list[str]  # 선행 서브태스크 ID (DAG 구성)
    priority: int
    timeout_s: float


class WorkerResult(TypedDict):
    """워커 실행 결과.

    개별 서브태스크를 실행한 워커의 결과를 담는다.
    """

    subtask_id: str
    agent_name: str
    status: str  # "success" | "failed" | "timeout" | "cancelled"
    output: str
    duration_s: float
    error: str | None


class CoordinatorResult(TypedDict):
    """코디네이터 전체 결과.

    decompose → execute_parallel → synthesize 파이프라인의 최종 결과.
    """

    synthesized_response: str
    worker_results: list[WorkerResult]
    total_duration_s: float
    parallel_efficiency: float  # 실제시간 / 순차시간 비율 (낮을수록 효율적)
