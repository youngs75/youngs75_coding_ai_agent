"""동적 SubAgent 스키마.

Puppeteer 논문 원칙: "효율성은 제거에서 온다"
— 모든 에이전트를 항상 활성화하지 않고, 태스크에 필요한 것만 동적 선택한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SubAgentStatus(str, Enum):
    """서브에이전트 수명주기 상태.

    8단계 수명주기(CREATED → ASSIGNED → RUNNING → COMPLETED 등)와
    하위호환 별칭(AVAILABLE, BUSY, DISABLED)을 정의한다.
    """

    # ── 8단계 수명주기 ──
    CREATED = "created"        # 인스턴스 생성됨
    ASSIGNED = "assigned"      # 작업 할당됨
    RUNNING = "running"        # 추론/도구 사용 수행 중
    BLOCKED = "blocked"        # 입력 부족, 도구 실패, 의존 작업 대기
    COMPLETED = "completed"    # 정상 완료
    FAILED = "failed"          # 재시도 후에도 실패
    CANCELLED = "cancelled"    # 상위 오케스트레이터가 중단
    DESTROYED = "destroyed"    # 상태 정리 후 수명주기 종료

    # ── 하위호환 별칭 (deprecated) ──
    AVAILABLE = "available"    # → CREATED와 동일 의미로 취급
    BUSY = "busy"              # → RUNNING과 동일 의미로 취급
    DISABLED = "disabled"      # → CANCELLED과 동일 의미로 취급


# ── 유효한 상태 전이 정의 ──
VALID_TRANSITIONS: dict[SubAgentStatus, set[SubAgentStatus]] = {
    SubAgentStatus.CREATED: {
        SubAgentStatus.ASSIGNED,
        SubAgentStatus.CANCELLED,
        SubAgentStatus.DESTROYED,
    },
    SubAgentStatus.ASSIGNED: {
        SubAgentStatus.RUNNING,
        SubAgentStatus.BLOCKED,
        SubAgentStatus.CANCELLED,
    },
    SubAgentStatus.RUNNING: {
        SubAgentStatus.COMPLETED,
        SubAgentStatus.FAILED,
        SubAgentStatus.BLOCKED,
        SubAgentStatus.CANCELLED,
    },
    SubAgentStatus.BLOCKED: {
        SubAgentStatus.RUNNING,
        SubAgentStatus.FAILED,
        SubAgentStatus.CANCELLED,
    },
    SubAgentStatus.COMPLETED: {SubAgentStatus.DESTROYED},
    SubAgentStatus.FAILED: {
        SubAgentStatus.ASSIGNED,   # 재시도 가능
        SubAgentStatus.DESTROYED,
    },
    SubAgentStatus.CANCELLED: {SubAgentStatus.DESTROYED},
    SubAgentStatus.DESTROYED: set(),  # 종료 상태
}


class SubAgentSpec(BaseModel):
    """서브에이전트 사양 — 능력, 비용, 상태를 기술한다.

    Args:
        name: 에이전트 고유 이름.
        description: 에이전트 설명.
        capabilities: 지원 능력 목록.
        endpoint: A2A 엔드포인트 URL. None이면 로컬 에이전트.
        model_tier: 사용할 모델 티어 (기본: "default").
        cost_weight: 비용 가중치 (높을수록 비쌈).
        status: 현재 상태.
        metadata: 추가 메타데이터.
    """

    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    endpoint: str | None = None  # A2A 엔드포인트 또는 None(로컬)
    model_tier: str = "default"
    cost_weight: float = 1.0  # 비용 가중치 (높을수록 비쌈)
    status: SubAgentStatus = SubAgentStatus.CREATED
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentInstance(BaseModel):
    """런타임 서브에이전트 인스턴스 — 수명주기 추적.

    SubAgentSpec 기반으로 생성되며, 재시도 횟수, 에러 메시지,
    결과 요약 등 런타임 상태를 추적한다.

    Args:
        agent_id: 인스턴스 고유 ID (UUID hex).
        spec_name: 참조하는 SubAgentSpec.name.
        role: 역할/전문성 설명.
        task_summary: 할당된 작업 요약.
        parent_id: 부모 에이전트/오케스트레이터 ID.
        state: 현재 수명주기 상태.
        retry_count: 현재 재시도 횟수.
        max_retries: 최대 재시도 횟수.
        created_at: 생성 시각 (UTC).
        updated_at: 마지막 갱신 시각 (UTC).
        metadata: 추가 메타데이터.
        error_message: 실패 시 에러 메시지.
        result_summary: 완료 시 결과 요약.
    """

    agent_id: str = Field(default_factory=lambda: uuid4().hex)
    spec_name: str                    # SubAgentSpec.name 참조
    role: str = ""                    # 역할/전문성 설명
    task_summary: str = ""            # 할당된 작업 요약
    parent_id: str | None = None      # 부모 에이전트/오케스트레이터 ID
    state: SubAgentStatus = SubAgentStatus.CREATED
    retry_count: int = 0              # 현재 재시도 횟수
    max_retries: int = 2              # 최대 재시도 횟수
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None  # 실패 시 에러 메시지
    result_summary: str | None = None  # 완료 시 결과 요약


class SubAgentEvent(BaseModel):
    """서브에이전트 상태 전이 이벤트 (로깅용).

    Args:
        agent_id: 에이전트 인스턴스 ID.
        from_state: 전이 전 상태.
        to_state: 전이 후 상태.
        timestamp: 전이 시각 (UTC).
        reason: 전이 사유.
        metadata: 추가 메타데이터.
    """

    agent_id: str
    from_state: SubAgentStatus
    to_state: SubAgentStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentUsageRecord(BaseModel):
    """서브에이전트 사용 기록.

    Args:
        agent_name: 에이전트 이름.
        task_type: 태스크 유형.
        success: 성공 여부.
        duration_ms: 실행 소요 시간(밀리초).
        failure_reason: 실패 원인 (성공 시 None).
        timestamp: 기록 시각 (UTC).
    """

    agent_name: str
    task_type: str
    success: bool
    duration_ms: float = 0.0
    failure_reason: str | None = None  # 실패 원인 (성공 시 None)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SelectionResult(BaseModel):
    """에이전트 선택 결과.

    Args:
        agent: 선택된 에이전트 사양.
        score: R = r(quality) - λ·C(cost) 점수.
        reason: 선택 근거 설명.
    """

    agent: SubAgentSpec
    score: float
    reason: str


# ── 프로세스 기반 SubAgent 관련 스키마 ──


class ResourceUsage(BaseModel):
    """프로세스 자원 사용량.

    Args:
        pid: 프로세스 ID.
        agent_id: 에이전트 인스턴스 ID.
        start_time: 시작 시각 (epoch seconds).
        end_time: 종료 시각 (epoch seconds). 미종료 시 None.
        peak_memory_mb: 최대 메모리 사용량(MB).
        cpu_time_s: CPU 시간(초).
        exit_code: 프로세스 종료 코드. 미종료 시 None.
    """

    pid: int
    agent_id: str
    start_time: float
    end_time: float | None = None
    peak_memory_mb: float = 0.0
    cpu_time_s: float = 0.0
    exit_code: int | None = None


class SubAgentResult(BaseModel):
    """SubAgent 프로세스 실행 결과 (stdout JSON 프로토콜).

    Args:
        status: 실행 상태 ("completed" 또는 "failed").
        result: 실행 결과 텍스트.
        written_files: 생성/수정된 파일 경로 리스트.
        duration_s: 실행 소요 시간(초).
        token_usage: 토큰 사용량 (input_tokens, output_tokens 등).
        error: 에러 메시지 (성공 시 None).
        test_passed: 테스트 통과 여부.
        exit_reason: 종료 사유 (budget_exceeded, llm_error 등).
    """

    status: str = "completed"  # "completed" | "failed"
    result: str | None = None
    written_files: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
    test_passed: bool = True       # 테스트 통과 여부
    exit_reason: str = ""          # 종료 사유 (budget_exceeded, llm_error 등)

    @property
    def success(self) -> bool:
        return self.status == "completed" and self.error is None
