"""Planner Agent 상태 스키마.

analyze → explore → create_plan 간 데이터를 전달하는 상태 정의.
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Phase당 파일 수 경고 임계값 (참고용, 강제 분할하지 않음)
FILES_PER_PHASE_WARN_THRESHOLD = 12


class Phase(TypedDict, total=False):
    """구현 계획의 단일 페이즈."""

    id: str  # "phase_1", "phase_2", ...
    title: str  # "데이터베이스 스키마 설계"
    description: str  # 상세 설명
    files: list[str]  # 생성/수정할 파일 경로
    depends_on: list[str]  # 선행 페이즈 ID
    instructions: str  # 코딩 에이전트에 전달할 구체적 지시


class TaskPlan(TypedDict, total=False):
    """구조화된 태스크 실행 계획."""

    complexity: str  # "simple" | "moderate" | "complex"
    summary: str  # 전체 계획 요약
    architecture: str  # 아키텍처 설명
    file_structure: list[str]  # 예상 파일 트리
    phases: list[Phase]  # 순서화된 구현 페이즈
    tech_stack: list[str]  # 사용 기술 스택
    constraints: list[str]  # 제약 조건


def validate_task_plan(plan: TaskPlan) -> TaskPlan:
    """TaskPlan 전체를 검증한다.

    LLM이 결정한 Phase 구조를 존중하되, 극단적인 경우만 경고한다.
    강제 분할은 하지 않는다 — LLM의 자율적 판단을 신뢰한다.
    """
    phases = plan.get("phases", [])
    for phase in phases:
        file_count = len(phase.get("files", []))
        if file_count > FILES_PER_PHASE_WARN_THRESHOLD:
            logger.warning(
                "Phase '%s'의 파일 수(%d)가 %d개를 초과합니다. "
                "LLM 출력 토큰 한도에 주의하세요.",
                phase.get("id", "?"),
                file_count,
                FILES_PER_PHASE_WARN_THRESHOLD,
            )
    return plan


class PlannerState(TypedDict, total=False):
    """Planner Agent 상태."""

    messages: Annotated[list[BaseMessage], add_messages]

    # 사용자 요청 원문
    user_request: str

    # 탐색 컨텍스트 (기존 파일 내용 등)
    explored_context: list[str]

    # 외부 API/서비스 조사 결과 (웹 검색)
    research_context: list[str]

    # 충돌 분석 결과 (HITL 승인 후)
    conflict_resolution: dict | None

    # 최종 계획 출력
    task_plan: TaskPlan

    # 계획 원문 (마크다운)
    plan_text: str
