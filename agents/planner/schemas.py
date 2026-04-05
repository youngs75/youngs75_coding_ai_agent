"""Planner Agent 상태 스키마.

analyze → explore → create_plan 간 데이터를 전달하는 상태 정의.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


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


class PlannerState(TypedDict, total=False):
    """Planner Agent 상태."""

    messages: Annotated[list[BaseMessage], add_messages]

    # 사용자 요청 원문
    user_request: str

    # 탐색 컨텍스트 (기존 파일 내용 등)
    explored_context: list[str]

    # 최종 계획 출력
    task_plan: TaskPlan

    # 계획 원문 (마크다운)
    plan_text: str
