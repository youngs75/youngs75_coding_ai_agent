"""Planner Agent 상태 스키마.

analyze → explore → create_plan 간 데이터를 전달하는 상태 정의.
Phase별 파일 수 제한 등 검증 로직 포함.
"""

from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Phase당 최대 파일 수 하드 리밋
MAX_FILES_PER_PHASE = 8


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


def validate_phase_file_limit(phases: list[Phase]) -> list[Phase]:
    """Phase별 파일 수 하드 리밋(MAX_FILES_PER_PHASE)을 검증한다.

    초과하는 phase가 있으면 자동으로 분할하여 반환한다.
    원본 리스트를 변경하지 않고 새 리스트를 반환한다.
    """
    result: list[Phase] = []
    phase_counter = 0

    for phase in phases:
        files = phase.get("files", [])
        if len(files) <= MAX_FILES_PER_PHASE:
            phase_counter += 1
            # id를 재번호 매김
            new_phase = dict(phase)
            new_phase["id"] = f"phase_{phase_counter}"
            result.append(new_phase)  # type: ignore[arg-type]
        else:
            # 초과 — 자동 분할
            logger.warning(
                "Phase '%s'의 파일 수(%d)가 하드 리밋(%d)을 초과하여 자동 분할합니다.",
                phase.get("id", "?"),
                len(files),
                MAX_FILES_PER_PHASE,
            )
            # 파일을 MAX_FILES_PER_PHASE개씩 나눈다
            for i in range(0, len(files), MAX_FILES_PER_PHASE):
                phase_counter += 1
                chunk = files[i : i + MAX_FILES_PER_PHASE]
                sub_phase: Phase = {
                    "id": f"phase_{phase_counter}",
                    "title": phase.get("title", "")
                    + (f" (파트 {i // MAX_FILES_PER_PHASE + 1})" if i > 0 else ""),
                    "description": phase.get("description", ""),
                    "files": chunk,
                    "depends_on": (
                        phase.get("depends_on", [])
                        if i == 0
                        else [f"phase_{phase_counter - 1}"]
                    ),
                    "instructions": phase.get("instructions", ""),
                }
                result.append(sub_phase)

    return result


def validate_task_plan(plan: TaskPlan) -> TaskPlan:
    """TaskPlan 전체를 검증하고 필요 시 보정한다.

    현재 검증 항목:
    - Phase별 파일 수 하드 리밋 (초과 시 자동 분할)
    """
    phases = plan.get("phases", [])
    validated_phases = validate_phase_file_limit(phases)

    if len(validated_phases) != len(phases):
        logger.info(
            "Phase 자동 분할: %d개 → %d개",
            len(phases),
            len(validated_phases),
        )

    new_plan = dict(plan)
    new_plan["phases"] = validated_phases
    return new_plan  # type: ignore[return-value]


class PlannerState(TypedDict, total=False):
    """Planner Agent 상태."""

    messages: Annotated[list[BaseMessage], add_messages]

    # 사용자 요청 원문
    user_request: str

    # 탐색 컨텍스트 (기존 파일 내용 등)
    explored_context: list[str]

    # 외부 API/서비스 조사 결과 (웹 검색)
    research_context: list[str]

    # 최종 계획 출력
    task_plan: TaskPlan

    # 계획 원문 (마크다운)
    plan_text: str
