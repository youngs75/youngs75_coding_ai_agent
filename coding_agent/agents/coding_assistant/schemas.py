"""Coding Assistant 상태 스키마.

parse → execute(ReAct) → verify 간 데이터를 전달하는 상태 정의.
CoALA 메모리 체계 통합: semantic_context로 프로젝트 규칙/컨벤션 주입.

Structured Output용 Pydantic 모델도 함께 정의:
- ParseResultModel: _parse_request 노드의 LLM 응답 강제
- VerifyResultModel: _verify_result 노드의 LLM 응답 강제
- FileManifest: _plan_file_manifest 노드의 파일 간 인터페이스 선언
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from coding_agent.core.reducers import override_reducer


# ── TypedDict (그래프 상태 전달용) ──


class ParseResult(TypedDict, total=False):
    """parse_request 노드 출력."""

    task_type: str  # "generate" | "fix" | "refactor" | "explain" | "analyze"
    language: str  # 감지된 프로그래밍 언어 (기본: "python")
    description: str  # 작업 설명
    target_files: list[str]  # 대상 파일 경로
    requirements: list[str]  # 세부 요구사항


class VerifyResult(TypedDict, total=False):
    """verify_result 노드 출력."""

    passed: bool
    issues: list[str]  # 발견된 문제 목록
    suggestions: list[str]  # 개선 제안


# ── Pydantic BaseModel (Structured Output 강제용) ──


class ParseResultModel(BaseModel):
    """LLM 파싱 결과 — with_structured_output으로 강제."""

    task_type: Literal[
        "generate", "fix", "refactor", "analyze", "scaffold", "explain"
    ] = "generate"
    language: str = "python"
    framework: str = ""
    description: str = ""
    target_files: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)


class VerifyResultModel(BaseModel):
    """코드 검증 결과 — with_structured_output으로 강제."""

    passed: bool = True
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class FunctionContract(BaseModel):
    """파일 간 공유되는 함수의 계약."""

    name: str = Field(description="함수명 (예: create_app)")
    defined_in: str = Field(description="정의 파일 경로 (예: backend/__init__.py)")
    params: list[str] = Field(
        default_factory=list,
        description="매개변수 목록 (예: [\"config_name='development'\"])",
    )
    called_from: list[str] = Field(
        default_factory=list,
        description="호출 파일 경로 목록 (예: [\"tests/conftest.py\"])",
    )


class SharedObject(BaseModel):
    """파일 간 공유되는 객체."""

    name: str = Field(description="객체명 (예: db)")
    defined_in: str = Field(description="정의 파일 경로 (예: backend/extensions.py)")
    imported_by: list[str] = Field(
        default_factory=list,
        description="import하는 파일 목록",
    )


class FileManifest(BaseModel):
    """코드 생성 전 파일 간 인터페이스 선언 — with_structured_output으로 강제."""

    files: list[str] = Field(description="생성할 파일 경로 목록")
    contracts: list[FunctionContract] = Field(
        default_factory=list,
        description="파일 간 공유 함수 계약 목록",
    )
    shared_objects: list[SharedObject] = Field(
        default_factory=list,
        description="파일 간 공유 객체 목록",
    )
    entry_point: str = Field(default="", description="앱 진입점 파일 경로")


class CodingState(TypedDict, total=False):
    """Coding Assistant 에이전트 상태."""

    messages: Annotated[list[BaseMessage], add_messages]

    # CoALA Semantic Memory — 프로젝트 규칙/컨벤션 (덮어쓰기 가능)
    semantic_context: Annotated[list[str], override_reducer]

    # Skills 컨텍스트 — 활성 스킬 메타데이터 (L1)
    skill_context: Annotated[list[str], override_reducer]

    # Episodic Memory — 이전 실행 이력
    episodic_log: Annotated[list[str], override_reducer]

    # Procedural Memory — 학습된 스킬 패턴 (Voyager식 누적)
    procedural_skills: Annotated[list[str], override_reducer]

    # User Profile — 사용자 선호/습관/반복 피드백
    user_profile_context: Annotated[list[str], override_reducer]

    # Domain Knowledge — 비즈니스 용어/업무 규칙/API 계약
    domain_knowledge_context: Annotated[list[str], override_reducer]

    # parse_request 출력
    parse_result: ParseResult

    # execute_code 출력 (ReAct 루프에서 도구 사용 포함)
    generated_code: str  # 최종 생성/수정된 코드
    execution_log: list[str]  # 실행 과정 로그
    project_context: list[str]  # 프로젝트 파일 읽기 결과 (JIT 원본 참조)

    # verify_result 출력 (LLM 기반 검증)
    verify_result: VerifyResult

    # run_tests 출력 (실행 기반 검증)
    test_output: str  # 테스트 실행 stdout/stderr
    test_passed: bool  # 테스트 통과 여부

    # 반복 제어
    iteration: int  # 현재 반복 횟수
    max_iterations: int  # 최대 반복 횟수
    tool_call_count: int  # ReAct 루프 내 도구 호출 누적 횟수

    # 다층 안전장치 — 루프 탈출 사유
    # "" (정상) | "stall_detected" | "budget_exceeded" | "turn_limit"
    exit_reason: str

    # apply_code 출력 — 디스크에 저장된 파일 경로 목록 (재시도 시 누적)
    written_files: Annotated[list[str], override_reducer]

    # Planner가 지정한 Phase별 생성 예정 파일 경로 목록
    planned_files: list[str]

    # FileManifest — 코드 생성 전 파일 간 인터페이스 선언 (Structured Output)
    file_manifest: dict

    # 환경 승인 HITL — venv/의존성 설치 전 사용자 승인 여부
    env_approved: bool

    # 반복 감지용 — 이전 생성 코드 및 테스트 에러 캐시
    _prev_generated_code: str
    _prev_test_output: str

    # 의존성 변경 플래그 — inject_test_failure에서 requirements.txt 수정 시 True
    _deps_changed: bool
