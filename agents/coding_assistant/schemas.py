"""Coding Assistant 상태 스키마.

parse → execute(ReAct) → verify 간 데이터를 전달하는 상태 정의.
CoALA 메모리 체계 통합: semantic_context로 프로젝트 규칙/컨벤션 주입.
"""

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from youngs75_a2a.core.reducers import override_reducer


class ParseResult(TypedDict, total=False):
    """parse_request 노드 출력."""
    task_type: str          # "generate" | "fix" | "refactor" | "explain"
    language: str           # 감지된 프로그래밍 언어 (기본: "python")
    description: str        # 작업 설명
    target_files: list[str] # 대상 파일 경로
    requirements: list[str] # 세부 요구사항


class VerifyResult(TypedDict, total=False):
    """verify_result 노드 출력."""
    passed: bool
    issues: list[str]       # 발견된 문제 목록
    suggestions: list[str]  # 개선 제안


class CodingState(TypedDict, total=False):
    """Coding Assistant 에이전트 상태."""
    messages: Annotated[list[BaseMessage], add_messages]

    # CoALA Semantic Memory — 프로젝트 규칙/컨벤션 (덮어쓰기 가능)
    semantic_context: Annotated[list[str], override_reducer]

    # Skills 컨텍스트 — 활성 스킬 메타데이터 (L1)
    skill_context: Annotated[list[str], override_reducer]

    # Episodic Memory — 이전 실행 이력
    episodic_log: Annotated[list[str], override_reducer]

    # parse_request 출력
    parse_result: ParseResult

    # execute_code 출력 (ReAct 루프에서 도구 사용 포함)
    generated_code: str         # 최종 생성/수정된 코드
    execution_log: list[str]    # 실행 과정 로그
    project_context: list[str]  # 프로젝트 파일 읽기 결과 (JIT 원본 참조)

    # verify_result 출력
    verify_result: VerifyResult

    # 반복 제어
    iteration: int              # 현재 반복 횟수
    max_iterations: int         # 최대 반복 횟수
