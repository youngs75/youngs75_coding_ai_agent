"""Verification Agent 상태 스키마."""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CheckResult(TypedDict, total=False):
    """개별 검증 체크 결과."""

    check_type: str  # "lint" | "test" | "llm_review"
    passed: bool
    output: str  # 도구 실행 결과 또는 LLM 리뷰 내용
    issues: list[str]


class VerificationResult(TypedDict, total=False):
    """최종 검증 결과."""

    passed: bool
    checks: list[CheckResult]
    issues: list[str]
    suggestions: list[str]
    summary: str


class VerificationState(TypedDict, total=False):
    """Verification Agent 상태."""

    messages: Annotated[list[BaseMessage], add_messages]

    # 검증 대상 입력
    code: str  # 검증할 코드
    written_files: list[str]  # 디스크에 저장된 파일 경로
    language: str  # 프로그래밍 언어
    requirements: str  # 원래 요구사항

    # 각 체크 결과
    lint_result: CheckResult
    test_result: CheckResult
    review_result: CheckResult

    # 최종 결과
    verification_result: VerificationResult
