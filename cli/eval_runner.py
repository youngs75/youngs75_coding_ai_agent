"""평가 파이프라인 CLI 연동 모듈.

eval_pipeline의 기존 함수를 호출하여 에이전트 평가를 실행하고,
결과를 data/eval_results/에 저장합니다.
DeepEval이나 Langfuse 키가 없어도 graceful하게 동작합니다.

Loop 3 Remediation 연동:
- run_remediation_async(): remediation 에이전트 비동기 실행
- load_last_remediation_report(): 마지막 remediation 리포트 로드
- format_remediation_summary(): 리포트 요약 텍스트 생성
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 프로젝트 루트 기준 데이터 디렉토리
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EVAL_RESULTS_DIR = _PROJECT_ROOT / "data" / "eval_results"
_DEFAULT_RESULTS_FILE = _EVAL_RESULTS_DIR / "eval_results.json"


class EvalResult:
    """평가 실행 결과를 담는 데이터 클래스."""

    def __init__(
        self,
        *,
        success: bool,
        total: int = 0,
        passed: int = 0,
        failed: int = 0,
        results: list[dict[str, Any]] | None = None,
        error_message: str = "",
        timestamp: str = "",
    ):
        self.success = success
        self.total = total
        self.passed = passed
        self.failed = failed
        self.results = results or []
        self.error_message = error_message
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    @property
    def pass_rate(self) -> float:
        """통과율 계산 (0.0 ~ 1.0)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def _run_evaluation_sync(
    *,
    metric_categories: list[str] | None = None,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
) -> EvalResult:
    """동기 방식으로 오프라인 평가를 실행합니다.

    eval_pipeline.loop2_evaluation.batch_evaluator.evaluate_golden_dataset을 호출합니다.
    DeepEval 의존성이 없거나 Golden Dataset이 없을 경우 에러를 반환합니다.
    """
    try:
        from youngs75_a2a.eval_pipeline.loop2_evaluation.batch_evaluator import (
            evaluate_golden_dataset,
        )
    except ImportError as exc:
        return EvalResult(
            success=False,
            error_message=f"평가 파이프라인 의존성을 찾을 수 없습니다: {exc}",
        )

    try:
        results = evaluate_golden_dataset(
            metric_categories=metric_categories or ["rag", "custom"],
            sample_ratio=sample_ratio,
            max_sample_size=max_sample_size,
        )
    except FileNotFoundError as exc:
        return EvalResult(
            success=False,
            error_message=f"Golden Dataset 파일을 찾을 수 없습니다: {exc}",
        )
    except Exception as exc:
        return EvalResult(
            success=False,
            error_message=f"평가 실행 중 오류 발생: {exc}",
        )

    if not results:
        return EvalResult(
            success=False,
            error_message="평가 결과가 없습니다. Golden Dataset이 비어있거나 경로를 확인하세요.",
        )

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    return EvalResult(
        success=True,
        total=total,
        passed=passed,
        failed=total - passed,
        results=results,
    )


async def run_evaluation_async(
    *,
    metric_categories: list[str] | None = None,
    sample_ratio: float | None = None,
    max_sample_size: int | None = None,
) -> EvalResult:
    """비동기 방식으로 평가를 실행합니다.

    평가는 시간이 오래 걸릴 수 있으므로 별도 스레드에서 실행합니다.
    """
    return await asyncio.to_thread(
        _run_evaluation_sync,
        metric_categories=metric_categories,
        sample_ratio=sample_ratio,
        max_sample_size=max_sample_size,
    )


def load_last_eval_results() -> EvalResult:
    """마지막 평가 결과를 파일에서 로드합니다.

    data/eval_results/eval_results.json 파일을 읽어 EvalResult로 변환합니다.
    """
    if not _DEFAULT_RESULTS_FILE.exists():
        return EvalResult(
            success=False,
            error_message="평가 결과 파일이 없습니다. /eval 명령으로 먼저 평가를 실행하세요.",
        )

    try:
        with open(_DEFAULT_RESULTS_FILE, encoding="utf-8") as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return EvalResult(
            success=False,
            error_message=f"평가 결과 파일 읽기 실패: {exc}",
        )

    if not results:
        return EvalResult(
            success=False,
            error_message="평가 결과 파일이 비어있습니다.",
        )

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    # 가장 최근 타임스탬프 사용
    timestamps = [r.get("timestamp", "") for r in results if r.get("timestamp")]
    latest_timestamp = max(timestamps) if timestamps else ""

    return EvalResult(
        success=True,
        total=total,
        passed=passed,
        failed=total - passed,
        results=results,
        timestamp=latest_timestamp,
    )


def format_eval_summary(result: EvalResult) -> str:
    """평가 결과를 사람이 읽기 쉬운 텍스트로 포맷팅합니다."""
    if not result.success:
        return f"평가 실패: {result.error_message}"

    lines = [
        f"평가 결과 요약 (총 {result.total}건)",
        f"  통과: {result.passed}건",
        f"  실패: {result.failed}건",
        f"  통과율: {result.pass_rate:.1%}",
    ]

    if result.timestamp:
        lines.append(f"  평가 시각: {result.timestamp}")

    # 개별 항목 상세 (최대 10건)
    if result.results:
        lines.append("")
        lines.append("상세 결과:")
        display_results = result.results[:10]
        for r in display_results:
            status = "PASS" if r.get("passed", False) else "FAIL"
            item_id = r.get("id", "unknown")
            scores = r.get("scores", {})
            scores_str = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
            # 입력 텍스트 (60자 제한)
            input_text = r.get("input", "")
            if len(input_text) > 60:
                input_text = input_text[:57] + "..."
            lines.append(f"  [{status}] {item_id}: {input_text}")
            if scores_str:
                lines.append(f"         점수: {scores_str}")

        if len(result.results) > 10:
            lines.append(f"  ... 외 {len(result.results) - 10}건")

    return "\n".join(lines)


# ── Remediation (Loop 3) ──


class RemediationResult:
    """Remediation 실행 결과를 담는 데이터 클래스."""

    def __init__(
        self,
        *,
        success: bool,
        report: Any | None = None,
        error_message: str = "",
        report_path: str = "",
    ):
        self.success = success
        self.report = report
        self.error_message = error_message
        self.report_path = report_path


async def run_remediation_async(
    *,
    eval_results_dir: str | None = None,
) -> RemediationResult:
    """비동기 방식으로 Remediation Agent를 실행합니다.

    eval_pipeline의 run_remediation을 호출하고,
    결과 리포트를 파일로 저장합니다.
    """
    try:
        from youngs75_a2a.eval_pipeline.loop3_remediation.recommendation import (
            save_remediation_report,
        )
        from youngs75_a2a.eval_pipeline.loop3_remediation.remediation_agent import (
            run_remediation,
        )
    except ImportError as exc:
        return RemediationResult(
            success=False,
            error_message=f"Remediation 의존성을 찾을 수 없습니다: {exc}",
        )

    try:
        report = await run_remediation(eval_results_dir=eval_results_dir)
        output_path = save_remediation_report(report)
        return RemediationResult(
            success=True,
            report=report,
            report_path=str(output_path),
        )
    except Exception as exc:
        return RemediationResult(
            success=False,
            error_message=f"Remediation 실행 중 오류 발생: {exc}",
        )


def load_last_remediation_report() -> RemediationResult:
    """마지막 Remediation 리포트를 파일에서 로드합니다.

    data/eval_results/remediation_report.json을 읽어 RemediationResult로 변환합니다.
    """
    report_path = _EVAL_RESULTS_DIR / "remediation_report.json"

    if not report_path.exists():
        return RemediationResult(
            success=False,
            error_message="Remediation 리포트가 없습니다. /eval remediate 명령으로 먼저 실행하세요.",
        )

    try:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        from youngs75_a2a.eval_pipeline.loop3_remediation.recommendation import (
            RecommendationReport,
        )

        report = RecommendationReport(**data)
        return RemediationResult(
            success=True,
            report=report,
            report_path=str(report_path),
        )
    except Exception as exc:
        return RemediationResult(
            success=False,
            error_message=f"Remediation 리포트 읽기 실패: {exc}",
        )


def format_remediation_summary(result: RemediationResult) -> str:
    """Remediation 결과를 사람이 읽기 쉬운 텍스트로 포맷팅합니다."""
    if not result.success:
        return f"Remediation 실패: {result.error_message}"

    report = result.report
    if report is None:
        return "Remediation 리포트가 비어있습니다."

    # RecommendationReport의 format_report() 메서드 활용
    if hasattr(report, "format_report"):
        return report.format_report()

    return f"Remediation 완료: {report.summary}"
