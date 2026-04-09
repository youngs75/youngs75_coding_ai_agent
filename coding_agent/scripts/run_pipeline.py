#!/usr/bin/env python3
"""Day3 AI Agent Ops 통합 파이프라인 실행 스크립트.

8개 단계를 하나의 CLI로 통합하여 순서대로 실행할 수 있습니다.
각 단계는 독립적으로 실행하거나, 범위를 지정하여 연속 실행할 수 있습니다.

파이프라인 단계:
    Step 1: Synthetic Dataset 생성 (DeepEval Synthesizer)
    Step 2: Human Review용 CSV 내보내기
    Step 3: 리뷰된 CSV 가져오기 + LLM 피드백 보강
    Step 4: Golden Dataset 확정 (Loop 1 오케스트레이터)
    Step 5: DeepEval 오프라인 평가 실행
    Step 6: Langfuse 모니터링 스코어 동기화/실패 추출
    Step 7: Remediation Agent 실행 (개선안 추천)
    Step 8: Evaluation Prompt 자동 최적화

실행 예시:
    # 전체 파이프라인 실행 (Human Review 건너뛰기)
    python scripts/run_pipeline.py --all --skip-review

    # 특정 단계만 실행
    python scripts/run_pipeline.py --step 5

    # 범위 지정 실행 (Step 4부터 8까지)
    python scripts/run_pipeline.py --from 4 --to 8

    # 에러 발생 시에도 다음 단계로 계속 진행
    python scripts/run_pipeline.py --all --continue-on-error

의존성 관계:
    Step 1 → Step 2 → Step 3 → Step 4  (Loop 1: Dataset)
    Step 4 → Step 5                     (Loop 2: Offline Evaluation)
    Step 4 → Step 6                     (Loop 2: Monitoring Snapshot, Langfuse 필요)
    Step 5 → Step 7                     (Loop 3: Remediation)
    Step 6 → Step 8                     (Loop 2: Prompt Optimization with Langfuse 실패샘플 힌트)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path


# ── Step 함수 정의 ────────────────────────────────────────────


def _parse_metric_threshold_pairs(items: list[str] | None) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(
                f"Invalid metric threshold format: {raw} (expected key=value)"
            )
        key, value = raw.split("=", 1)
        metric_key = key.strip()
        if not metric_key:
            raise ValueError(f"Invalid metric threshold key: {raw}")
        parsed[metric_key] = float(value.strip())
    return parsed


def step1_generate_synthetic(
    *,
    num_goldens: int = 10,
    max_contexts: int = 3,
) -> list[dict]:
    """Step 1: Synthetic Dataset을 생성합니다.

    DeepEval Synthesizer를 사용하여 data/corpus/ 문서를 기반으로
    합성 질문-답변 쌍을 생성합니다.

    Args:
        num_goldens: 생성할 항목 수 (기본: 10)
        max_contexts: 항목당 최대 컨텍스트 수 (기본: 3)

    Returns:
        생성된 합성 데이터 딕셔너리 리스트
    """
    from coding_agent.eval_pipeline.loop1_dataset.synthesizer import (
        generate_synthetic_dataset,
    )

    items = generate_synthetic_dataset(
        num_goldens=num_goldens,
        max_goldens_per_context=max_contexts,
    )
    print(f"  생성된 합성 데이터: {len(items)}건")
    return items


def step2_export_csv() -> Path:
    """Step 2: Synthetic Dataset을 Human Review용 CSV로 내보냅니다.

    data/synthetic/ 디렉토리의 JSON을 CSV 형식으로 변환하여
    Excel/Sheets에서 리뷰할 수 있도록 합니다.

    Returns:
        생성된 CSV 파일 경로
    """
    from coding_agent.eval_pipeline.loop1_dataset.csv_exporter import (
        export_to_review_csv,
    )
    from coding_agent.eval_pipeline.settings import get_settings

    settings = get_settings()
    synthetic_dir = settings.data_dir / "synthetic"
    json_files = sorted(synthetic_dir.glob("*.json"))

    if not json_files:
        print(
            "  경고: data/synthetic/ 에 JSON 파일이 없습니다. Step 1을 먼저 실행하세요."
        )
        return Path()

    synthetic_path = json_files[-1]
    review_path = settings.data_dir / "review" / "review_dataset.csv"
    csv_path = export_to_review_csv(synthetic_path, review_path)
    print(f"  CSV 내보내기 완료: {csv_path}")
    return csv_path


def step3_import_reviewed(
    *,
    csv_path: str | None = None,
    augment: bool = True,
) -> list[dict]:
    """Step 3: 리뷰된 CSV를 가져오고 LLM 피드백 보강을 수행합니다.

    Human Reviewer가 approved/feedback 칼럼을 작성한 CSV를 읽고,
    피드백이 있는 항목에 대해 LLM으로 expected_output을 개선합니다.

    Args:
        csv_path: 리뷰된 CSV 경로 (기본: data/review/ 내 최근 파일)
        augment: LLM 피드백 보강 수행 여부 (기본: True)

    Returns:
        가져온 (+ 보강된) 데이터 딕셔너리 리스트
    """
    from coding_agent.eval_pipeline.loop1_dataset.csv_importer import (
        import_reviewed_csv,
    )
    from coding_agent.eval_pipeline.loop1_dataset.feedback_augmenter import (
        augment_with_feedback,
    )
    from coding_agent.eval_pipeline.settings import get_settings

    if csv_path:
        review_path = Path(csv_path)
    else:
        settings = get_settings()
        review_dir = settings.data_dir / "review"
        csv_files = sorted(review_dir.glob("*.csv"))
        if not csv_files:
            print(
                "  경고: data/review/ 에 CSV 파일이 없습니다. Step 2를 먼저 실행하세요."
            )
            return []
        review_path = csv_files[-1]

    settings = get_settings()
    golden_path = settings.data_dir / "golden" / "golden_dataset.json"
    items = import_reviewed_csv(
        review_path,
        golden_path,
        only_approved=True,
    )
    print(f"  CSV 가져오기 완료: {len(items)}건")

    if augment:
        # 피드백이 있는 항목만 LLM 보강
        feedback_items = [it for it in items if it.get("feedback")]
        if feedback_items:
            items = augment_with_feedback(items)
            print(f"  LLM 피드백 보강 완료: {len(feedback_items)}건 보강됨")
        else:
            print("  피드백이 있는 항목이 없어 보강을 건너뜁니다.")

    return items


def step4_build_golden(
    *,
    num_goldens: int = 10,
    skip_review: bool = False,
) -> list[dict]:
    """Step 4: Golden Dataset을 확정합니다 (Loop 1 오케스트레이터).

    Synthetic 생성 → (선택) Human Review → (선택) LLM 보강 → Golden 확정
    전체 Loop 1 워크플로우를 한 번에 실행합니다.

    Args:
        num_goldens: 생성할 Golden 항목 수 (기본: 10)
        skip_review: Human Review 단계 건너뛰기 (기본: False)

    Returns:
        확정된 Golden Dataset 딕셔너리 리스트
    """
    from coding_agent.eval_pipeline.loop1_dataset.golden_builder import (
        build_golden_dataset,
    )

    golden_items = build_golden_dataset(
        num_goldens=num_goldens,
        skip_review=skip_review,
    )
    print(f"  Golden Dataset 확정: {len(golden_items)}건")
    return golden_items


def step5_run_evaluation(
    *,
    categories: list[str] | None = None,
    sample_ratio: float | None = None,
    sample_size: int | None = None,
    sample_seed: int = 42,
    stratify_by: list[str] | None = None,
) -> list[dict]:
    """Step 5: DeepEval 오프라인 평가를 실행합니다.

    Golden Dataset에 대해 지정된 메트릭 카테고리로 평가를 수행합니다.
    결과는 data/eval_results/eval_results.json에 저장됩니다.

    Args:
        categories: 메트릭 카테고리 목록 (기본: ["rag", "custom"])
        sample_ratio: Golden Dataset 샘플링 비율 (0.0~1.0)
        sample_size: Golden Dataset 샘플 최대 건수 상한
        sample_seed: deterministic 샘플링 시드
        stratify_by: Golden Dataset 층화 샘플링 기준 필드 목록

    Returns:
        평가 결과 딕셔너리 리스트
    """
    from coding_agent.eval_pipeline.loop2_evaluation.batch_evaluator import (
        evaluate_golden_dataset,
    )

    cats = categories or ["rag", "custom"]
    results = evaluate_golden_dataset(
        metric_categories=cats,
        sample_ratio=sample_ratio,
        max_sample_size=sample_size,
        sample_seed=sample_seed,
        stratify_by=stratify_by,
    )

    passed = sum(1 for r in results if r["passed"])
    print(f"  평가 완료: {passed}/{len(results)} 통과")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        print(f"    [{status}] {r.get('id', 'N/A')}: {scores_str}")

    return results


def step6_batch_langfuse(
    *,
    tags: list[str] | None = None,
    from_hours_ago: int = 24,
    limit: int = 200,
    sample_ratio: float | None = None,
    sample_size: int | None = None,
    sample_seed: int = 42,
    score_prefix: str = "eval",
    default_threshold: float = 0.7,
    metric_thresholds: dict[str, float] | None = None,
    include_missing_scores: bool = False,
) -> dict:
    """Step 6: Langfuse 모니터링 스코어를 샘플링하여 실패 샘플을 추출합니다.

    이 단계는 DeepEval을 호출하지 않고 Langfuse에 이미 기록된 스코어를 집계합니다.
    (샘플링 → 실패 추출 → 로컬 아티팩트 저장)

    Args:
        tags: Langfuse trace 필터 태그 (예: ["prod"])
        from_hours_ago: 몇 시간 전부터 가져올지 (기본: 24)
        limit: 조회할 최대 trace 수 (기본: 200)
        sample_ratio: 조회 trace 중 평가 비율 (0.0~1.0)
        sample_size: 샘플 최대 건수 상한
        sample_seed: deterministic 샘플링 시드
        score_prefix: 모니터링 스코어 prefix (예: "eval")
        default_threshold: 기본 실패 임계값
        metric_thresholds: 메트릭별 임계값 override
        include_missing_scores: True면 prefix score 없는 trace도 결과에 포함

    Returns:
        모니터링 스냅샷 딕셔너리
    """
    from coding_agent.eval_pipeline.loop2_evaluation.batch_evaluator import (
        monitor_langfuse_scores,
    )

    snapshot = monitor_langfuse_scores(
        tags=tags,
        from_hours_ago=from_hours_ago,
        limit=limit,
        sample_ratio=sample_ratio,
        max_sample_size=sample_size,
        sample_seed=sample_seed,
        score_prefix=score_prefix,
        default_threshold=default_threshold,
        metric_thresholds=metric_thresholds,
        require_score_prefix=not include_missing_scores,
    )
    summary = snapshot["summary"]
    print(
        "  Langfuse 모니터링 완료: "
        f"sampled={summary['total_sampled']}, "
        f"collected={summary['total_collected']}, "
        f"failed={summary['traces_failed']}"
    )
    return snapshot


async def step7_run_remediation() -> object:
    """Step 7: Remediation Agent를 실행하여 개선안을 추천합니다.

    DeepAgents 기반 에이전트가 평가 결과를 분석하고,
    실패 패턴을 분류하여 구체적인 개선 추천 리포트를 생성합니다.

    Returns:
        RecommendationReport Pydantic 모델 인스턴스
    """
    from coding_agent.eval_pipeline.loop3_remediation.remediation_agent import (
        run_remediation,
    )

    report = await run_remediation()
    print(f"  요약: {report.summary[:200]}")
    print(f"  추천 건수: {len(report.recommendations)}")

    for rec in report.recommendations[:5]:
        print(f"    - [{rec.priority}] {rec.title}")

    if report.next_steps:
        print("  다음 단계:")
        for step in report.next_steps[:3]:
            print(f"    → {step}")

    return report


def step8_optimize_prompts(
    *,
    iterations: int = 2,
    max_cases: int | None = None,
    apply: bool = False,
    failure_hints_path: str | None = None,
    failure_hints_max: int = 6,
) -> dict:
    """Step 8: Evaluation 프롬프트 자동 최적화를 실행합니다."""
    from coding_agent.eval_pipeline.loop2_evaluation.prompt_optimizer import (
        apply_best_prompts_to_file,
        load_langfuse_failure_hints,
        optimize_all_prompts,
        save_optimization_artifacts,
    )
    from coding_agent.eval_pipeline.settings import get_settings

    settings = get_settings()
    default_hints_path = (
        settings.data_dir / "eval_results" / "langfuse_failed_samples.json"
    )
    resolved_hints_path = (
        Path(failure_hints_path) if failure_hints_path else default_hints_path
    )
    failure_hints = None
    if resolved_hints_path.exists():
        failure_hints = load_langfuse_failure_hints(
            resolved_hints_path,
            max_per_metric=max(1, failure_hints_max),
        )

    artifacts = optimize_all_prompts(
        iterations=iterations,
        max_cases=max_cases,
        langfuse_failure_hints=failure_hints,
    )
    output_paths = save_optimization_artifacts(
        artifacts,
        report_dir=settings.data_dir / "prompt_optimization",
    )

    print(f"  최적화 리포트 저장: {output_paths['report_path']}")
    print(f"  최적 프롬프트 저장: {output_paths['best_prompts_path']}")
    for metric_name, result in artifacts["metric_results"].items():
        baseline_fit = result["baseline_evaluation"]["fit_score"]
        best_fit = result["best_evaluation"]["fit_score"]
        print(f"    - {metric_name}: baseline={baseline_fit:.2f}, best={best_fit:.2f}")
    if failure_hints is not None:
        hint_counts = {key: len(value) for key, value in failure_hints.items()}
        print(f"  Langfuse 실패 힌트 반영: {hint_counts} ({resolved_hints_path})")
    else:
        print(f"  Langfuse 실패 힌트 파일 없음: {resolved_hints_path}")

    if apply:
        prompts_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "loop2_evaluation"
            / "prompts.py"
        )
        apply_best_prompts_to_file(
            artifacts["best_prompts"],
            prompts_path=prompts_path,
        )
        print(f"  prompts.py 업데이트 완료: {prompts_path}")

    return {
        "artifacts": artifacts,
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "applied": apply,
        "failure_hints_path": str(resolved_hints_path),
        "failure_hints_used": failure_hints is not None,
    }


# ── 단계 레지스트리 ──────────────────────────────────────────

STEPS: dict[int, dict] = {
    1: {
        "name": "Synthetic Dataset 생성",
        "loop": "Loop 1 - Dataset",
        "func": step1_generate_synthetic,
        "is_async": False,
    },
    2: {
        "name": "Human Review CSV 내보내기",
        "loop": "Loop 1 - Dataset",
        "func": step2_export_csv,
        "is_async": False,
    },
    3: {
        "name": "리뷰된 CSV 가져오기 + LLM 보강",
        "loop": "Loop 1 - Dataset",
        "func": step3_import_reviewed,
        "is_async": False,
    },
    4: {
        "name": "Golden Dataset 확정",
        "loop": "Loop 1 - Dataset",
        "func": step4_build_golden,
        "is_async": False,
    },
    5: {
        "name": "DeepEval 오프라인 평가",
        "loop": "Loop 2 - Evaluation",
        "func": step5_run_evaluation,
        "is_async": False,
    },
    6: {
        "name": "Langfuse 모니터링 스코어 동기화",
        "loop": "Loop 2 - Evaluation",
        "func": step6_batch_langfuse,
        "is_async": False,
    },
    7: {
        "name": "Remediation Agent 실행",
        "loop": "Loop 3 - Remediation",
        "func": step7_run_remediation,
        "is_async": True,
    },
    8: {
        "name": "Evaluation Prompt 자동 최적화",
        "loop": "Loop 2 - Evaluation",
        "func": step8_optimize_prompts,
        "is_async": False,
    },
}


# ── 파이프라인 실행 엔진 ─────────────────────────────────────


async def run_pipeline(
    *,
    steps_to_run: list[int],
    skip_review: bool = False,
    num_goldens: int = 10,
    categories: list[str] | None = None,
    eval_sample_ratio: float | None = None,
    eval_sample_size: int | None = None,
    eval_sample_seed: int = 42,
    eval_stratify_by: list[str] | None = None,
    continue_on_error: bool = False,
    optimizer_iterations: int = 2,
    optimizer_max_cases: int | None = None,
    optimizer_apply: bool = False,
    lf_tags: list[str] | None = None,
    lf_hours: int = 24,
    lf_limit: int = 200,
    lf_sample_ratio: float | None = None,
    lf_sample_size: int | None = None,
    lf_sample_seed: int = 42,
    lf_score_prefix: str = "eval",
    lf_threshold: float = 0.7,
    lf_metric_thresholds: dict[str, float] | None = None,
    lf_include_missing_scores: bool = False,
    optimizer_failure_hints_path: str | None = None,
    optimizer_failure_hints_max: int = 6,
) -> dict[int, dict]:
    """파이프라인 단계를 순서대로 실행합니다.

    각 단계의 성공/실패 상태와 소요 시간을 추적하고,
    최종 요약을 출력합니다.

    Args:
        steps_to_run: 실행할 단계 번호 리스트 (정렬됨)
        skip_review: Step 4에서 Human Review 건너뛰기
        num_goldens: Synthetic/Golden 생성 항목 수
        categories: DeepEval 메트릭 카테고리
        eval_sample_ratio: Step 5 Golden 샘플링 비율 (0.0~1.0)
        eval_sample_size: Step 5 샘플 최대 건수 상한
        eval_sample_seed: Step 5 deterministic 샘플링 시드
        eval_stratify_by: Step 5 층화 샘플링 기준 필드 목록
        continue_on_error: 에러 발생 시 다음 단계로 계속 진행 여부
        optimizer_iterations: Step 8 최적화 반복 횟수
        optimizer_max_cases: Step 8 메트릭별 최대 캘리브레이션 케이스 수
        optimizer_apply: Step 8 완료 후 prompts.py 반영 여부
        lf_tags: Step 6 Langfuse 태그 필터
        lf_hours: Step 6 Langfuse 조회 시간 범위 (최근 N시간)
        lf_limit: Step 6 Langfuse 조회 최대 건수
        lf_sample_ratio: Step 6 비율 샘플링 (0.0~1.0)
        lf_sample_size: Step 6 샘플 최대 건수 상한
        lf_sample_seed: Step 6 deterministic 샘플링 시드
        lf_score_prefix: Step 6 모니터링 스코어 prefix
        lf_threshold: Step 6 기본 실패 임계값
        lf_metric_thresholds: Step 6 메트릭별 임계값 override
        lf_include_missing_scores: Step 6 prefix score 없는 trace 포함 여부
        optimizer_failure_hints_path: Step 8에서 사용할 Langfuse 실패샘플 파일 경로
        optimizer_failure_hints_max: Step 8 메트릭별 힌트 최대 개수

    Returns:
        {단계번호: {"status": "success"|"failed"|"skipped", "duration": float, ...}}
    """
    results: dict[int, dict] = {}

    print("=" * 60)
    print("  Day3 AI Agent Ops - 통합 파이프라인")
    print("=" * 60)
    print(f"  실행 단계: {steps_to_run}")
    print(f"  Human Review 건너뛰기: {skip_review}")
    print(f"  Golden 항목 수: {num_goldens}")
    print(f"  메트릭 카테고리: {categories or ['rag', 'custom']}")
    print(f"  Step5 sample_ratio: {eval_sample_ratio}")
    print(f"  Step5 sample_size: {eval_sample_size}")
    print(f"  Step5 sample_seed: {eval_sample_seed}")
    print(f"  Step5 stratify_by: {eval_stratify_by}")
    print(f"  에러 시 계속: {continue_on_error}")
    print(f"  Prompt 최적화 반복: {optimizer_iterations}")
    print(f"  Prompt 최적화 max_cases: {optimizer_max_cases}")
    print(f"  Prompt 적용: {optimizer_apply}")
    print(f"  Langfuse tags: {lf_tags}")
    print(f"  Langfuse 조회 시간(시): {lf_hours}")
    print(f"  Langfuse 조회 limit: {lf_limit}")
    print(f"  Langfuse sample_ratio: {lf_sample_ratio}")
    print(f"  Langfuse sample_size: {lf_sample_size}")
    print(f"  Langfuse sample_seed: {lf_sample_seed}")
    print(f"  Langfuse score_prefix: {lf_score_prefix}")
    print(f"  Langfuse default_threshold: {lf_threshold}")
    print(f"  Langfuse metric_thresholds: {lf_metric_thresholds}")
    print(f"  Langfuse include_missing_scores: {lf_include_missing_scores}")
    print(f"  Step8 failure_hints_path: {optimizer_failure_hints_path}")
    print(f"  Step8 failure_hints_max: {optimizer_failure_hints_max}")
    print("=" * 60)

    for step_num in steps_to_run:
        step_info = STEPS[step_num]
        loop_label = step_info["loop"]
        step_name = step_info["name"]

        print(f"\n{'─' * 60}")
        print(f"  [{loop_label}] Step {step_num}: {step_name}")
        print(f"{'─' * 60}")

        start_time = time.time()

        try:
            # 각 단계별 파라미터를 전달하여 실행
            if step_num == 1:
                result = step_info["func"](num_goldens=num_goldens)
            elif step_num == 3:
                result = step_info["func"](augment=True)
            elif step_num == 4:
                result = step_info["func"](
                    num_goldens=num_goldens,
                    skip_review=skip_review,
                )
            elif step_num == 5:
                result = step_info["func"](
                    categories=categories,
                    sample_ratio=eval_sample_ratio,
                    sample_size=eval_sample_size,
                    sample_seed=eval_sample_seed,
                    stratify_by=eval_stratify_by,
                )
            elif step_num == 6:
                result = step_info["func"](
                    tags=lf_tags,
                    from_hours_ago=lf_hours,
                    limit=lf_limit,
                    sample_ratio=lf_sample_ratio,
                    sample_size=lf_sample_size,
                    sample_seed=lf_sample_seed,
                    score_prefix=lf_score_prefix,
                    default_threshold=lf_threshold,
                    metric_thresholds=lf_metric_thresholds,
                    include_missing_scores=lf_include_missing_scores,
                )
            elif step_num == 8:
                result = step_info["func"](
                    iterations=optimizer_iterations,
                    max_cases=optimizer_max_cases,
                    apply=optimizer_apply,
                    failure_hints_path=optimizer_failure_hints_path,
                    failure_hints_max=optimizer_failure_hints_max,
                )
            elif step_info["is_async"]:
                result = await step_info["func"]()
            else:
                result = step_info["func"]()

            elapsed = time.time() - start_time
            results[step_num] = {
                "status": "success",
                "duration": elapsed,
                "result": result,
            }
            print(f"\n  ✓ Step {step_num} 완료 ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start_time
            results[step_num] = {
                "status": "failed",
                "duration": elapsed,
                "error": str(e),
            }
            print(f"\n  ✗ Step {step_num} 실패 ({elapsed:.1f}s): {e}")

            if not continue_on_error:
                print(
                    "\n  파이프라인 중단. --continue-on-error 옵션으로 계속 진행할 수 있습니다."
                )
                break

    # ── 최종 요약 ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  파이프라인 실행 요약")
    print(f"{'=' * 60}")

    total_time = sum(r["duration"] for r in results.values())
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    fail_count = sum(1 for r in results.values() if r["status"] == "failed")

    for step_num in sorted(results.keys()):
        r = results[step_num]
        icon = "✓" if r["status"] == "success" else "✗"
        name = STEPS[step_num]["name"]
        print(f"  {icon} Step {step_num}: {name} ({r['duration']:.1f}s)")
        if r["status"] == "failed":
            print(f"    └─ 에러: {r.get('error', 'Unknown')}")

    print(f"\n  총 소요 시간: {total_time:.1f}s")
    print(f"  성공: {success_count} / 실패: {fail_count} / 전체: {len(results)}")
    print(f"{'=' * 60}")

    return results


# ── CLI 진입점 ───────────────────────────────────────────────


def main():
    """CLI 진입점.

    argparse를 사용하여 다양한 실행 모드를 제공합니다:
    - --all: 전체 파이프라인 실행 (Step 1~8)
    - --step N: 특정 단계만 실행
    - --from N --to M: 범위 지정 실행
    - --skip-review: Human Review 건너뛰기 (Step 4에서 자동 승인)
    - --num-goldens: 생성할 Golden 항목 수
    - --categories: DeepEval 메트릭 카테고리
    - --continue-on-error: 에러 발생 시에도 계속 진행
    """
    parser = argparse.ArgumentParser(
        description="Day3 AI Agent Ops 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
실행 예시:
  전체 파이프라인:    python scripts/run_pipeline.py --all --skip-review
  단일 단계:         python scripts/run_pipeline.py --step 5
  범위 지정:         python scripts/run_pipeline.py --from 4 --to 8
  에러 무시:         python scripts/run_pipeline.py --all --continue-on-error
  Loop 1만:         python scripts/run_pipeline.py --from 1 --to 4 --skip-review
  Loop 2+3만:       python scripts/run_pipeline.py --from 5 --to 8
  Prompt 최적화만:   python scripts/run_pipeline.py --step 8 --opt-iters 2 --opt-max-cases 4
        """,
    )

    # 실행 모드 (상호 배타적 그룹)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="전체 파이프라인 실행 (Step 1~8)",
    )
    mode_group.add_argument(
        "--step",
        type=int,
        choices=range(1, 9),
        metavar="N",
        help="특정 단계만 실행 (1~8)",
    )
    mode_group.add_argument(
        "--from",
        type=int,
        choices=range(1, 9),
        metavar="N",
        dest="from_step",
        help="시작 단계 번호 (--to와 함께 사용)",
    )

    # 범위 종료 (--from과 함께)
    parser.add_argument(
        "--to",
        type=int,
        choices=range(1, 9),
        metavar="M",
        dest="to_step",
        help="종료 단계 번호 (--from과 함께 사용)",
    )

    # 공통 옵션
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Human Review 건너뛰기 (Step 4에서 자동 승인)",
    )
    parser.add_argument(
        "--num-goldens",
        type=int,
        default=10,
        help="생성할 Golden 항목 수 (기본: 10)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="DeepEval 메트릭 카테고리 (기본: rag custom)",
    )
    parser.add_argument(
        "--eval-sample-ratio",
        type=float,
        default=None,
        help="Step 5 Golden 샘플링 비율 (0.0~1.0)",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=None,
        help="Step 5 Golden 샘플 최대 건수 상한",
    )
    parser.add_argument(
        "--eval-sample-seed",
        type=int,
        default=42,
        help="Step 5 deterministic 샘플링 시드 (기본: 42)",
    )
    parser.add_argument(
        "--eval-stratify-by",
        nargs="+",
        default=["source_file"],
        help="Step 5 층화 샘플링 기준 필드 (기본: source_file)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="에러 발생 시에도 다음 단계로 계속 진행",
    )
    parser.add_argument(
        "--opt-iters",
        type=int,
        default=2,
        help="Step 8 프롬프트 최적화 반복 횟수 (기본: 2)",
    )
    parser.add_argument(
        "--opt-max-cases",
        type=int,
        default=None,
        help="Step 8 메트릭별 최대 캘리브레이션 케이스 수",
    )
    parser.add_argument(
        "--opt-apply",
        action="store_true",
        help="Step 8 완료 후 prompts.py에 최적 프롬프트 반영",
    )
    parser.add_argument(
        "--lf-hours",
        type=int,
        default=24,
        help="Step 6 Langfuse 조회 시간 범위 (기본: 24)",
    )
    parser.add_argument(
        "--lf-tags",
        nargs="+",
        default=None,
        help="Step 6 Langfuse 태그 필터 (예: env:prod version:0.1.0)",
    )
    parser.add_argument(
        "--lf-limit",
        type=int,
        default=200,
        help="Step 6 Langfuse 조회 최대 건수 (기본: 200)",
    )
    parser.add_argument(
        "--lf-sample-ratio",
        type=float,
        default=None,
        help="Step 6 비율 샘플링 (0.0~1.0)",
    )
    parser.add_argument(
        "--lf-sample-size",
        type=int,
        default=None,
        help="Step 6 샘플 최대 건수 상한",
    )
    parser.add_argument(
        "--lf-sample-seed",
        type=int,
        default=42,
        help="Step 6 deterministic 샘플링 시드 (기본: 42)",
    )
    parser.add_argument(
        "--lf-score-prefix",
        type=str,
        default="eval",
        help="Step 6 모니터링 스코어 prefix (기본: eval)",
    )
    parser.add_argument(
        "--lf-threshold",
        type=float,
        default=0.7,
        help="Step 6 기본 실패 임계값 (기본: 0.7)",
    )
    parser.add_argument(
        "--lf-metric-threshold",
        action="append",
        default=None,
        help="Step 6 메트릭별 임계값 override (예: faithfulness=0.8)",
    )
    parser.add_argument(
        "--lf-include-missing-scores",
        action="store_true",
        help="Step 6 prefix score가 없는 trace도 결과에 포함",
    )
    parser.add_argument(
        "--opt-lf-failures",
        type=str,
        default=None,
        help="Step 8에 반영할 Langfuse 실패샘플 JSON 경로",
    )
    parser.add_argument(
        "--opt-lf-hints-max",
        type=int,
        default=6,
        help="Step 8 메트릭별 Langfuse 힌트 최대 개수 (기본: 6)",
    )

    args = parser.parse_args()

    try:
        lf_metric_thresholds = _parse_metric_threshold_pairs(args.lf_metric_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    # 실행할 단계 번호 리스트 결정
    if args.all:
        steps_to_run = list(range(1, 9))
    elif args.step:
        steps_to_run = [args.step]
    elif args.from_step:
        to_step = args.to_step or 8
        if to_step < args.from_step:
            parser.error("--to 값은 --from 값보다 크거나 같아야 합니다.")
        steps_to_run = list(range(args.from_step, to_step + 1))
    else:
        parser.error("--all, --step, 또는 --from 중 하나를 지정하세요.")

    # 비동기 파이프라인 실행
    pipeline_results = asyncio.run(
        run_pipeline(
            steps_to_run=steps_to_run,
            skip_review=args.skip_review,
            num_goldens=args.num_goldens,
            categories=args.categories,
            eval_sample_ratio=args.eval_sample_ratio,
            eval_sample_size=args.eval_sample_size,
            eval_sample_seed=args.eval_sample_seed,
            eval_stratify_by=args.eval_stratify_by,
            continue_on_error=args.continue_on_error,
            optimizer_iterations=max(0, args.opt_iters),
            optimizer_max_cases=args.opt_max_cases,
            optimizer_apply=args.opt_apply,
            lf_tags=args.lf_tags,
            lf_hours=max(1, args.lf_hours),
            lf_limit=max(1, args.lf_limit),
            lf_sample_ratio=args.lf_sample_ratio,
            lf_sample_size=args.lf_sample_size,
            lf_sample_seed=args.lf_sample_seed,
            lf_score_prefix=args.lf_score_prefix,
            lf_threshold=args.lf_threshold,
            lf_metric_thresholds=lf_metric_thresholds or None,
            lf_include_missing_scores=args.lf_include_missing_scores,
            optimizer_failure_hints_path=args.opt_lf_failures,
            optimizer_failure_hints_max=max(1, args.opt_lf_hints_max),
        )
    )

    # 실패한 단계가 있으면 exit code 1 반환
    has_failure = any(r["status"] == "failed" for r in pipeline_results.values())
    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    main()
