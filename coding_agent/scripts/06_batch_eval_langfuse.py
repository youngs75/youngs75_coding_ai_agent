#!/usr/bin/env python3
"""Step 6: Langfuse 모니터링 스코어 샘플링/실패 추출.

Usage:
    python scripts/06_batch_eval_langfuse.py --tags env:prod --hours 24 --limit 500 \
        --sample-ratio 0.2 --sample-size 80 --score-prefix eval
"""

from __future__ import annotations

import argparse
from pathlib import Path


from coding_agent.eval_pipeline.loop2_evaluation.batch_evaluator import (
    batch_evaluate_langfuse,
    monitor_langfuse_scores,
)


def _parse_metric_thresholds(items: list[str] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid threshold format: {raw} (expected key=value)")
        key, value = raw.split("=", 1)
        metric_key = key.strip()
        if not metric_key:
            raise ValueError(f"Invalid threshold key: {raw}")
        thresholds[metric_key] = float(value.strip())
    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Sync Langfuse monitoring scores and failed samples"
    )
    parser.add_argument(
        "--mode",
        choices=["monitor", "external-deepeval"],
        default="monitor",
        help="monitor: Langfuse 점수만 집계(기본), external-deepeval: 기존 외부평가 실행",
    )
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=["rag", "custom"])

    parser.add_argument("--score-prefix", type=str, default="eval")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--metric-threshold",
        action="append",
        default=None,
        help="메트릭별 임계값 override (예: faithfulness=0.8, eval.safety=0.9)",
    )
    parser.add_argument(
        "--include-missing-scores",
        action="store_true",
        help="prefix score가 없는 trace도 결과에 포함",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="모니터링 스냅샷 저장 경로"
    )
    parser.add_argument(
        "--failed-output", type=str, default=None, help="실패 샘플 저장 경로"
    )
    args = parser.parse_args()

    if args.mode == "external-deepeval":
        results = batch_evaluate_langfuse(
            tags=args.tags,
            from_hours_ago=args.hours,
            limit=args.limit,
            sample_ratio=args.sample_ratio,
            max_sample_size=args.sample_size,
            sample_seed=args.sample_seed,
            metric_categories=args.categories,
        )
        if not results:
            print("[Step 6] No traces evaluated")
            return
        passed = sum(1 for result in results if result["passed"])
        print(f"\n[Step 6] External DeepEval: {passed}/{len(results)} passed")
        return

    try:
        metric_thresholds = _parse_metric_thresholds(args.metric_threshold)
    except ValueError as exc:
        parser.error(str(exc))

    snapshot = monitor_langfuse_scores(
        tags=args.tags,
        from_hours_ago=args.hours,
        limit=args.limit,
        sample_ratio=args.sample_ratio,
        max_sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        score_prefix=args.score_prefix,
        default_threshold=args.threshold,
        metric_thresholds=metric_thresholds or None,
        require_score_prefix=not args.include_missing_scores,
        output_path=Path(args.output) if args.output else None,
        failed_output_path=Path(args.failed_output) if args.failed_output else None,
    )

    summary = snapshot["summary"]
    print(
        "\n[Step 6] Langfuse monitor summary: "
        f"sampled={summary['total_sampled']}, "
        f"collected={summary['total_collected']}, "
        f"failed={summary['traces_failed']}, "
        f"failure_rate={summary['failure_rate']:.1%}"
    )

    failed_samples = snapshot.get("failed_samples", [])
    if failed_samples:
        print("\nTop failed traces:")
        for sample in failed_samples[:10]:
            metric_list = ", ".join(sample["failed_metrics"])
            print(f"  - {sample['trace_id'][:12]}: {metric_list}")


if __name__ == "__main__":
    main()
