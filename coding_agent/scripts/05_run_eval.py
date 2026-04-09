#!/usr/bin/env python3
"""Step 5: DeepEval 평가 실행.

Usage:
    python scripts/05_run_eval.py [--categories rag custom] [--sample-ratio 0.3 --sample-size 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path


from coding_agent.eval_pipeline.loop2_evaluation.batch_evaluator import (
    evaluate_golden_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run DeepEval evaluation on golden dataset"
    )
    parser.add_argument("--golden-path", type=str, default=None)
    parser.add_argument("--categories", nargs="+", default=["rag", "custom"])
    parser.add_argument("--sample-ratio", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by",
        nargs="+",
        default=["source_file"],
        help="Golden 샘플링 층화 기준 필드 (기본: source_file)",
    )
    args = parser.parse_args()

    golden = Path(args.golden_path) if args.golden_path else None
    results = evaluate_golden_dataset(
        golden_path=golden,
        metric_categories=args.categories,
        sample_ratio=args.sample_ratio,
        max_sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        stratify_by=args.stratify_by,
    )
    if not results:
        print("[Step 5] No evaluation results (golden dataset may be missing)")
        return

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n[Step 5] Evaluation complete: {passed}/{total} passed")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        print(f"  [{status}] {r['id']}: {scores_str}")


if __name__ == "__main__":
    main()
