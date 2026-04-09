#!/usr/bin/env python3
"""전체 폐쇄 루프 데모: Loop 1 → Loop 2 → Loop 3.

Usage:
    python scripts/demo_full_loop.py [--num-goldens 5]
"""

from __future__ import annotations

import argparse
import asyncio


async def _main():
    parser = argparse.ArgumentParser(description="Full closed-loop demo")
    parser.add_argument("--num-goldens", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("Day3: AI Agent Ops Closed-Loop System Demo")
    print("=" * 60)

    # ── Loop 1: Dataset ──────────────────────────────────────
    print("\n[Loop 1] Dataset Generation")
    print("-" * 40)
    from coding_agent.eval_pipeline.loop1_dataset.golden_builder import (
        build_golden_dataset,
    )

    golden_items = build_golden_dataset(
        num_goldens=args.num_goldens,
        skip_review=True,  # 데모에서는 Human Review 건너뛰기
    )
    print(f"  → Golden Dataset: {len(golden_items)} items\n")

    if not golden_items:
        print("[Demo] No golden items generated. Exiting.")
        return

    # ── Loop 2: Evaluation ───────────────────────────────────
    print("[Loop 2] Evaluation")
    print("-" * 40)
    from coding_agent.eval_pipeline.loop2_evaluation.batch_evaluator import (
        evaluate_golden_dataset,
    )

    eval_results = evaluate_golden_dataset(metric_categories=["rag", "custom"])
    passed = sum(1 for r in eval_results if r["passed"])
    print(f"  → Evaluation: {passed}/{len(eval_results)} passed")

    for r in eval_results:
        status = "PASS" if r["passed"] else "FAIL"
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in r["scores"].items())
        print(f"    [{status}] {r['id']}: {scores_str}")
    print()

    # ── Loop 3: Remediation ──────────────────────────────────
    print("[Loop 3] Remediation")
    print("-" * 40)
    from coding_agent.eval_pipeline.loop3_remediation.remediation_agent import (
        run_remediation,
    )

    report = await run_remediation()
    print(f"  → Summary: {report.summary[:200]}")
    print(f"  → Recommendations: {len(report.recommendations)}")
    for rec in report.recommendations[:3]:
        print(f"    - [{rec.priority}] {rec.title}")

    print(f"\n{'=' * 60}")
    print("Closed-loop complete!")
    print(f"{'=' * 60}")


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
