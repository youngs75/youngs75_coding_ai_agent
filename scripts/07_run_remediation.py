#!/usr/bin/env python3
"""Step 7: Remediation Agent 실행.

Usage:
    python scripts/07_run_remediation.py [--eval-dir data/eval_results]
"""

from __future__ import annotations

import argparse
import asyncio
import json


from youngs75_a2a.eval_pipeline.loop3_remediation.remediation_agent import run_remediation


async def _main():
    parser = argparse.ArgumentParser(description="Run remediation agent")
    parser.add_argument("--eval-dir", type=str, default=None)
    args = parser.parse_args()

    print("[Step 7] Running Remediation Agent...")
    report = await run_remediation(eval_results_dir=args.eval_dir)

    print(f"\n{'=' * 60}")
    print("REMEDIATION REPORT")
    print(f"{'=' * 60}")
    print(f"\nSummary: {report.summary}")
    print("\nFailure Analysis:")
    print(f"  Total evaluated: {report.failure_analysis.total_evaluated}")
    print(f"  Total failed: {report.failure_analysis.total_failed}")
    print(f"  Failure rate: {report.failure_analysis.failure_rate:.1%}")

    if report.prompt_optimizations:
        print(f"\nPrompt Optimizations ({len(report.prompt_optimizations)}):")
        for opt in report.prompt_optimizations:
            print(f"  - [{opt.target_prompt}] {opt.current_issue}")
            print(f"    → {opt.suggested_change}")

    if report.recommendations:
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for rec in report.recommendations:
            print(f"  - [{rec.priority.upper()}] {rec.title}")
            print(f"    Category: {rec.category}, Complexity: {rec.implementation_complexity}")
            print(f"    Impact: {rec.expected_impact}")

    if report.next_steps:
        print("\nNext Steps:")
        for step in report.next_steps:
            print(f"  - {step}")

    # JSON 저장
    from youngs75_a2a.eval_pipeline.settings import get_settings

    settings = get_settings()
    output_path = settings.data_dir / "eval_results" / "remediation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"\n[Step 7] Report saved: {output_path}")


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
