#!/usr/bin/env python3
"""Step 8: Evaluation 프롬프트 자동 최적화.

Usage:
    python scripts/08_optimize_eval_prompts.py [--iters 2] [--max-cases 6] [--apply]
"""

from __future__ import annotations

import argparse
from pathlib import Path


from youngs75_a2a.eval_pipeline.loop2_evaluation.prompt_optimizer import (
    apply_best_prompts_to_file,
    load_langfuse_failure_hints,
    optimize_all_prompts,
    save_optimization_artifacts,
)
from youngs75_a2a.eval_pipeline.settings import get_settings


def main():
    parser = argparse.ArgumentParser(
        description="Optimize Loop2 evaluation prompts via calibration cases",
    )
    parser.add_argument("--iters", type=int, default=2, help="Optimization iterations")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Max calibration cases per metric",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply best prompts to src/loop2_evaluation/prompts.py",
    )
    parser.add_argument(
        "--lf-failures",
        type=str,
        default=None,
        help="Langfuse failed samples JSON path (default: data/eval_results/langfuse_failed_samples.json if exists)",
    )
    parser.add_argument(
        "--lf-hints-max",
        type=int,
        default=6,
        help="Max Langfuse failure hints per metric (default: 6)",
    )
    args = parser.parse_args()

    settings = get_settings()
    default_hints_path = (
        settings.data_dir / "eval_results" / "langfuse_failed_samples.json"
    )
    hints_path = Path(args.lf_failures) if args.lf_failures else default_hints_path
    failure_hints = None
    if hints_path.exists():
        failure_hints = load_langfuse_failure_hints(
            hints_path,
            max_per_metric=max(1, args.lf_hints_max),
        )

    artifacts = optimize_all_prompts(
        iterations=max(0, args.iters),
        max_cases=args.max_cases,
        langfuse_failure_hints=failure_hints,
    )
    output_paths = save_optimization_artifacts(
        artifacts,
        report_dir=settings.data_dir / "prompt_optimization",
    )

    print(f"[Step 8] Report saved: {output_paths['report_path']}")
    print(f"[Step 8] Best prompts saved: {output_paths['best_prompts_path']}")

    for metric_name, result in artifacts["metric_results"].items():
        baseline_fit = result["baseline_evaluation"]["fit_score"]
        best_fit = result["best_evaluation"]["fit_score"]
        print(f"  - {metric_name}: baseline={baseline_fit:.2f}, best={best_fit:.2f}")
    if failure_hints is not None:
        hint_counts = {metric: len(items) for metric, items in failure_hints.items()}
        print(f"[Step 8] Langfuse hints: {hint_counts} ({hints_path})")
    else:
        print(f"[Step 8] Langfuse hints not found: {hints_path}")

    if args.apply:
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
        print(f"[Step 8] prompts.py updated: {prompts_path}")


if __name__ == "__main__":
    main()
