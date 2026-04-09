#!/usr/bin/env python3
"""Step 4: Golden Dataset 확정 (전체 Loop 1 오케스트레이션).

Usage:
    python scripts/04_build_golden.py [--num-goldens 10] [--skip-review]
"""

from __future__ import annotations

import argparse
from pathlib import Path


from coding_agent.eval_pipeline.loop1_dataset.golden_builder import build_golden_dataset


def main():
    parser = argparse.ArgumentParser(description="Build golden dataset (full Loop 1)")
    parser.add_argument("--num-goldens", type=int, default=10)
    parser.add_argument(
        "--skip-review", action="store_true", help="Skip human review step"
    )
    parser.add_argument("--reviewed-csv", type=str, default=None)
    parser.add_argument("--corpus-dir", type=str, default=None)
    args = parser.parse_args()

    reviewed = Path(args.reviewed_csv) if args.reviewed_csv else None
    corpus = Path(args.corpus_dir) if args.corpus_dir else None

    items = build_golden_dataset(
        corpus_dir=corpus,
        num_goldens=args.num_goldens,
        skip_review=args.skip_review,
        reviewed_csv_path=reviewed,
    )
    print(f"[Step 4] Golden dataset: {len(items)} items")


if __name__ == "__main__":
    main()
