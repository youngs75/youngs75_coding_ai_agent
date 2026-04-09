#!/usr/bin/env python3
"""Step 3: 리뷰된 CSV 가져오기 + LLM 보강.

Usage:
    python scripts/03_import_reviewed.py [--csv-path data/review/review_dataset.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path


from coding_agent.eval_pipeline.loop1_dataset.csv_importer import import_reviewed_csv
from coding_agent.eval_pipeline.loop1_dataset.feedback_augmenter import (
    augment_with_feedback,
)
from coding_agent.eval_pipeline.settings import get_settings


def main():
    parser = argparse.ArgumentParser(
        description="Import reviewed CSV and augment with LLM"
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Reviewed CSV path")
    parser.add_argument(
        "--skip-augment", action="store_true", help="Skip LLM augmentation"
    )
    args = parser.parse_args()

    settings = get_settings()
    csv_path = (
        Path(args.csv_path)
        if args.csv_path
        else (settings.data_dir / "review" / "review_dataset.csv")
    )
    golden_path = settings.data_dir / "golden" / "golden_dataset.json"

    if not csv_path.exists():
        print(f"[Step 3] Review CSV not found: {csv_path}")
        return

    items = import_reviewed_csv(csv_path, golden_path, only_approved=True)
    print(f"[Step 3] Imported {len(items)} approved items")

    if not args.skip_augment:
        items_with_feedback = [i for i in items if i.get("feedback")]
        if items_with_feedback:
            print(
                f"[Step 3] Augmenting {len(items_with_feedback)} items with LLM feedback..."
            )
            items = augment_with_feedback(items)

            import json

            with open(golden_path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"[Step 3] Augmented golden dataset saved: {golden_path}")
        else:
            print("[Step 3] No items with feedback to augment")

    print(f"[Step 3] Golden dataset: {golden_path}")


if __name__ == "__main__":
    main()
