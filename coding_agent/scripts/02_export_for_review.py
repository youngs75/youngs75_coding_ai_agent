#!/usr/bin/env python3
"""Step 2: Human Review CSV 내보내기.

Usage:
    python scripts/02_export_for_review.py
"""

from __future__ import annotations


from coding_agent.eval_pipeline.loop1_dataset.csv_exporter import export_to_review_csv
from coding_agent.eval_pipeline.settings import get_settings


def main():
    settings = get_settings()
    synthetic_path = settings.data_dir / "synthetic" / "synthetic_dataset.json"
    review_path = settings.data_dir / "review" / "review_dataset.csv"

    if not synthetic_path.exists():
        print(f"[Step 2] Synthetic dataset not found: {synthetic_path}")
        print("  → Run scripts/01_generate_synthetic.py first")
        return

    output = export_to_review_csv(synthetic_path, review_path)
    print(f"[Step 2] Review CSV exported: {output}")
    print("  → Open in Excel/Google Sheets, fill 'approved' and 'feedback' columns")


if __name__ == "__main__":
    main()
