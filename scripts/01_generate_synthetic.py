#!/usr/bin/env python3
"""Step 1: Synthetic Dataset 생성.

Usage:
    python scripts/01_generate_synthetic.py [--num-goldens 10] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path


from youngs75_a2a.eval_pipeline.loop1_dataset.synthesizer import generate_synthetic_dataset
from youngs75_a2a.eval_pipeline.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from corpus")
    parser.add_argument(
        "--num-goldens", type=int, default=10, help="Number of golden samples to generate"
    )
    parser.add_argument("--corpus-dir", type=str, default=None, help="Corpus directory path")
    parser.add_argument("--dry-run", action="store_true", help="Verify API connection only")
    args = parser.parse_args()

    settings = get_settings()
    print(f"[Step 1] Model: {settings.openrouter_model_name}")

    if args.dry_run:
        from youngs75_a2a.eval_pipeline.llm.deepeval_model import get_deepeval_model

        model = get_deepeval_model()
        result = model.generate("Say 'OK' if you can hear me.")
        print(f"[Dry Run] API response: {result}")
        return

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else None
    items = generate_synthetic_dataset(
        corpus_dir=corpus_dir,
        num_goldens=args.num_goldens,
    )
    print(f"[Step 1] Generated {len(items)} synthetic samples")


if __name__ == "__main__":
    main()
