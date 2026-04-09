"""Golden Dataset 빌더 - Loop 1 오케스트레이터 (Loop 1 - Step 4).

Loop 1의 전체 흐름을 관리하는 오케스트레이터입니다.
Synthetic → Review → Augment → Golden의 4단계를 순차적으로 실행합니다.

전체 흐름:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Synthesizer  │ →  │ CSV Export   │ →  │ Human Review │ →  │ CSV Import   │
    │ (자동 생성)   │    │ (내보내기)    │    │ (수동 검토)   │    │ + LLM 보강   │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                    ↓
                                                            ┌─────────────┐
                                                            │ Golden Dataset│
                                                            │ (최종 확정)   │
                                                            └─────────────┘

세 가지 실행 모드:
    1. skip_review=True: Human Review 없이 바로 Golden으로 확정 (데모/테스트용)
    2. reviewed_csv_path 지정: 이미 리뷰된 CSV에서 시작 (3단계부터)
    3. 기본: Synthetic 생성 → CSV 내보내기 → 사용자에게 리뷰 안내

사용 예시:
    from coding_agent.eval_pipeline.loop1_dataset.golden_builder import build_golden_dataset

    # 데모 모드 (Human Review 건너뛰기)
    items = build_golden_dataset(num_goldens=10, skip_review=True)

    # 리뷰된 CSV에서 Golden 빌드
    items = build_golden_dataset(reviewed_csv_path=Path("data/review/reviewed.csv"))
"""

from __future__ import annotations

import json
from pathlib import Path

from coding_agent.eval_pipeline.loop1_dataset.csv_exporter import export_to_review_csv
from coding_agent.eval_pipeline.loop1_dataset.csv_importer import import_reviewed_csv
from coding_agent.eval_pipeline.loop1_dataset.feedback_augmenter import (
    augment_with_feedback,
)
from coding_agent.eval_pipeline.loop1_dataset.synthesizer import (
    generate_synthetic_dataset,
)
from coding_agent.eval_pipeline.settings import get_settings


def build_golden_dataset(
    *,
    corpus_dir: Path | None = None,
    num_goldens: int = 10,
    skip_review: bool = False,
    reviewed_csv_path: Path | None = None,
) -> list[dict]:
    """Loop 1 전체를 오케스트레이션하여 Golden Dataset을 빌드합니다.

    Args:
        corpus_dir: 소스 문서 디렉토리. None이면 settings.local_corpus_dir 사용
        num_goldens: 생성할 합성 데이터 수
        skip_review: True이면 Human Review 없이 모든 항목을 바로 승인
        reviewed_csv_path: 이미 리뷰된 CSV 경로. 지정 시 Synthetic 생성을 건너뜀

    Returns:
        최종 Golden Dataset 항목 리스트.
        기본 모드(skip_review=False, reviewed_csv_path=None)에서는
        CSV 내보내기 후 빈 리스트를 반환합니다. (사용자가 리뷰 후 재실행 필요)
    """
    settings = get_settings()
    corpus_dir = corpus_dir or settings.local_corpus_dir
    data_dir = settings.data_dir

    # 각 단계의 데이터 경로 설정
    synthetic_path = data_dir / "synthetic" / "synthetic_dataset.json"
    review_csv_path = data_dir / "review" / "review_dataset.csv"
    golden_path = data_dir / "golden" / "golden_dataset.json"

    # ── 모드 1: 이미 리뷰된 CSV가 있는 경우 ──────────────────
    if reviewed_csv_path and reviewed_csv_path.exists():
        print(f"[Loop1] 리뷰된 CSV 사용: {reviewed_csv_path}")
        golden_items = import_reviewed_csv(
            reviewed_csv_path,
            golden_path,
            only_approved=True,
        )

    # ── 모드 2: Human Review 건너뛰기 (데모/테스트용) ─────────
    elif skip_review:
        print(f"[Loop1] {num_goldens}개 합성 데이터 생성 (skip_review=True)...")
        items = generate_synthetic_dataset(
            corpus_dir=corpus_dir,
            output_path=synthetic_path,
            num_goldens=num_goldens,
        )
        # 모든 항목을 자동 승인 처리
        for item in items:
            item["approved"] = True
            item["feedback"] = ""
            item["reviewer"] = "auto"

        golden_path.parent.mkdir(parents=True, exist_ok=True)
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        golden_items = items

    # ── 모드 3: 기본 흐름 (Synthetic → CSV → 리뷰 대기) ──────
    else:
        print(f"[Loop1] {num_goldens}개 합성 데이터 생성 중...")
        generate_synthetic_dataset(
            corpus_dir=corpus_dir,
            output_path=synthetic_path,
            num_goldens=num_goldens,
        )

        print(f"[Loop1] Review CSV 내보내기: {review_csv_path}")
        export_to_review_csv(synthetic_path, review_csv_path)

        # 사용자에게 리뷰 안내 메시지 출력
        print("[Loop1] CSV가 내보내졌습니다. 리뷰 후 import 스크립트를 실행하세요.")
        print(f"  → 편집: {review_csv_path}")
        print("  → 실행: python scripts/03_import_reviewed.py")
        return []  # 리뷰 대기 상태이므로 빈 리스트 반환

    # ── 공통: LLM 피드백 보강 단계 ────────────────────────────
    # feedback 필드가 있는 항목에 대해 LLM이 expected_output을 개선
    items_with_feedback = [item for item in golden_items if item.get("feedback")]
    if items_with_feedback:
        print(f"[Loop1] {len(items_with_feedback)}개 항목 LLM 피드백 보강 중...")
        golden_items = augment_with_feedback(golden_items)

        # 보강된 결과를 Golden Dataset에 덮어쓰기
        with open(golden_path, "w", encoding="utf-8") as f:
            json.dump(golden_items, f, ensure_ascii=False, indent=2)

    print(f"[Loop1] Golden Dataset 완성: {len(golden_items)}개 항목 → {golden_path}")
    return golden_items
