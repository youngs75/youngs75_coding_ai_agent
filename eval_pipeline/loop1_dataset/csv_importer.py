"""리뷰된 CSV 가져오기 모듈 (Loop 1 - Step 3).

Human Review가 완료된 CSV를 읽어 Golden Dataset JSON으로 변환합니다.
approved=True인 항목만 필터링하거나, 전체 항목을 가져올 수 있습니다.

작동 흐름:
    1. CSV 파일 읽기 (stdlib csv)
    2. approved 필드 파싱 (True/False/빈값)
    3. only_approved=True이면 승인된 항목만 필터
    4. context 문자열을 다시 리스트로 분리
    5. Golden Dataset JSON으로 저장

사용 예시:
    from youngs75_a2a.eval_pipeline.loop1_dataset.csv_importer import import_reviewed_csv
    items = import_reviewed_csv(
        csv_path=Path("data/review/review_dataset.csv"),
        output_path=Path("data/golden/golden_dataset.json"),
        only_approved=True,
    )
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _parse_bool(value) -> bool | None:
    """CSV의 approved 필드를 bool로 안전하게 파싱합니다.

    다양한 형태의 불리언 표현을 처리합니다:
        True 계열: "true", "1", "yes", "y", True
        False 계열: "false", "0", "no", "n", False
        None: NaN, 빈 문자열, 기타

    Args:
        value: CSV에서 읽은 원시 값

    Returns:
        True, False, 또는 None (파싱 불가)
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def import_reviewed_csv(
    csv_path: Path,
    output_path: Path,
    *,
    only_approved: bool = True,
) -> list[dict]:
    """리뷰된 CSV를 Golden Dataset JSON으로 변환합니다.

    CSV의 각 행을 Golden Dataset 항목으로 변환합니다.
    세미콜론으로 구분된 context 문자열을 리스트로 복원합니다.

    Args:
        csv_path: Human Review가 완료된 CSV 파일 경로
        output_path: 출력 Golden Dataset JSON 경로
        only_approved: True이면 approved=True인 항목만 포함.
                       False이면 모든 항목을 포함합니다.

    Returns:
        Golden Dataset 항목 리스트. 각 항목은 dict:
            - id, input, expected_output, context(리스트), source_file
            - synthetic_input_quality, approved, feedback, reviewer
    """
    # utf-8-sig: BOM이 포함된 CSV도 올바르게 읽기
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    golden_items = []
    for row in rows:
        approved = _parse_bool(row.get("approved"))

        # only_approved 모드에서는 명시적으로 승인된 항목만 포함
        if only_approved and approved is not True:
            continue

        # 세미콜론 구분 문자열 → 리스트 복원
        context_str = str(row.get("context", ""))
        context_list = [c.strip() for c in context_str.split(";") if c.strip()]

        # synthetic_input_quality 안전 변환
        try:
            quality = float(row.get("synthetic_input_quality", 0.0))
        except (TypeError, ValueError):
            quality = 0.0

        item = {
            "id": str(row.get("id", "")),
            "input": str(row.get("input", "")),
            "expected_output": str(row.get("expected_output", "")),
            "context": context_list,
            "source_file": str(row.get("source_file", "")),
            "synthetic_input_quality": quality,
            "approved": True,
            "feedback": str(row.get("feedback", "") or ""),
            "reviewer": str(row.get("reviewer", "") or ""),
        }
        golden_items.append(item)

    # Golden Dataset JSON 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_items, f, ensure_ascii=False, indent=2)

    return golden_items
