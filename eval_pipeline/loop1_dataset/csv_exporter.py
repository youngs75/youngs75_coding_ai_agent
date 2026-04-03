"""Human Review용 CSV 내보내기 모듈 (Loop 1 - Step 2).

Synthetic Dataset JSON을 사람이 검토할 수 있는 CSV 형식으로 변환합니다.
CSV 파일은 Excel이나 Google Sheets에서 열어 approved/feedback 칼럼을 채울 수 있습니다.

CSV 스키마:
    | 칼럼                  | 타입  | 출처        | 설명                          |
    |----------------------|-------|------------|-------------------------------|
    | id                   | str   | Auto       | 고유 식별자                     |
    | input                | str   | Synthesizer| 생성된 질문                     |
    | expected_output      | str   | Synthesizer| 기대 답변                       |
    | context              | str   | Synthesizer| 컨텍스트 (세미콜론 구분)          |
    | source_file          | str   | Synthesizer| 원본 문서 경로                   |
    | synthetic_input_quality| float| Synthesizer| 자동 품질 점수                  |
    | approved             | bool  | Human      | True/False 승인 여부            |
    | feedback             | str   | Human      | 자유 텍스트 피드백               |
    | reviewer             | str   | Human      | 리뷰어 이름                     |

사용 예시:
    from youngs75_a2a.eval_pipeline.loop1_dataset.csv_exporter import export_to_review_csv
    export_to_review_csv(
        synthetic_path=Path("data/synthetic/synthetic_dataset.json"),
        output_path=Path("data/review/review_dataset.csv"),
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_to_review_csv(
    synthetic_path: Path,
    output_path: Path,
) -> Path:
    """Synthetic Dataset JSON을 Human Review CSV로 내보냅니다.

    context 리스트는 세미콜론(;)으로 구분된 문자열로 변환됩니다.
    approved, feedback, reviewer 칼럼은 빈 값으로 내보내어
    리뷰어가 채울 수 있도록 합니다.

    Args:
        synthetic_path: 입력 Synthetic Dataset JSON 파일 경로
        output_path: 출력 CSV 파일 경로

    Returns:
        생성된 CSV 파일의 Path 객체
    """
    # Synthetic Dataset JSON 로드
    with open(synthetic_path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        # context가 리스트인 경우 세미콜론으로 합침
        # (CSV에서는 리스트를 직접 표현할 수 없으므로)
        context_list = item.get("context", [])
        if isinstance(context_list, list):
            context_str = ";".join(context_list)
        else:
            context_str = str(context_list)

        rows.append(
            {
                "id": item.get("id", ""),
                "input": item.get("input", ""),
                "expected_output": item.get("expected_output", ""),
                "context": context_str,
                "source_file": item.get("source_file", ""),
                "synthetic_input_quality": item.get("synthetic_input_quality", 0.0),
                # Human Review 필드: 리뷰어가 직접 채울 빈 칼럼
                "approved": "",
                "feedback": "",
                "reviewer": "",
            }
        )

    df = pd.DataFrame(rows)

    # 출력 디렉토리 생성 및 CSV 저장
    # utf-8-sig: Excel에서 한글이 깨지지 않도록 BOM 포함
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path
