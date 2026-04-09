#!/usr/bin/env python3
"""Golden Dataset을 Langfuse Datasets에 등록한다.

golden_dataset.json의 각 항목을 Langfuse Dataset Item으로 업로드.
이미 동일 id가 존재하면 upsert(덮어쓰기)된다.

실행:
    python -m coding_agent.scripts.09_upload_golden_to_langfuse

옵션:
    --dataset-name NAME   Langfuse 데이터셋 이름 (기본: coding-assistant-golden)
    --golden-path PATH    Golden Dataset JSON 경로 (기본: settings.data_dir 기반)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse  # noqa: E402

from coding_agent.eval_pipeline.settings import get_settings  # noqa: E402
from coding_agent.eval_pipeline.observability.langfuse import enabled  # noqa: E402


def upload_golden_dataset(
    dataset_name: str = "coding-assistant-golden",
    golden_path: Path | None = None,
) -> None:
    settings = get_settings()

    if not enabled():
        print("❌ Langfuse가 비활성화 상태입니다. .env의 LANGFUSE_* 설정을 확인하세요.")
        return

    if golden_path is None:
        golden_path = settings.data_dir / "golden" / "golden_dataset.json"

    if not golden_path.exists():
        print(f"❌ Golden Dataset을 찾을 수 없습니다: {golden_path}")
        return

    with open(golden_path, encoding="utf-8") as f:
        golden_data = json.load(f)

    print(f"📂 Golden Dataset 로드: {golden_path} ({len(golden_data)}개 항목)")

    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

    # 데이터셋 생성 (이미 존재하면 기존 데이터셋 반환)
    lf.create_dataset(
        name=dataset_name,
        description="Coding Assistant 평가용 Golden Dataset",
        metadata={
            "source": str(golden_path),
            "service": settings.service_name,
            "version": settings.app_version,
        },
    )
    print(f"✅ 데이터셋 생성/확인: {dataset_name}")

    # 각 항목을 Dataset Item으로 업로드
    uploaded = 0
    for item in golden_data:
        lf.create_dataset_item(
            dataset_name=dataset_name,
            id=item["id"],
            input={"query": item["input"]},
            expected_output={"response": item["expected_output"]},
            metadata={
                "context": item.get("context", []),
                "source_file": item.get("source_file", ""),
                "reviewer": item.get("reviewer", ""),
                "approved": item.get("approved", False),
                "feedback": item.get("feedback", ""),
                "synthetic_input_quality": item.get("synthetic_input_quality", 0.0),
            },
        )
        uploaded += 1
        print(
            f"   [{uploaded}/{len(golden_data)}] {item['id']}: {item['input'][:50]}..."
        )

    lf.flush()
    print(
        f"\n🎉 완료: {uploaded}개 항목이 Langfuse 데이터셋 '{dataset_name}'에 등록되었습니다."
    )
    print(f"   확인: {settings.langfuse_host}/datasets/{dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Golden Dataset → Langfuse Datasets 등록"
    )
    parser.add_argument(
        "--dataset-name",
        default="coding-assistant-golden",
        help="Langfuse 데이터셋 이름 (기본: coding-assistant-golden)",
    )
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=None,
        help="Golden Dataset JSON 경로 (기본: settings 기반)",
    )
    args = parser.parse_args()
    upload_golden_dataset(dataset_name=args.dataset_name, golden_path=args.golden_path)


if __name__ == "__main__":
    main()
