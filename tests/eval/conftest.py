from __future__ import annotations

import os
from pathlib import Path

import pytest

# 테스트 환경에서는 Langfuse 비활성화
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "0")


@pytest.fixture
def settings():
    from youngs75_a2a.eval_pipeline.settings import Settings

    return Settings(
        openrouter_api_key="test-key",
        openrouter_model_name="openai/gpt-5.4",
        langfuse_host="",
        langfuse_public_key="",
        langfuse_secret_key="",
    )


@pytest.fixture
def sample_golden_items() -> list[dict]:
    return [
        {
            "id": "test001",
            "input": "SLA란 무엇인가요?",
            "expected_output": "SLA는 서비스 제공자가 사용자에게 약속하는 서비스 수준을 문서화한 합의입니다.",
            "context": [
                "SLA는 Service Level Agreement의 약자로, 가용성/지연/오류율 등을 정의합니다."
            ],
            "source_file": "00_sla.md",
            "synthetic_input_quality": 0.85,
            "approved": True,
            "feedback": "",
            "reviewer": "test",
        },
        {
            "id": "test002",
            "input": "가용성은 어떻게 측정하나요?",
            "expected_output": "가용성은 1 - (장애 시간 / 총 시간)으로 산출하고 월 단위로 보고합니다.",
            "context": [
                "가용성: 1 - (장애 시간 / 총 시간)으로 산출하고 월 단위로 보고합니다."
            ],
            "source_file": "00_sla.md",
            "synthetic_input_quality": 0.90,
            "approved": True,
            "feedback": "좀 더 구체적인 예시를 추가해주세요",
            "reviewer": "test",
        },
    ]


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """임시 데이터 디렉토리 구조 생성."""
    for subdir in ["corpus", "synthetic", "review", "golden", "eval_results"]:
        (tmp_path / subdir).mkdir()
    return tmp_path
