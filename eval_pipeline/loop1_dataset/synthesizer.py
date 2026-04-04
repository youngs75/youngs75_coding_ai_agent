"""합성 데이터셋 생성 모듈 (Loop 1 - Step 1).

DeepEval의 Synthesizer를 사용하여 소스 문서(corpus)에서
질문-답변 쌍(Golden)을 자동 생성합니다.

작동 흐름:
    1. corpus 디렉토리에서 .md/.txt 문서를 로드
    2. DeepEval Synthesizer에 문서를 컨텍스트로 전달
    3. LLM이 각 컨텍스트에서 질문-답변 쌍을 생성
    4. 결과를 JSON 파일로 저장

사용 예시:
    from youngs75_a2a.eval_pipeline.loop1_dataset.synthesizer import generate_synthetic_dataset
    items = generate_synthetic_dataset(num_goldens=10)
    # → data/synthetic/synthetic_dataset.json 생성

핵심 개념:
    - Synthetic Dataset: LLM이 자동 생성한 평가용 데이터
    - Golden: 하나의 질문-답변-컨텍스트 세트
    - 이 데이터는 Human Review를 거쳐 Golden Dataset으로 확정됩니다
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer

from youngs75_a2a.eval_pipeline.llm.deepeval_model import get_deepeval_model
from youngs75_a2a.eval_pipeline.settings import get_settings


def _load_corpus_documents(corpus_dir: Path) -> tuple[list[list[str]], list[str]]:
    """corpus 디렉토리에서 텍스트 문서를 로드합니다.

    .md와 .txt 파일을 알파벳 순으로 읽어 문자열 리스트로 반환합니다.
    빈 파일은 건너뜁니다.

    Args:
        corpus_dir: 소스 문서가 있는 디렉토리 경로

    Returns:
        - contexts: 문서 내용을 담은 컨텍스트 리스트 (DeepEval 입력 포맷)
        - source_files: 각 컨텍스트의 원본 파일 경로 리스트
    """
    contexts: list[list[str]] = []
    source_files: list[str] = []
    if not corpus_dir.exists():
        return contexts, source_files

    # .md 파일 먼저, 그 다음 .txt 파일 로드
    for pattern in ("*.md", "*.txt"):
        for path in sorted(corpus_dir.glob(pattern)):
            content = path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                contexts.append([content])
                source_files.append(str(path))

    return contexts, source_files


def generate_synthetic_dataset(
    *,
    corpus_dir: Path | None = None,
    output_path: Path | None = None,
    num_goldens: int = 10,
    max_goldens_per_context: int = 2,
) -> list[dict]:
    """DeepEval Synthesizer를 사용하여 합성 데이터셋을 생성합니다.

    소스 문서를 기반으로 LLM이 다양한 질문-답변 쌍을 자동 생성합니다.
    생성된 데이터는 Human Review를 거쳐 Golden Dataset이 됩니다.

    Args:
        corpus_dir: 소스 문서 디렉토리 (기본: settings.local_corpus_dir)
        output_path: 출력 JSON 경로 (기본: data/synthetic/synthetic_dataset.json)
        num_goldens: 생성할 총 golden 수 (많을수록 다양한 질문 생성)
        max_goldens_per_context: 하나의 컨텍스트에서 생성할 최대 golden 수

    Returns:
        생성된 synthetic dataset 항목 리스트. 각 항목은 dict:
            - id: 고유 식별자 (12자리 hex)
            - input: 생성된 질문
            - expected_output: 기대 답변
            - context: 컨텍스트 문자열 리스트
            - source_file: 원본 문서 경로
            - synthetic_input_quality: 자동 품질 점수 (0.0~1.0)

    Raises:
        ValueError: corpus 디렉토리에 문서가 없을 때
    """
    settings = get_settings()
    corpus_dir = corpus_dir or settings.local_corpus_dir
    output_path = output_path or (
        settings.data_dir / "synthetic" / "synthetic_dataset.json"
    )

    # OpenRouter 모델을 DeepEval Synthesizer에 전달
    model = get_deepeval_model()
    contexts, source_files = _load_corpus_documents(corpus_dir)

    if not contexts:
        raise ValueError(f"corpus 디렉토리에 문서가 없습니다: {corpus_dir}")

    # DeepEval Synthesizer: 문서 → 질문-답변 쌍 자동 생성
    synthesizer = Synthesizer(model=model)

    # generate_goldens_from_contexts는 DeepEval 3.8.x에서 지원하는 안정 경로입니다.
    generated_goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        source_files=source_files,
        max_goldens_per_context=max_goldens_per_context,
    )
    if num_goldens > 0:
        generated_goldens = generated_goldens[:num_goldens]

    # EvaluationDataset으로 래핑하여 구조화된 접근 제공
    dataset = EvaluationDataset(goldens=generated_goldens)

    # Golden 객체를 직렬화 가능한 dict로 변환
    items = []
    for golden in dataset.goldens:
        item = {
            "id": uuid4().hex[:12],  # 12자리 짧은 고유 ID
            "input": golden.input if hasattr(golden, "input") else "",
            "expected_output": golden.expected_output
            if hasattr(golden, "expected_output")
            else "",
            "context": golden.context if hasattr(golden, "context") else [],
            "source_file": golden.source_file if hasattr(golden, "source_file") else "",
            "synthetic_input_quality": golden.synthetic_input_quality
            if hasattr(golden, "synthetic_input_quality")
            else 0.0,
        }
        items.append(item)

    # JSON 파일로 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return items
