#!/usr/bin/env python3
"""Langfuse Dataset 기반 평가 실험을 실행한다.

Langfuse v4의 run_experiment() API를 사용하여
Dataset → Task → Evaluator 파이프라인을 실행하고,
결과를 Langfuse Experiments UI에 자동 등록한다.

실행:
    python -m coding_agent.scripts.10_run_langfuse_experiment
    python -m coding_agent.scripts.10_run_langfuse_experiment --run-name "v1-baseline"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# 패키지 루트의 .env를 로드
_package_root = Path(__file__).resolve().parent.parent
load_dotenv(_package_root / ".env", override=True)

from langfuse import Evaluation, Langfuse  # noqa: E402
from coding_agent.eval_pipeline.settings import get_settings  # noqa: E402
from coding_agent.eval_pipeline.observability.langfuse import enabled  # noqa: E402

_settings = get_settings()


def _task(*, item, **kwargs) -> str:
    """각 dataset item에 대해 Coding Agent를 실행한다."""
    from coding_agent.eval_pipeline.my_agent import run_coding_agent

    query = (
        item.input.get("query", "") if isinstance(item.input, dict) else str(item.input)
    )
    print(f"   🔧 에이전트 실행: {query[:60]}...")

    result = run_coding_agent(query)
    print(f"   ✅ 응답 ({len(result)}자)")
    return result


def _response_completeness_evaluator(
    *,
    input: Any,
    output: Any,
    expected_output: Any,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Evaluation:
    """DeepEval의 response_completeness 메트릭으로 평가한다."""
    from deepeval.test_case import LLMTestCase
    from coding_agent.eval_pipeline.loop2_evaluation.custom_metrics import (
        create_response_completeness_metric,
    )

    query = input.get("query", "") if isinstance(input, dict) else str(input)
    expected = (
        expected_output.get("response", "")
        if isinstance(expected_output, dict)
        else str(expected_output)
    )
    actual = output if isinstance(output, str) else str(output)

    try:
        tc = LLMTestCase(input=query, actual_output=actual, expected_output=expected)
        metric = create_response_completeness_metric(threshold=0.5)
        metric.measure(tc)
        score = metric.score
        reason = getattr(metric, "reason", "")
        print(f"   📊 response_completeness: {score:.2f}")
    except Exception as e:
        score = 0.0
        reason = str(e)
        print(f"   ⚠️ 평가 실패: {e}")

    return Evaluation(
        name="response_completeness",
        value=score,
        comment=reason[:200] if reason else None,
    )


def run_experiment(dataset_name: str, run_name: str | None) -> None:
    if not enabled():
        print("❌ Langfuse가 비활성화 상태입니다.")
        print(f"   LANGFUSE_HOST={_settings.langfuse_host}")
        print(
            f"   LANGFUSE_PUBLIC_KEY={_settings.langfuse_public_key[:10] if _settings.langfuse_public_key else 'EMPTY'}..."
        )
        return

    lf = Langfuse()
    dataset = lf.get_dataset(dataset_name)

    print(f"📂 데이터셋: {dataset_name} ({len(dataset.items)}개 항목)")
    print(f"🚀 실험: {run_name or '(자동 생성)'}")
    print(f"   모델: {os.getenv('MODEL_NAME', 'deepseek/deepseek-v3.2')}")
    print()

    result = lf.run_experiment(
        name=dataset_name,
        run_name=run_name,
        data=dataset.items,
        task=_task,
        evaluators=[_response_completeness_evaluator],
        max_concurrency=1,
    )

    # 결과 요약
    scores = []
    for item_result in result.item_results:
        for eval_result in item_result.evaluations or []:
            try:
                scores.append(float(eval_result.value))
            except (TypeError, ValueError, AttributeError):
                pass

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n🎉 완료: {len(scores)}개 평가, 평균 score: {avg:.2f}")
    if result.dataset_run_url:
        print(f"   결과 URL: {result.dataset_run_url}")
    print(f"   확인: {_settings.langfuse_host}")

    lf.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="coding-assistant-golden")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_experiment(args.dataset_name, args.run_name)


if __name__ == "__main__":
    main()
