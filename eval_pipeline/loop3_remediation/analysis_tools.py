"""Remediation Agent 분석 도구 모듈 (Loop 3).

Remediation Agent의 analyzer 서브에이전트가 사용하는 도구(tool)를 정의합니다.
LangChain의 @tool 데코레이터를 사용하여 DeepAgents가 호출할 수 있는
함수형 도구로 등록됩니다.

제공 도구:
    1. read_eval_results: 로컬 평가 결과 JSON 읽기 (data/eval_results/)
    2. read_langfuse_scores: Langfuse에서 deepeval.* 스코어 읽기

사용 패턴 (DeepAgents 내부):
    analyzer 서브에이전트가 자동으로 도구를 선택하여 호출합니다.
    - "평가 결과를 분석해주세요" → read_eval_results 호출
    - "Langfuse 스코어를 확인해주세요" → read_langfuse_scores 호출
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from youngs75_a2a.eval_pipeline.observability.langfuse import client, enabled
from youngs75_a2a.eval_pipeline.settings import get_settings

_EVAL_RESULT_FILES = {
    "eval_results.json",
    "langfuse_batch_results.json",
    "langfuse_monitoring_snapshot.json",
    "langfuse_failed_samples.json",
}


@tool
def read_eval_results(results_dir: str | None = None) -> list[dict[str, Any]]:
    """평가 결과 파일을 읽어 분석 데이터를 반환합니다.

    data/eval_results/ 디렉토리에서 평가 관련 JSON 파일만 선택적으로 로드합니다.
    remediation_report.json 등 비평가 파일은 제외하여 혼란을 방지합니다.

    Args:
        results_dir: 평가 결과 디렉토리 경로 (기본: data/eval_results)

    Returns:
        평가 결과 딕셔너리 리스트. 각 항목은:
            - id/trace_id: 항목 식별자
            - scores: {메트릭명: 점수}
            - passed: 통과 여부
        디렉토리가 없거나 파일 읽기 실패 시 error 딕셔너리 포함
    """
    settings = get_settings()
    eval_dir = (
        Path(results_dir) if results_dir else (settings.data_dir / "eval_results")
    )

    results = []
    if not eval_dir.exists():
        return [{"error": f"평가 결과 디렉토리를 찾을 수 없습니다: {eval_dir}"}]

    # 평가 관련 JSON 파일만 선택적으로 읽기
    for path in sorted(eval_dir.glob("*.json")):
        if path.name not in _EVAL_RESULT_FILES:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # 리스트 또는 단일 dict 모두 처리
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        except Exception as e:
            results.append({"error": f"파일 읽기 실패 {path.name}: {e}"})

    return results


@tool
def read_langfuse_scores(
    *,
    trace_ids: list[str] | None = None,
    score_prefix: str = "deepeval",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Langfuse에서 deepeval.* 스코어를 읽어옵니다.

    특정 trace_ids가 주어지면 해당 trace만 조회하고,
    주어지지 않으면 최근 트레이스에서 스코어를 가져옵니다.

    score_prefix로 필터링하여 Day3 평가 스코어만 추출합니다.
    (Day2의 기존 스코어와 분리)

    Args:
        trace_ids: 조회할 특정 trace ID 목록 (None이면 최근 전체)
        score_prefix: 스코어 이름 접두사 필터 (기본: "deepeval")
        limit: 최대 조회 건수 (기본 100)

    Returns:
        스코어 딕셔너리 리스트. 각 항목은:
            - trace_id: Langfuse trace ID
            - metric: 스코어 이름 (예: "deepeval.faithfulness")
            - value: 점수 값
            - comment: 코멘트 (있는 경우)
        Langfuse 미설정 또는 오류 시 error 딕셔너리 포함
    """
    settings = get_settings()
    if not enabled(settings):
        return [{"error": "Langfuse가 설정되지 않았습니다"}]

    lf = client()
    scores_data = []

    try:
        if trace_ids:
            # 특정 trace ID에 대한 스코어 조회
            # Langfuse SDK v3: lf.api.trace.get() 사용
            for tid in trace_ids[:limit]:
                trace_obj = lf.api.trace.get(tid)
                if hasattr(trace_obj, "scores") and trace_obj.scores:
                    for s in trace_obj.scores:
                        # 접두사 필터링: deepeval.* 스코어만 추출
                        if hasattr(s, "name") and s.name.startswith(f"{score_prefix}."):
                            scores_data.append(
                                {
                                    "trace_id": tid,
                                    "metric": s.name,
                                    "value": s.value,
                                    "comment": getattr(s, "comment", None),
                                }
                            )
        else:
            # 최근 트레이스에서 스코어 조회
            # Langfuse SDK v3: lf.api.trace.list() + get() 사용
            response = lf.api.trace.list(limit=limit)
            trace_list = response.data if hasattr(response, "data") else []
            for trace_summary in trace_list:
                tid = trace_summary.id if hasattr(trace_summary, "id") else ""
                trace_obj = lf.api.trace.get(tid)
                if hasattr(trace_obj, "scores") and trace_obj.scores:
                    for s in trace_obj.scores:
                        if hasattr(s, "name") and s.name.startswith(f"{score_prefix}."):
                            scores_data.append(
                                {
                                    "trace_id": tid,
                                    "metric": s.name,
                                    "value": s.value,
                                    "comment": getattr(s, "comment", None),
                                }
                            )
    except Exception as e:
        return [{"error": f"Langfuse 스코어 조회 실패: {e}"}]

    return scores_data
