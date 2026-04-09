"""Remediation Agent 모듈 (Loop 3 핵심).

평가 결과를 분석하고 개선안을 추천하는 에이전트입니다.
deepagents 의존성 없이 OpenAI API를 직접 사용하여 구현합니다.

실행 흐름:
    1. read_eval_results 도구로 평가 데이터 읽기
    2. 분석 → 최적화 제안 → 추천 리포트 생성을 순차 실행
    3. 구조화된 RecommendationReport JSON 출력

사용 예시:
    from coding_agent.eval_pipeline.loop3_remediation.remediation_agent import run_remediation
    report = await run_remediation()
    print(report.summary)
    for rec in report.recommendations:
        print(f"[{rec.priority}] {rec.title}")
"""

from __future__ import annotations

import json

from coding_agent.eval_pipeline.llm.deepeval_model import get_deepeval_model
from coding_agent.eval_pipeline.llm.json_utils import extract_json_object
from coding_agent.eval_pipeline.loop3_remediation.analysis_tools import (
    read_eval_results,
)
from coding_agent.eval_pipeline.loop3_remediation.prompts import (
    ANALYZER_PROMPT,
    OPTIMIZER_PROMPT,
    RECOMMENDER_PROMPT,
)
from coding_agent.eval_pipeline.loop3_remediation.recommendation import (
    RecommendationReport,
)


async def run_remediation(
    *,
    eval_results_dir: str | None = None,
    thread_id: str = "remediation",
) -> RecommendationReport:
    """Remediation Agent를 실행하여 구조화된 추천 리포트를 생성합니다.

    deepagents 의존성 없이 OpenAI API를 직접 사용하여
    분석 → 최적화 → 추천 워크플로우를 순차적으로 실행합니다.

    Args:
        eval_results_dir: 평가 결과 디렉토리 경로 (기본: data/eval_results)
        thread_id: 세션 식별자 (미사용, 호환성 유지용)

    Returns:
        RecommendationReport: 구조화된 추천 리포트 Pydantic 모델.
        JSON 파싱 실패 시 summary에 원본 텍스트를 담은 기본 리포트를 반환합니다.
    """
    model = get_deepeval_model()

    # ── 1단계: 평가 결과 읽기 ──
    eval_data = read_eval_results.invoke({"results_dir": eval_results_dir})
    eval_data_str = json.dumps(eval_data, ensure_ascii=False, indent=2)

    # ── 2단계: 분석 (Analyzer) ──
    analysis_prompt = (
        f"{ANALYZER_PROMPT}\n\n"
        f"다음은 평가 결과 데이터입니다:\n```json\n{eval_data_str}\n```\n\n"
        "위 데이터를 분석하여 실패 패턴과 원인을 분류해주세요."
    )
    analysis_result = model.generate(analysis_prompt)

    # ── 3단계: 최적화 제안 (Optimizer) ──
    optimizer_prompt = (
        f"{OPTIMIZER_PROMPT}\n\n"
        f"다음은 분석 결과입니다:\n{analysis_result}\n\n"
        "구체적인 최적화 제안을 생성해주세요."
    )
    optimization_result = model.generate(optimizer_prompt)

    # ── 4단계: 추천 리포트 생성 (Recommender) ──
    recommender_prompt = (
        f"{RECOMMENDER_PROMPT}\n\n"
        f"다음은 분석 결과입니다:\n{analysis_result}\n\n"
        f"다음은 최적화 제안입니다:\n{optimization_result}\n\n"
        "위 내용을 종합하여 RecommendationReport JSON을 생성해주세요."
    )
    report_result = model.generate(recommender_prompt)

    # LLM 응답에서 JSON 추출 → RecommendationReport로 파싱
    try:
        data = extract_json_object(report_result)
        return RecommendationReport(**data)
    except Exception as exc:
        print(f"[WARN] Remediation report JSON parsing failed: {exc}")
        # JSON 파싱 실패 시: 원본 텍스트를 summary에 담은 기본 리포트 반환
        return RecommendationReport(
            summary=report_result[:500]
            if report_result
            else "Remediation 리포트 파싱에 실패했습니다",
            failure_analysis={  # type: ignore[arg-type]
                "total_evaluated": 0,
                "total_failed": 0,
                "failure_rate": 0.0,
                "categories": [],
            },
            recommendations=[],
            next_steps=["구조화된 평가 결과로 다시 실행하세요"],
        )
