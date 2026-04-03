"""Remediation Agent 모듈 (Loop 3 핵심).

DeepAgents 기반으로 평가 결과를 분석하고 개선안을 추천하는 에이전트입니다.
Day2의 deep_researcher.py와 동일한 create_deep_agent() 패턴을 재사용합니다.

아키텍처 (Day2 패턴 재사용):
    ┌─────────────────────────────────┐
    │     Supervisor (도구 없음)        │ ← 워크플로우 관리
    │  ┌──────────────────────────┐   │
    │  │ analyzer (2 도구)         │   │ ← read_eval_results, read_langfuse_scores
    │  ├──────────────────────────┤   │
    │  │ optimizer (도구 없음)      │   │ ← 사고(thinking)만 수행
    │  ├──────────────────────────┤   │
    │  │ recommender (도구 없음)    │   │ ← JSON 리포트 생성
    │  └──────────────────────────┘   │
    └─────────────────────────────────┘

실행 흐름:
    1. Supervisor가 analyzer에게 평가 결과 분석을 위임
    2. analyzer가 도구를 사용하여 데이터를 읽고 실패 패턴을 분류
    3. Supervisor가 optimizer에게 최적화 제안 생성을 위임
    4. Supervisor가 recommender에게 최종 리포트 생성을 위임
    5. 구조화된 RecommendationReport JSON 출력

사용 예시:
    from youngs75_a2a.eval_pipeline.loop3_remediation.remediation_agent import run_remediation
    report = await run_remediation()
    print(report.summary)
    for rec in report.recommendations:
        print(f"[{rec.priority}] {rec.title}")
"""

from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.state import StateBackend
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from youngs75_a2a.eval_pipeline.llm.json_utils import extract_json_object
from youngs75_a2a.eval_pipeline.loop3_remediation.analysis_tools import read_eval_results, read_langfuse_scores
from youngs75_a2a.eval_pipeline.loop3_remediation.prompts import (
    ANALYZER_PROMPT,
    OPTIMIZER_PROMPT,
    RECOMMENDER_PROMPT,
    SUPERVISOR_PROMPT,
)
from youngs75_a2a.eval_pipeline.loop3_remediation.recommendation import RecommendationReport
from youngs75_a2a.eval_pipeline.settings import get_settings


def _get_chat_model() -> ChatOpenAI:
    """OpenRouter를 통한 ChatOpenAI 모델을 생성합니다.

    DeepAgents의 create_deep_agent()는 ChatOpenAI 인스턴스를 받으므로,
    OpenRouter의 base_url을 설정한 ChatOpenAI를 생성합니다.

    Returns:
        ChatOpenAI: OpenRouter 엔드포인트가 설정된 LangChain 채팅 모델
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openrouter_model_name,
        api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
    )


def _subagents() -> list[dict[str, Any]]:
    """Remediation 서브에이전트를 정의합니다.

    Day2의 _subagents() 패턴과 동일한 구조:
    각 서브에이전트는 name, description, system_prompt, tools를 갖습니다.

    Returns:
        서브에이전트 정의 딕셔너리 리스트 (3개)
    """
    return [
        {
            "name": "analyzer",
            "description": "평가 결과와 Langfuse 스코어를 읽어 실패 패턴을 분류합니다.",
            "system_prompt": ANALYZER_PROMPT,
            "tools": [read_eval_results, read_langfuse_scores],  # 도구 2개 제공
        },
        {
            "name": "optimizer",
            "description": "분석 결과를 기반으로 프롬프트/워크플로우 최적화를 제안합니다.",
            "system_prompt": OPTIMIZER_PROMPT,
            "tools": [],  # 도구 없음: 사고(thinking)만 수행
        },
        {
            "name": "recommender",
            "description": "구조화된 JSON 추천 리포트를 생성합니다.",
            "system_prompt": RECOMMENDER_PROMPT,
            "tools": [],  # 도구 없음: JSON 생성만 수행
        },
    ]


def create_remediation_agent():
    """DeepAgents 기반 Remediation Agent를 생성합니다.

    Day2의 create_deep_agent() 패턴을 재사용합니다:
    - model: OpenRouter를 통한 ChatOpenAI
    - backend: StateBackend (인메모리 상태 관리)
    - store: InMemoryStore (세션 내 데이터 유지)
    - checkpointer: MemorySaver (LangGraph 체크포인트)

    Returns:
        tuple: (agent, store, checkpointer)
    """
    model = _get_chat_model()
    store = InMemoryStore()
    checkpointer = MemorySaver()

    # 백엔드 팩토리: DeepAgents가 상태를 관리하는 방식 정의
    def backend_factory(rt):
        return StateBackend(rt)

    # Supervisor는 도구 없음 (서브에이전트 위임만 수행)
    agent = create_deep_agent(
        model=model,
        tools=[],
        system_prompt=SUPERVISOR_PROMPT,
        subagents=_subagents(),  # type: ignore[arg-type]
        backend=backend_factory,
        store=store,
        checkpointer=checkpointer,
        name="remediation_supervisor",
    )
    return agent, store, checkpointer


async def run_remediation(
    *,
    eval_results_dir: str | None = None,
    thread_id: str = "remediation",
) -> RecommendationReport:
    """Remediation Agent를 실행하여 구조화된 추천 리포트를 생성합니다.

    에이전트가 평가 결과를 자동으로 읽고, 실패 패턴을 분석하고,
    구체적인 개선 추천을 담은 리포트를 생성합니다.

    Args:
        eval_results_dir: 평가 결과 디렉토리 경로 (기본: data/eval_results)
        thread_id: LangGraph thread ID (체크포인팅용)

    Returns:
        RecommendationReport: 구조화된 추천 리포트 Pydantic 모델.
        JSON 파싱 실패 시 summary에 원본 텍스트를 담은 기본 리포트를 반환합니다.
    """
    agent, _store, _checkpointer = create_remediation_agent()

    # 에이전트에게 전달할 초기 지시 메시지
    query = (
        "Analyze the evaluation results and generate a comprehensive remediation report. "
        "Use the read_eval_results tool to access the evaluation data. "
        "If Langfuse is configured, also use read_langfuse_scores to get online metrics. "
        "Then provide specific, actionable recommendations for improvement."
    )

    if eval_results_dir:
        query += f"\nEvaluation results directory: {eval_results_dir}"

    input_state = {"messages": [HumanMessage(content=query)]}
    cfg = {"configurable": {"thread_id": thread_id}}

    # 에이전트 비동기 실행
    result = await agent.ainvoke(input_state, config=cfg)

    # 최종 AI 메시지에서 응답 텍스트 추출
    response_text = ""
    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "type", None) == "ai":
            response_text = str(getattr(msg, "content", "") or "")
            break

    # LLM 응답에서 JSON 추출 → RecommendationReport로 파싱
    try:
        data = extract_json_object(response_text)
        return RecommendationReport(**data)
    except Exception as exc:
        print(f"[WARN] Remediation report JSON parsing failed: {exc}")
        # JSON 파싱 실패 시: 원본 텍스트를 summary에 담은 기본 리포트 반환
        return RecommendationReport(
            summary=response_text[:500]
            if response_text
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
