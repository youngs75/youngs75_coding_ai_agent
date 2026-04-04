"""HITL (Human-In-The-Loop) 승인 및 수정 노드.

LangGraph interrupt() 기반으로만 동작한다.
"""

import logging
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, interrupt

from youngs75_a2a.agents.deep_research.config import ResearchConfig
from youngs75_a2a.agents.deep_research.schemas import HITLAgentState

logger = logging.getLogger(__name__)


async def hitl_final_approval(state: HITLAgentState, config: RunnableConfig) -> Command:
    """최종 보고서 승인 게이트.

    interrupt()로 그래프 실행을 일시 중단하고,
    사용자가 승인하면 완료, 거절하면 수정 루프로 이동한다.

    resume 값:
      - {"approved": True}  → 완료
      - {"approved": False, "feedback": "..."} → 수정
      - 문자열 → 거절 사유로 간주하여 수정
    """
    final_report = state.get("final_report", "")
    logger.info(f"HITL 승인 대기 중 (보고서 길이: {len(final_report)}자)")

    user_response = interrupt(
        {
            "action": "최종 보고서를 검토하고 승인해 주세요.",
            "report_preview": final_report[:500] + "..."
            if len(final_report) > 500
            else final_report,
        }
    )

    # 응답 파싱
    if isinstance(user_response, dict):
        approved = user_response.get("approved", False)
        feedback = user_response.get("feedback", "")
    elif isinstance(user_response, str):
        approved = user_response.lower() in ("yes", "y", "approved", "승인", "확인")
        feedback = "" if approved else user_response
    else:
        approved = bool(user_response)
        feedback = ""

    if approved:
        logger.info("HITL: 보고서 승인됨")
        return Command(goto="__end__")
    else:
        logger.info(f"HITL: 보고서 거절됨 - {feedback}")
        return Command(
            update={"human_feedback": feedback},
            goto="revise_final_report",
        )


async def revise_final_report(state: HITLAgentState, config: RunnableConfig) -> Command:
    """거절된 보고서를 수정하기 위해 연구를 재수행한다.

    revision_count를 증가시키고 피드백을 연구 브리프에 반영한 뒤
    supervisor로 돌아간다.
    """
    rc = ResearchConfig.from_runnable_config(config)
    revision_count = (state.get("revision_count") or 0) + 1

    if revision_count > rc.max_revision_loops:
        logger.warning(
            f"최대 수정 횟수({rc.max_revision_loops}) 도달. 현재 보고서로 확정."
        )
        return Command(goto="__end__")

    feedback = state.get("human_feedback", "")
    original_brief = state.get("research_brief", "")
    revised_brief = f"{original_brief}\n\n## 수정 요청 (#{revision_count})\n{feedback}"

    logger.info(f"수정 루프 #{revision_count}: 재연구 시작")

    return Command(
        update={
            "revision_count": revision_count,
            "research_brief": revised_brief,
            "notes": {"type": "override", "value": []},
            "raw_notes": {"type": "override", "value": []},
        },
        goto="research_supervisor",
    )
