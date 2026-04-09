"""최종 보고서 생성 노드."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from coding_agent.agents.deep_research.config import ResearchConfig
from coding_agent.agents.deep_research.schemas import AgentState
from coding_agent.agents.deep_research.prompts import FINAL_REPORT_PROMPT, get_today_str

logger = logging.getLogger(__name__)


async def final_report_generation(state: AgentState, config: RunnableConfig) -> dict:
    """연구 결과를 종합하여 최종 보고서를 작성한다."""
    rc = ResearchConfig.from_runnable_config(config)
    llm = rc.get_model("final_report")

    notes = state.get("notes") or []
    raw_notes = state.get("raw_notes") or []
    all_findings = notes + raw_notes
    findings_text = (
        "\n\n---\n\n".join(all_findings) if all_findings else "연구 결과가 없습니다."
    )

    prompt = FINAL_REPORT_PROMPT.format(
        date=get_today_str(),
        research_brief=state.get("research_brief", ""),
        findings=findings_text,
    )

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content="당신은 전문 보고서 작성자입니다."),
                HumanMessage(content=prompt),
            ]
        )
        return {"final_report": response.content}
    except Exception as e:
        logger.error(f"최종 보고서 생성 실패: {e}")
        # 실패 시 수집된 노트를 그대로 반환
        return {"final_report": f"# 연구 결과 요약\n\n{findings_text}"}
