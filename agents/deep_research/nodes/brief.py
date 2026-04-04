"""연구 브리프 작성 노드."""

import logging
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from youngs75_a2a.agents.deep_research.config import ResearchConfig
from youngs75_a2a.agents.deep_research.schemas import AgentState
from youngs75_a2a.agents.deep_research.prompts import (
    RESEARCH_BRIEF_PROMPT,
    SUPERVISOR_PROMPT,
    get_today_str,
)

logger = logging.getLogger(__name__)


async def write_research_brief(state: AgentState, config: RunnableConfig) -> dict:
    """사용자 대화를 분석하여 구체적인 연구 브리프를 작성한다."""
    rc = ResearchConfig.from_runnable_config(config)

    messages_str = get_buffer_string(state["messages"])
    llm = rc.get_model("research")

    prompt = RESEARCH_BRIEF_PROMPT.format(
        date=get_today_str(),
        messages=messages_str,
        max_concurrent_research_units=rc.max_concurrent_research_units,
    )

    # Semantic Memory 주입 — 프로젝트 규칙/컨벤션
    semantic_context = state.get("semantic_context") or []
    if semantic_context:
        prompt += "\n\n## 프로젝트 컨텍스트 (Semantic Memory)\n"
        prompt += "\n".join(f"- {ctx}" for ctx in semantic_context)

    # Episodic Memory 주입 — 과거 유사 연구 이력
    episodic_log = state.get("episodic_log") or []
    if episodic_log:
        prompt += "\n\n## 과거 연구 이력 (Episodic Memory)\n"
        prompt += "\n".join(f"- {entry}" for entry in episodic_log)

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    research_brief = response.content

    # Supervisor 초기 메시지 구성
    supervisor_initial = SystemMessage(
        content=SUPERVISOR_PROMPT.format(
            date=get_today_str(),
            research_brief=research_brief,
            max_concurrent_research_units=rc.max_concurrent_research_units,
        )
    )

    return {
        "research_brief": research_brief,
        "supervisor_messages": [supervisor_initial],
    }
