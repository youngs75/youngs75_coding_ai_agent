"""사용자 질문 명확화 노드."""

import logging
from langchain_core.messages import AIMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from coding_agent.agents.deep_research.config import ResearchConfig
from coding_agent.agents.deep_research.schemas import AgentState, ClarifyWithUser
from coding_agent.agents.deep_research.prompts import CLARIFY_INSTRUCTIONS

logger = logging.getLogger(__name__)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command:
    """사용자 질문이 연구를 시작하기에 충분히 명확한지 판단한다.

    명확하면 write_research_brief로, 불명확하면 사용자에게 추가 질문을 한다.
    """
    rc = ResearchConfig.from_runnable_config(config)

    if not rc.allow_clarification:
        return Command(goto="retrieve_memory")

    messages_str = get_buffer_string(state["messages"])
    llm = rc.get_model("research", structured=ClarifyWithUser)

    for attempt in range(rc.max_structured_output_retries):
        try:
            result = await llm.ainvoke(
                CLARIFY_INSTRUCTIONS.format(messages=messages_str)
            )
            parsed = result.get("parsed") if isinstance(result, dict) else result
            if parsed is None:
                continue

            if parsed.need_clarification:
                return Command(
                    update={"messages": [AIMessage(content=parsed.question)]},
                    goto="__end__",
                )
            return Command(goto="retrieve_memory")
        except Exception as e:
            logger.warning(f"명확화 판단 시도 {attempt + 1} 실패: {e}")

    # 모든 시도 실패 시 그냥 진행
    return Command(goto="retrieve_memory")
