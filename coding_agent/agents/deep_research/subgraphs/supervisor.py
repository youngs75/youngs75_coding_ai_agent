"""Supervisor 서브그래프.

연구 작업을 조정하고 병렬 연구를 실행하는 감독자 역할.
ConductResearch 도구 호출을 감지하여 Researcher 서브그래프를 병렬 실행한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from coding_agent.core.tool_call_utils import tc_name, tc_id, tc_args
from coding_agent.agents.deep_research.config import ResearchConfig
from coding_agent.agents.deep_research.schemas import (
    ConductResearch,
    ResearchComplete,
    SupervisorInputState,
    SupervisorOutputState,
    SupervisorState,
)
from .researcher import build_researcher_subgraph

logger = logging.getLogger(__name__)


async def supervisor(state: SupervisorState, config: RunnableConfig) -> dict:
    """연구 계획을 수립하고 ConductResearch/ResearchComplete를 호출하는 노드."""
    rc = ResearchConfig.from_runnable_config(config)
    llm = rc.get_model("research")
    llm_with_tools = llm.bind_tools([ConductResearch, ResearchComplete])

    messages = list(state.get("supervisor_messages") or [])

    # 초기 반복에서 강제 연구 수행
    iterations = state.get("research_iterations") or 0
    if (
        rc.supervisor_force_conduct_research_enabled
        and iterations < rc.supervisor_force_conduct_research_until_iteration
        and state.get("research_brief")
    ):
        messages.append(
            HumanMessage(
                content=f"연구 브리프를 분석하여 ConductResearch 도구를 사용해 연구를 시작하세요.\n\n{state['research_brief']}"
            )
        )

    response = await llm_with_tools.ainvoke(messages)
    return {"supervisor_messages": [response]}


def _check_terminate(state: SupervisorState, config: RunnableConfig) -> bool:
    """연구 종료 조건을 확인한다."""
    rc = ResearchConfig.from_runnable_config(config)
    iterations = state.get("research_iterations") or 0
    messages = state.get("supervisor_messages") or []

    if iterations >= rc.max_researcher_iterations:
        return True

    last_msg = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    for call in tool_calls:
        if tc_name(call) == "ResearchComplete":
            return True

    # 첫 반복 이후 도구 호출이 없으면 종료
    if iterations > 0 and not tool_calls:
        return True

    return False


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> dict:
    """Supervisor의 도구 호출을 처리한다.

    ConductResearch 호출을 수집하여 Researcher 서브그래프를 병렬로 실행한다.
    """
    rc = ResearchConfig.from_runnable_config(config)
    messages = state.get("supervisor_messages") or []
    iterations = state.get("research_iterations") or 0

    last_msg = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    if not tool_calls:
        return {"research_iterations": iterations + 1}

    # ConductResearch 호출 수집
    conduct_calls = []
    tool_messages = []

    for call in tool_calls:
        name = tc_name(call)
        call_id = tc_id(call) or "unknown"

        if name == "ConductResearch":
            conduct_calls.append(call)
        elif name == "ResearchComplete":
            tool_messages.append(
                ToolMessage(
                    content="연구 완료 신호 접수됨",
                    tool_call_id=call_id,
                )
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"알 수 없는 도구: {name}",
                    tool_call_id=call_id,
                )
            )

    # 병렬 연구 실행
    all_notes = []
    all_raw_notes = []

    if conduct_calls:
        researcher_graph = build_researcher_subgraph()
        semaphore = asyncio.Semaphore(rc.max_concurrent_research_units)

        async def run_one(call: Any) -> dict:
            topic = tc_args(call).get("research_topic", "")
            call_id = tc_id(call) or "unknown"
            async with semaphore:
                try:
                    result = await researcher_graph.ainvoke(
                        {
                            "researcher_messages": [HumanMessage(content=topic)],
                            "research_topic": topic,
                        },
                        config=config,
                    )
                    return {"call_id": call_id, "result": result, "topic": topic}
                except Exception as e:
                    logger.error(f"연구 실행 실패 ({topic}): {e}")
                    return {
                        "call_id": call_id,
                        "result": None,
                        "topic": topic,
                        "error": str(e),
                    }

        tasks = [run_one(call) for call in conduct_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                continue
            if not isinstance(r, dict):
                continue

            call_id = r["call_id"]
            topic = r["topic"]

            if r.get("error"):
                tool_messages.append(
                    ToolMessage(
                        content=f"연구 실패 ({topic}): {r['error']}",
                        tool_call_id=call_id,
                    )
                )
                continue

            result = r.get("result") or {}
            compressed = result.get("compressed_research", "")
            raw = result.get("raw_notes") or []

            if compressed:
                all_notes.append(compressed)
            all_raw_notes.extend(raw)

            tool_messages.append(
                ToolMessage(
                    content=compressed or f"연구 완료: {topic}",
                    tool_call_id=call_id,
                )
            )

    # Grace period
    if rc.supervisor_research_grace_seconds > 0:
        await asyncio.sleep(rc.supervisor_research_grace_seconds)

    return {
        "supervisor_messages": tool_messages,
        "research_iterations": iterations + 1,
        "notes": all_notes,
        "raw_notes": all_raw_notes,
    }


def _should_continue_supervisor(state: SupervisorState, config: RunnableConfig) -> str:
    """Supervisor 루프를 계속할지 결정한다."""
    if _check_terminate(state, config):
        return "__end__"
    return "supervisor_tools"


def build_supervisor_subgraph(context_schema: type | None = None) -> CompiledStateGraph:
    """Supervisor 서브그래프를 빌드한다."""
    builder = StateGraph(
        SupervisorState,
        input_schema=SupervisorInputState,
        output_schema=SupervisorOutputState,
        context_schema=context_schema,
    )

    builder.add_node("supervisor", supervisor)
    builder.add_node("supervisor_tools", supervisor_tools)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", _should_continue_supervisor)
    builder.add_edge("supervisor_tools", "supervisor")

    return builder.compile()
