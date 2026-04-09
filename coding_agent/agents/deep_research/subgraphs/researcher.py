"""Researcher 서브그래프.

MCP 도구를 사용하여 특정 주제에 대한 연구를 수행하고
결과를 요약하는 ReAct 루프를 구현한다.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from coding_agent.core.mcp_loader import MCPToolLoader
from coding_agent.core.tool_call_utils import tc_name, tc_id, tc_args
from coding_agent.agents.deep_research.config import ResearchConfig
from coding_agent.agents.deep_research.schemas import (
    ConductResearch,
    ResearchComplete,
    ResearcherInputState,
    ResearcherOutputState,
    ResearcherState,
)
from coding_agent.agents.deep_research.prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    COMPRESS_RESEARCH_PROMPT,
    get_today_str,
)

logger = logging.getLogger(__name__)


async def _execute_tool_safely(tool: Any, args: dict, config: RunnableConfig) -> str:
    """도구를 안전하게 실행하고 결과를 문자열로 반환한다."""
    try:
        result = await tool.ainvoke(args, config=config)
        return str(result) if result else "결과 없음"
    except Exception as e:
        return f"도구 실행 오류: {e}"


def _get_mcp_tools_by_name(tools: list[Any]) -> dict[str, Any]:
    """도구 리스트를 이름 → 도구 매핑으로 변환한다."""
    mapping = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            mapping[name] = tool
    return mapping


async def researcher(state: ResearcherState, config: RunnableConfig) -> dict:
    """연구 주제에 대해 LLM이 도구 호출을 결정하는 노드."""
    rc = ResearchConfig.from_runnable_config(config)

    # MCP 도구 로딩
    mcp_loader = MCPToolLoader(rc.mcp_servers)
    tools = await mcp_loader.load()
    tool_descriptions = mcp_loader.get_tool_descriptions()

    # 첫 호출 시 시스템 메시지 주입
    messages = list(state.get("researcher_messages") or [])
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(
            content=RESEARCHER_SYSTEM_PROMPT.format(
                date=get_today_str(),
                research_topic=state.get("research_topic", ""),
                mcp_prompt=tool_descriptions,
            )
        )
        messages.insert(0, sys_msg)

    # 도구 바인딩
    llm = rc.get_model("research")
    tool_schemas = [ConductResearch, ResearchComplete] + tools
    llm_with_tools = llm.bind_tools(tool_schemas)

    response = await llm_with_tools.ainvoke(messages)
    return {"researcher_messages": [response]}


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> dict:
    """LLM의 도구 호출을 실행하고 결과를 반환하는 노드.

    ResearchComplete가 호출되면 compress_research로 라우팅한다.
    """
    rc = ResearchConfig.from_runnable_config(config)
    messages = state.get("researcher_messages") or []
    iterations = state.get("tool_call_iterations") or 0

    last_msg = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) or []

    if not tool_calls:
        return {"tool_call_iterations": iterations + 1}

    # MCP 도구 로딩
    mcp_loader = MCPToolLoader(rc.mcp_servers)
    tools = await mcp_loader.load()
    tools_by_name = _get_mcp_tools_by_name(tools)

    tool_messages = []

    for call in tool_calls:
        name = tc_name(call)
        call_id = tc_id(call)
        args = tc_args(call)

        if name == "ResearchComplete":
            # research_complete 신호 — tool_messages에 완료 메시지 추가로 처리
            tool_messages.append(
                ToolMessage(
                    content="연구 완료",
                    tool_call_id=call_id or "unknown",
                )
            )
        elif name == "ConductResearch":
            tool_messages.append(
                ToolMessage(
                    content=f"연구 주제 '{args.get('research_topic', '')}' 접수됨",
                    tool_call_id=call_id or "unknown",
                )
            )
        elif name in tools_by_name:
            result = await _execute_tool_safely(tools_by_name[name], args, config)
            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id or "unknown",
                )
            )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"알 수 없는 도구: {name}",
                    tool_call_id=call_id or "unknown",
                )
            )

    new_iterations = iterations + 1
    return {
        "researcher_messages": tool_messages,
        "tool_call_iterations": new_iterations,
    }


def _should_continue_research(state: ResearcherState, config: RunnableConfig) -> str:
    """연구를 계속할지 압축으로 넘어갈지 결정한다."""
    configurable = (config or {}).get("configurable", {})
    max_iters = int(configurable.get("max_react_tool_calls", 5))
    min_iters = int(configurable.get("researcher_min_iterations_before_compress", 1))

    iterations = state.get("tool_call_iterations") or 0
    messages = state.get("researcher_messages") or []

    # 최대 반복 초과
    if iterations >= max_iters:
        return "compress_research"

    # 마지막 메시지에서 ResearchComplete 확인
    last_msg = messages[-1] if messages else None
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    for call in tool_calls:
        if tc_name(call) == "ResearchComplete" and iterations >= min_iters:
            return "compress_research"

    # 도구 호출이 없으면 압축으로
    if not tool_calls and iterations > 0:
        return "compress_research"

    return "researcher"


async def compress_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """수집된 연구 결과를 요약한다."""
    rc = ResearchConfig.from_runnable_config(config)
    messages = state.get("researcher_messages") or []

    # ToolMessage와 AIMessage에서 내용 수집
    raw_notes = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.content:
            raw_notes.append(msg.content)
        elif (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            raw_notes.append(msg.content)

    if not raw_notes:
        return {"compressed_research": "연구 결과 없음", "raw_notes": []}

    findings_text = "\n\n".join(raw_notes)
    llm = rc.get_model("compression")

    try:
        response = await llm.ainvoke(
            [
                HumanMessage(
                    content=COMPRESS_RESEARCH_PROMPT.format(findings=findings_text)
                )
            ]
        )
        return {
            "compressed_research": response.content,
            "raw_notes": raw_notes,
        }
    except Exception as e:
        logger.warning(f"연구 결과 압축 실패: {e}")
        return {
            "compressed_research": findings_text[:2000],
            "raw_notes": raw_notes,
        }


def build_researcher_subgraph(context_schema: type | None = None) -> CompiledStateGraph:
    """Researcher 서브그래프를 빌드한다."""
    builder = StateGraph(
        ResearcherState,
        input_schema=ResearcherInputState,
        output_schema=ResearcherOutputState,
        context_schema=context_schema,
    )

    builder.add_node("researcher", researcher)
    builder.add_node("researcher_tools", researcher_tools)
    builder.add_node("compress_research", compress_research)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "researcher_tools")
    builder.add_conditional_edges("researcher_tools", _should_continue_research)
    builder.add_edge("compress_research", END)

    return builder.compile()
