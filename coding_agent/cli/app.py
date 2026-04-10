"""대화형 CLI 메인 애플리케이션.

prompt-toolkit + rich 기반 대화형 루프.
에이전트 선택, 스트리밍 출력, 슬래시 커맨드를 지원한다.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.console import Console

from coding_agent.cli.commands import handle_command
from coding_agent.cli.config import CLIConfig
from coding_agent.cli.renderer import CLIRenderer
from coding_agent.cli.session import CLISession
from coding_agent.core.abort_controller import AbortController, AbortReason
from coding_agent.core.base_agent import BaseGraphAgent
from coding_agent.core.context_manager import ContextManager
from coding_agent.core.memory.schemas import MemoryItem, MemoryType
from coding_agent.core.parallel_tool_executor import ParallelToolExecutor
from coding_agent.core.project_context import ProjectContextLoader
from coding_agent.core.skills.loader import SkillLoader
from coding_agent.core.skills.registry import SkillRegistry
from coding_agent.core.tool_permissions import ToolPermissionManager
from coding_agent.eval_pipeline.observability.callback_handler import (
    AgentMetricsCollector,
    build_observed_config,
    create_langfuse_handler,
    safe_flush,
)

logger = logging.getLogger(__name__)

# 노드 이름 → 한국어 상태 레이블 (모델명 동적 주입)
def _build_node_labels() -> dict[str, str]:
    """실제 사용 모델명을 포함한 노드 레이블을 생성한다."""
    try:
        from coding_agent.core.model_tiers import ModelTier, build_default_tiers

        tiers = build_default_tiers()
        fast_model = tiers.get(ModelTier.FAST)
        strong_model = tiers.get(ModelTier.STRONG)
        default_model = tiers.get(ModelTier.DEFAULT)
        fast_name = fast_model.model if fast_model else "FAST"
        strong_name = strong_model.model if strong_model else "STRONG"
        default_name = default_model.model if default_model else "DEFAULT"
    except Exception:
        fast_name, strong_name, default_name = "FAST", "STRONG", "DEFAULT"

    return {
        # coding_assistant — "티어 - 모델명" 형식
        "parse_request": f"요청 분석 (FAST - {fast_name})",
        "retrieve_memory": "메모리 검색",
        "execute_code": f"도구 호출 판단 (FAST - {fast_name})",
        "execute_tools": "도구 실행",
        "generate_final": f"코드 생성 (STRONG - {strong_name})",
        "verify_result": f"코드 검증 (DEFAULT - {default_name})",
    }


_STATIC_NODE_LABELS: dict[str, str] = {
    # deep_research
    "clarify_with_user": "질문 명확화",
    "write_research_brief": "연구 브리프 작성",
    "research_supervisor": "연구 수행",
    "final_report_generation": "보고서 작성",
    "record_episodic": "연구 이력 기록",
    # simple_react
    "react_agent": "처리",
    # apply_code / run_tests
    "apply_code": "파일 저장",
    "run_tests": "테스트 실행",
    # planner
    "analyze_task": "태스크 분석",
    "research_external": "외부 API 조사",
    "explore_context": "프로젝트 탐색",
    "create_plan": "구현 계획 수립",
    # orchestrator
    "classify": "요청 분류",
    "plan": "계획 수립 (Planner Agent)",
    "delegate": "에이전트 위임",
    "respond": "응답 생성",
}

# 동적(모델명) + 정적 노드 레이블 통합
_NODE_LABELS: dict[str, str] = {**_build_node_labels(), **_STATIC_NODE_LABELS}

# Episodic 메모리 안전장치 상수 (Agent-as-a-Judge)
_EPISODIC_MAX_ITEMS = 5
_EPISODIC_MAX_CHARS = 200


def _extract_node_summary(node: str, output: dict) -> str:
    """노드 완료 시 간략 요약을 추출한다."""
    if not isinstance(output, dict):
        return ""
    if node == "classify":
        selected = output.get("selected_agent", "")
        return f"→ {selected}" if selected else ""
    if node == "plan":
        plan = output.get("task_plan")
        if plan:
            return f"계획 수립 완료 ({len(plan)}자)"
        return "계획 없이 진행"
    if node == "generate_final":
        code = output.get("generated_code", "")
        if code:
            lines = code.strip().count("\n") + 1
            return f"{lines} lines 생성"
        return ""
    if node == "verify_result":
        verify = output.get("verify_result", {})
        if isinstance(verify, dict):
            return "통과" if verify.get("passed", True) else "이슈 발견"
        return ""
    if node == "apply_code":
        files = output.get("written_files", [])
        if files:
            return f"{len(files)}개 파일 저장"
        return ""
    if node == "run_tests":
        passed = output.get("test_passed", True)
        logs = output.get("execution_log", [])
        # 환경 설정 로그 요약
        env_info = []
        for entry in logs:
            if "[env]" in entry and "✓" in entry:
                env_info.append("venv")
            if "[install_deps]" in entry and "✓" in entry:
                env_info.append("deps")
            if "[runtime]" in entry and "✗" in entry:
                env_info.append("런타임 미설치")
        env_summary = f" ({'+'.join(dict.fromkeys(env_info))})" if env_info else ""
        if passed:
            return f"통과{env_summary}"
        test_out = output.get("test_output", "")
        if test_out:
            # pytest의 "E   " 라인 (실제 에러)을 우선 표시, 그 다음 Error/FAILED 키워드
            error_line = ""
            fallback_line = ""
            for line in test_out.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                # "E   실제에러메시지" 라인이 가장 유용 (pytest assertion/exception)
                if stripped.startswith("E ") and not error_line:
                    error_line = stripped[:80]
                # "FAILED" pytest 요약 라인
                elif stripped.startswith("FAILED") and not error_line:
                    error_line = stripped[:80]
                # "Error" 키워드 포함 줄은 fallback
                elif ("Error" in stripped or "error" in stripped) and not fallback_line:
                    fallback_line = stripped[:80]
                # "[syntax]" 또는 "[js-syntax]" 오류
                elif stripped.startswith("[syntax]") or stripped.startswith("[js-syntax]") or stripped.startswith("[ts-check]"):
                    if not error_line:
                        error_line = stripped[:80]
            if not error_line:
                # 첫 번째 비어있지 않은 줄 사용
                first_line = next((line.strip() for line in test_out.split("\n") if line.strip()), "")
                error_line = fallback_line or first_line[:80] or "상세 로그 없음"
        else:
            error_line = "상세 로그 없음"
        return f"실패{env_summary} — {error_line}"
    if node == "research_external":
        research = output.get("research_context", [])
        if research:
            # 검색 쿼리 키워드만 추출하여 간략 요약 ([조사 요약] 제외)
            queries = []
            for entry in research:
                if entry.startswith("[웹 검색: "):
                    q = entry.split("]")[0].replace("[웹 검색: ", "")
                    queries.append(q[:30])
            search_count = len(queries)
            return f"{search_count}건 조사 ({', '.join(queries[:3])})" if queries else "조사 완료"
        return "조사 결과 없음"
    if node == "delegate":
        phase_results = output.get("phase_results")
        if phase_results:
            parts = []
            for pr in phase_results:
                status = pr.get("status", "")
                icon = {"success": "✓", "failed": "✗", "skipped": "⊘"}.get(status, "?")
                n_files = len(pr.get("written_files", []))
                title = pr.get("title", "")[:30]
                parts.append(f"{icon} {pr.get('phase_id', '?')}: {title} {n_files}파일")
            return " | ".join(parts)
        resp = output.get("agent_response", "")
        if resp:
            return f"응답 수신 ({len(resp)}자)"
        return ""
    return ""


def _truncate(text: str, max_chars: int = _EPISODIC_MAX_CHARS) -> str:
    """텍스트를 최대 길이로 잘라낸다."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_episodic_summary(
    user_input: str,
    passed: bool,
    response_summary: str,
) -> str:
    """에피소딕 메모리에 저장할 요약 문자열을 생성한다."""
    status = "passed" if passed else "[주의] failed"
    request_brief = _truncate(user_input, 60)
    summary = f"[{status}] 요청: {request_brief} | 결과: {response_summary}"
    return _truncate(summary)


def _save_episodic_memory(
    session: CLISession,
    user_input: str,
    response: str,
    passed: bool = True,
) -> None:
    """실행 결과를 에피소딕 메모리로 저장한다."""
    response_brief = _truncate(response, 80)
    summary = _build_episodic_summary(user_input, passed, response_brief)

    tags = ["episodic"]
    if not passed:
        tags.append("주의")

    item = MemoryItem(
        type=MemoryType.EPISODIC,
        content=summary,
        tags=tags,
        session_id=session.session_id,
    )
    session.memory.put(item)


async def _create_agent(
    name: str,
    checkpointer: Any | None = None,
    memory_store: Any | None = None,
    skill_registry: SkillRegistry | None = None,
) -> BaseGraphAgent:
    """에이전트를 비동기로 생성한다."""
    if name == "coding_assistant":
        from coding_agent.agents.coding_assistant.agent import CodingAssistantAgent
        from coding_agent.agents.coding_assistant.config import CodingConfig

        return await CodingAssistantAgent.create(
            config=CodingConfig(),
            checkpointer=checkpointer,
            memory_store=memory_store,
            skill_registry=skill_registry,
        )

    if name == "deep_research":
        from coding_agent.agents.deep_research.agent import DeepResearchAgent
        from coding_agent.agents.deep_research.config import ResearchConfig

        return await DeepResearchAgent.create(
            config=ResearchConfig(), checkpointer=checkpointer
        )

    if name == "simple_react":
        from coding_agent.agents.simple_react.agent import SimpleMCPReActAgent
        from coding_agent.agents.simple_react.config import SimpleReActConfig

        return await SimpleMCPReActAgent.create(
            config=SimpleReActConfig(), checkpointer=checkpointer
        )

    if name == "orchestrator":
        from coding_agent.agents.orchestrator.agent import OrchestratorAgent
        from coding_agent.agents.orchestrator.config import OrchestratorConfig

        return await OrchestratorAgent.create(
            config=OrchestratorConfig(), checkpointer=checkpointer
        )

    raise ValueError(f"알 수 없는 에이전트: {name}")


async def _get_or_create_agent(
    session: CLISession, renderer: CLIRenderer
) -> BaseGraphAgent | None:
    """세션에서 에이전트를 가져오거나 새로 생성한다.

    Phase 10 통합: 생성된 에이전트에 세션의 Phase 10 기능을 주입한다.
    - project_context: 프로젝트 컨텍스트 문자열
    - permission_manager: 도구 실행 권한 관리자
    - tool_executor: 병렬 도구 실행기
    - context_manager: 컨텍스트 윈도우 관리자
    """
    name = session.info.agent_name
    cached = session.get_cached_agent(name)
    if cached is not None:
        return cached

    renderer.start_progress(f"{name} 에이전트 초기화")
    try:
        agent = await _create_agent(
            name,
            checkpointer=session.checkpointer,
            memory_store=session.memory,
            skill_registry=session.skills,
        )

        # Phase 10 기능 주입 — 옵셔널 (세션에 설정된 경우에만)
        if session.project_context:
            agent.project_context = session.project_context
        if session.permission_manager:
            agent.permission_manager = session.permission_manager
        if session.tool_executor:
            agent.tool_executor = session.tool_executor
        # ContextManager는 CodingAssistantAgent에 이미 내장되어 있지만,
        # 다른 에이전트에도 기본 인스턴스를 주입한다
        if agent.context_manager is None:
            agent.context_manager = ContextManager()

        session.cache_agent(name, agent)
        renderer.stop_progress_with_result(f"{name} 준비 완료")
        return agent
    except Exception as e:
        renderer.stop_progress_with_result(f"{name} 초기화 실패", success=False)
        logger.exception("에이전트 초기화 실패")
        renderer.error(f"{e}")
        return None


def _build_input_state(
    agent_name: str,
    user_input: str,
    session: CLISession,
) -> dict[str, Any]:
    """에이전트별 입력 상태를 구성한다."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=user_input)],
    }
    if agent_name == "coding_assistant":
        state["iteration"] = 0
        state["max_iterations"] = 3
        # Semantic Memory 주입
        semantic_items = session.memory.list_by_type(MemoryType.SEMANTIC)
        if semantic_items:
            state["semantic_context"] = [item.content for item in semantic_items]
        # Skills 컨텍스트 주입
        skill_entries = session.skills.get_context_entries()
        if skill_entries:
            state["skill_context"] = skill_entries
        # Episodic Memory 주입 — 세션 스코프, 최대 5개
        episodic_items = session.memory.list_by_type(
            MemoryType.EPISODIC,
            session_id=session.session_id,
        )
        recent = episodic_items[:_EPISODIC_MAX_ITEMS]
        if recent:
            state["episodic_log"] = [item.content for item in recent]
        # Procedural Memory 주입 — 학습된 코드 패턴 (Voyager식 누적)
        procedural_items = session.memory.retrieve_skills(
            query=user_input,
            limit=3,
        )
        if procedural_items:
            state["procedural_skills"] = [item.content for item in procedural_items]

    elif agent_name == "orchestrator":
        state["selected_agent"] = None
        state["agent_response"] = None

    return state


_CODE_BLOCK_RE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_FILEPATH_LINE_RE = re.compile(
    r"^\s*(?:#\s*|<!--\s*|\*\*)?filepath:\s*\S+.*$", re.MULTILINE
)
_SAVED_FILES_SECTION_RE = re.compile(r"\n*📁 저장된 파일[:\s].*$", re.DOTALL)
_CONSECUTIVE_NEWLINES_RE = re.compile(r"\n{3,}")


def _strip_code_blocks(text: str) -> str:
    """마크다운 코드 블록, filepath 헤더, 중복 파일 목록을 제거한다."""
    result = _CODE_BLOCK_RE.sub("", text)
    result = _FILEPATH_LINE_RE.sub("", result)
    result = _SAVED_FILES_SECTION_RE.sub("", result)
    return _CONSECUTIVE_NEWLINES_RE.sub("\n\n", result).strip()


def _extract_response(agent_name: str, data: dict[str, Any]) -> str:
    """에이전트별 최종 응답을 추출한다."""
    if agent_name == "coding_assistant":
        # 코드 전문 대신 간략 요약만 반환 (파일 목록은 files_written으로 별도 표시)
        written = data.get("written_files", [])
        if written:
            return f"{len(written)}개 파일이 생성되었습니다."
        # 파일 저장 안 된 경우 generated_code의 첫 부분만 표시
        code = data.get("generated_code", "")
        if code:
            lines = code.strip().split("\n")
            preview = "\n".join(lines[:5])
            if len(lines) > 5:
                preview += f"\n... (+{len(lines) - 5} lines)"
            return preview
        return data.get("last_ai_message", "응답을 생성하지 못했습니다.")

    if agent_name == "deep_research":
        return data.get("final_report") or data.get(
            "last_ai_message", "보고서를 생성하지 못했습니다."
        )

    if agent_name == "orchestrator":
        # orchestrator: agent_response 또는 마지막 AI 메시지
        agent_response = data.get("agent_response")
        if agent_response:
            selected = data.get("selected_agent", "unknown")
            # 코드 블록 제거 — 파일 저장 목록으로 대체됨
            cleaned = _strip_code_blocks(agent_response)
            return f"[{selected}] {cleaned}"
        return data.get("last_ai_message", "응답을 생성하지 못했습니다.")

    # simple_react 및 기타
    return data.get("last_ai_message", "응답을 생성하지 못했습니다.")


async def _run_agent_turn(
    user_input: str,
    session: CLISession,
    renderer: CLIRenderer,
    *,
    langfuse_handler: Any | None = None,
) -> None:
    """에이전트에 메시지를 전달하고 응답을 출력한다.

    astream_events(v2) 기반 토큰 단위 실시간 스트리밍을 사용한다.
    - on_chain_start: 노드 진행 상태 표시 (_NODE_LABELS)
    - on_chat_model_stream: LLM 토큰 실시간 출력
    - on_chain_end: 응답 데이터 수집

    Langfuse 콜백 핸들러가 주어지면 트레이스/메트릭을 자동 수집합니다.
    """
    session.add_message("user", user_input)
    renderer.start_turn()

    agent = await _get_or_create_agent(session, renderer)
    if agent is None:
        return

    input_state = _build_input_state(session.info.agent_name, user_input, session)

    # Langfuse 관측성이 적용된 실행 config 구성
    run_config = build_observed_config(
        handler=langfuse_handler,
        session_id=session.session_id,
        thread_id=session.thread_id,
        agent_name=session.info.agent_name,
    )

    # 메트릭 수집기 초기화
    metrics = AgentMetricsCollector(agent_name=session.info.agent_name)

    # 협력적 중단 제어 (Claude Code AbortController 패턴)
    abort_controller = AbortController()
    agent.abort_controller = abort_controller

    response_data: dict[str, Any] = {}
    last_node = ""
    node_summaries: dict[str, str] = {}
    passed = True
    token_streamed = False
    graph_input: Any = input_state
    stream_config = {**run_config, "recursion_limit": 50}
    try:
        # Human-in-the-loop 루프: interrupt 발생 시 승인 후 resume
        while True:
            async for event in agent.graph.astream_events(
                graph_input,
                config=stream_config,
                version="v2",
            ):
                kind = event["event"]
                node = event.get("metadata", {}).get("langgraph_node", "")

                # 노드 시작 시 이전 노드 완료 표시 + 새 스피너
                if (
                    kind == "on_chain_start"
                    and node in _NODE_LABELS
                    and node != last_node
                ):
                    if token_streamed:
                        renderer.flush_tokens()
                        token_streamed = False
                    # 이전 노드를 영구 완료 표시
                    if last_node and last_node in _NODE_LABELS:
                        summary = node_summaries.get(last_node, "")
                        renderer.complete_node(_NODE_LABELS[last_node], summary)
                    renderer.start_progress(_NODE_LABELS[node])
                    metrics.record_node_start(node)
                    last_node = node

                # LLM 토큰 스트리밍 및 사용량 수집
                # 사용자에게 보여줄 노드만 허용 (inclusion list)
                # - react_agent: simple_react 에이전트 응답
                # - respond: orchestrator 최종 응답
                # generate_final은 숨김 — 코드 전문 대신 파일 요약 표시
                elif kind in ("on_chat_model_stream", "on_llm_stream"):
                    chunk = event["data"].get("chunk")
                    if (
                        chunk
                        and hasattr(chunk, "content")
                        and chunk.content
                        and node in ("react_agent", "respond")
                    ):
                        if not token_streamed:
                            renderer.start_token_stream()
                            token_streamed = True
                        renderer.render_token(chunk.content)
                    # 토큰 사용량 수집 (usage_metadata가 있는 경우)
                    if (
                        chunk
                        and hasattr(chunk, "usage_metadata")
                        and chunk.usage_metadata
                    ):
                        usage = chunk.usage_metadata
                        metrics.record_llm_tokens(
                            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                        )

                # 도구 호출 표시
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    if tool_name:
                        renderer.tool_call(
                            tool_name,
                            tool_input if isinstance(tool_input, dict) else None,
                        )

                # 도구 실행 완료 — 에러 시 인라인 표시
                elif kind == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    tool_out_str = str(tool_output) if tool_output else ""
                    if tool_out_str and (
                        "error" in tool_out_str.lower()[:200]
                        or "Error" in tool_out_str[:200]
                        or "FAILED" in tool_out_str[:200]
                    ):
                        # 에러 결과만 축약 표시 (정상 결과는 표시하지 않음)
                        first_line = tool_out_str.strip().split("\n")[0][:80]
                        renderer.console.print(
                            f"    [dim italic]↳ {first_line}[/]"
                        )

                # 노드 완료 시 응답 데이터 수집 + 요약 추출
                elif kind == "on_chain_end" and node:
                    metrics.record_node_end(node)
                    output = event["data"].get("output")
                    # 노드 요약 추출
                    if isinstance(output, dict) and node in _NODE_LABELS:
                        summary = _extract_node_summary(node, output)
                        if summary:
                            node_summaries[node] = summary
                    if isinstance(output, dict):
                        for key in (
                            "generated_code",
                            "verify_result",
                            "final_report",
                            "selected_agent",
                            "agent_response",
                            "exit_reason",
                            "written_files",
                            "phase_results",
                        ):
                            if key in output:
                                response_data[key] = output[key]
                        if "messages" in output:
                            for msg in output["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    response_data["last_ai_message"] = msg.content

                        # 스킬 자동 활성화 표시 (parse 노드 완료 시만)
                        if node == "parse_request":
                            exec_log = output.get("execution_log", [])
                            for entry in exec_log:
                                if "[skills] 자동 활성화:" in entry:
                                    names = entry.split("자동 활성화: ", 1)[-1]
                                    renderer.skill_activated(names.split(", "))

                        # 서브에이전트 위임 표시 (orchestrator)
                        selected = output.get("selected_agent")
                        if selected and node in ("classify", "delegate"):
                            renderer.subagent_delegate(selected)

            # 이벤트 스트림 종료 — 마지막 노드 완료 표시
            if token_streamed:
                renderer.flush_tokens()
                token_streamed = False
            if last_node and last_node in _NODE_LABELS:
                summary = node_summaries.get(last_node, "")
                renderer.complete_node(_NODE_LABELS[last_node], summary)
                last_node = ""

            # Human-in-the-loop: interrupt 감지 (aget_state 기반)
            try:
                graph_state = await agent.graph.aget_state(stream_config)
                if graph_state.next and graph_state.tasks:
                    interrupt_value = None
                    for task in graph_state.tasks:
                        if task.interrupts:
                            interrupt_value = task.interrupts[0].value
                            break
                    if interrupt_value:
                        # 환경 승인 interrupt vs 계획 승인 interrupt 구분
                        is_env_approval = (
                            isinstance(interrupt_value, dict)
                            and interrupt_value.get("type") == "env_approval"
                        )

                        if is_env_approval:
                            # 환경 설정 승인 요청
                            renderer.show_env_approval(interrupt_value)
                            approved = await asyncio.to_thread(renderer.ask_env_approval)
                            if approved:
                                renderer.success("환경 설정 승인 — 테스트 환경을 구성합니다")
                                graph_input = Command(resume=True)
                            else:
                                renderer.warning("환경 설정 거부 — LLM 검증만 진행합니다")
                                graph_input = Command(resume=False)
                            continue
                        else:
                            # 기존 계획 승인 interrupt
                            renderer.show_plan(str(interrupt_value))
                            approved = await asyncio.to_thread(renderer.ask_plan_approval)
                            if approved:
                                renderer.success("계획 승인 — 실행을 시작합니다")
                                graph_input = Command(resume=True)
                                continue
                            else:
                                # 거부 시 피드백을 받아 재계획
                                feedback = await asyncio.to_thread(renderer.ask_rejection_feedback)
                                if feedback:
                                    renderer.info(f"재계획 요청: {feedback}")
                                    graph_input = Command(resume={"approved": False, "feedback": feedback})
                                    continue
                                else:
                                    renderer.warning("계획 거부 — 작업을 취소합니다")
                                    return
            except Exception:
                logger.debug("그래프 상태 확인 스킵 (checkpointer 미지원)")
            break  # 정상 완료 (interrupt 없음)

    except (asyncio.CancelledError, KeyboardInterrupt):
        abort_controller.abort(AbortReason.USER_INTERRUPT)
        if token_streamed:
            renderer.flush_tokens()
        renderer._stop_progress()
        renderer.warning(abort_controller.message)
        return
    except Exception as e:
        if token_streamed:
            renderer.flush_tokens()
        renderer._stop_progress()
        metrics.record_error()
        logger.exception("에이전트 실행 오류")
        renderer.error(f"에이전트 실행 오류: {e}")
        return
    finally:
        renderer._stop_progress()
        # 메트릭 수집 완료 및 Langfuse flush
        metrics.finalize()
        if langfuse_handler is not None:
            logger.debug("에이전트 메트릭: %s", metrics.to_dict())
            safe_flush()

    # 안전장치 발동 시 사용자에게 알림
    exit_reason = response_data.get("exit_reason", "")
    if exit_reason:
        _EXIT_REASON_MESSAGES = {
            "stall_detected": "반복 도구 호출이 감지되어 루프를 탈출했습니다.",
            "budget_exceeded": "LLM 호출 예산을 초과하여 실행을 중단했습니다.",
            "turn_limit": "도구 호출 한도에 도달하여 실행을 중단했습니다.",
        }
        msg = _EXIT_REASON_MESSAGES.get(exit_reason, f"실행 중단: {exit_reason}")
        renderer.warning(msg)

    # 검증 결과 표시 (새 렌더러 API)
    verify = response_data.get("verify_result")
    if isinstance(verify, dict):
        passed = verify.get("passed", True)
        renderer.verify_result(
            passed=passed,
            issues=verify.get("issues"),
            suggestions=verify.get("suggestions"),
        )

    # 파일 저장 결과 표시
    written_files = response_data.get("written_files", [])
    if written_files:
        renderer.files_written(written_files)

    response = _extract_response(session.info.agent_name, response_data)

    # 토큰 스트리밍이 된 경우 이미 출력되었으므로 중복 출력 방지
    if not token_streamed:
        renderer.agent_message(response)

    # 소요시간 + 컨텍스트 상태 표시
    renderer.end_turn()
    if metrics.total_prompt_tokens > 0:
        # 에이전트의 context_manager에서 max_tokens 가져오기
        max_ctx = 128_000
        compact_thresh = 0.8
        if agent and agent.context_manager:
            max_ctx = agent.context_manager.max_tokens
            compact_thresh = agent.context_manager.compact_threshold
        renderer.context_status(
            input_tokens=metrics.total_prompt_tokens,
            output_tokens=metrics.total_completion_tokens,
            max_tokens=max_ctx,
            compact_threshold=compact_thresh,
        )

    session.add_message("assistant", response)

    # Episodic Memory 저장
    _save_episodic_memory(session, user_input, response, passed=passed)


_local_mcp_process: subprocess.Popen | None = None


async def _sync_mcp_workspace(workspace: str, renderer: CLIRenderer) -> None:
    """MCP 서버의 workspace를 CLI의 현재 workspace와 동기화한다.

    1차: Docker MCP의 set_workspace 호출 시도
    2차: 실패 시 로컬 MCP 서버를 자동 시작 (CLI 모드용)
    """
    import subprocess
    import time

    from coding_agent.agents.coding_assistant.config import CodingConfig

    config = CodingConfig()
    mcp_url = config.mcp_servers.get("code_tools", "")
    if not mcp_url:
        return

    # 1차: 기존 MCP 서버에 set_workspace 시도
    try:
        from coding_agent.core.mcp_loader import MCPToolLoader

        loader = MCPToolLoader({"code_tools": mcp_url})
        tools = await loader.load()
        set_ws = next((t for t in tools if t.name == "set_workspace"), None)
        if set_ws:
            result = await set_ws.ainvoke({"path": workspace})
            result_text = result
            if isinstance(result, list):
                result_text = result[0].get("text", "") if result else ""
            if "변경됨" in str(result_text):
                renderer.success(f"MCP workspace 동기화: {workspace}")
                return
            # Docker mount root 제한으로 실패 → 로컬 MCP 시작
            logger.info("Docker MCP set_workspace 거부: %s → 로컬 MCP 시작", result_text)
    except Exception as e:
        logger.info("MCP 서버 미응답: %s → 로컬 MCP 시작", e)

    # 2차: 로컬 MCP 서버 자동 시작
    global _local_mcp_process
    local_port = 3013  # Docker MCP(3003)와 충돌 방지
    local_url = f"http://localhost:{local_port}/mcp"

    env = {**os.environ, "CODE_TOOLS_WORKSPACE": workspace, "CODE_TOOLS_PORT": str(local_port)}
    try:
        # server.py의 if __name__ == "__main__" 블록으로 직접 실행
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "mcp_servers", "code_tools", "server.py",
        )
        _local_mcp_process = subprocess.Popen(
            [sys.executable, server_path],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # 서버 시작 대기
        for _ in range(10):
            time.sleep(0.3)
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{local_port}/", timeout=1)
            except Exception:
                continue
            break

        # CodingConfig의 MCP URL을 로컬로 전환
        os.environ["CODE_TOOLS_MCP_URL"] = local_url
        renderer.success(f"로컬 MCP 서버 시작 (:{local_port}, workspace: {workspace})")
    except Exception as e:
        logger.warning("로컬 MCP 서버 시작 실패: %s", e)


def _init_skill_registry(
    skills_dir: str | None,
) -> tuple[SkillRegistry, list[str]]:
    """스킬 레지스트리를 초기화한다.

    Returns:
        (레지스트리, 발견된 스킬 이름 목록)
    """
    discovered: list[str] = []
    registry = SkillRegistry()
    if skills_dir:
        loader = SkillLoader(skills_dir)
        registry = SkillRegistry(loader=loader)
        discovered = registry.discover()
        if discovered:
            logger.info("스킬 %d개 발견: %s", len(discovered), discovered)
    return registry, discovered


def _create_checkpointer(config: CLIConfig) -> Any:
    """설정에 따라 체크포인터를 생성한다."""
    if config.checkpointer_backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            return SqliteSaver.from_conn_string(config.checkpointer_sqlite_path)
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite 미설치, MemorySaver로 대체")

    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


async def _main_loop(config: CLIConfig) -> None:
    """대화형 메인 루프."""
    console = Console()
    renderer = CLIRenderer(console)
    skill_registry, discovered_skills = _init_skill_registry(config.skills_dir)
    checkpointer = _create_checkpointer(config)
    session = CLISession(
        agent_name=config.default_agent,
        skill_registry=skill_registry,
        checkpointer=checkpointer,
    )

    # Phase 10: 프로젝트 컨텍스트 + 권한 + 병렬 실행기 초기화
    workspace = os.getenv("CODE_TOOLS_WORKSPACE", os.getcwd())

    # MCP 서버 workspace 동기화 (Docker MCP 사용 시 호스트 경로로 설정)
    await _sync_mcp_workspace(workspace, renderer)

    display_workspace = workspace

    # 프로젝트 컨텍스트 로더
    context_loader = ProjectContextLoader(workspace)
    context_section = context_loader.build_system_prompt_section()
    if context_section:
        session.project_context = context_section
        renderer.success("프로젝트 컨텍스트 로드 완료")
        logger.info("프로젝트 컨텍스트 파일 %d개 발견", len(context_loader.discover()))

    # 도구 권한 관리자
    session.permission_manager = ToolPermissionManager(workspace)
    logger.info("도구 권한 관리자 초기화 (workspace: %s)", workspace)

    # 병렬 도구 실행기
    session.tool_executor = ParallelToolExecutor()
    logger.info("병렬 도구 실행기 초기화 완료")

    # Langfuse 콜백 핸들러 초기화 (설정으로 on/off 가능)
    langfuse_handler = None
    if config.langfuse_enabled:
        langfuse_handler = create_langfuse_handler()
        if langfuse_handler is not None:
            renderer.success("Langfuse 관측성 활성화됨")

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(config.history_file),
    )

    renderer.welcome(session.info.agent_name, workspace=display_workspace)
    if discovered_skills:
        renderer.success(
            f"스킬 {len(discovered_skills)}개 로드: {', '.join(discovered_skills)}"
        )

    while True:
        try:
            prompt_text = ANSI(
                f"\x1b[1;34m[{session.info.agent_name}]\x1b[0m \x1b[36m❯\x1b[0m "
            )
            user_input = await asyncio.to_thread(
                prompt_session.prompt,
                prompt_text,
            )
        except (EOFError, KeyboardInterrupt):
            renderer.system_message("\n세션을 종료합니다.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # 슬래시 커맨드 처리
        result = handle_command(user_input, session, renderer)
        if result.should_quit:
            renderer.system_message(result.message)
            break
        if result.handled:
            continue

        # 에이전트 메시지 처리 (Langfuse 콜백 핸들러 자동 주입)
        await _run_agent_turn(
            user_input,
            session,
            renderer,
            langfuse_handler=langfuse_handler,
        )


def run_cli(config: CLIConfig | None = None) -> None:
    """CLI 진입점."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Youngs75 Coding AI Agent")
    parser.add_argument(
        "--server", action="store_true", help="A2A 서버 모드로 시작"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="서버 호스트 (기본: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="서버 포트 (기본: 8000)"
    )
    args = parser.parse_args()

    if args.server:
        from coding_agent.a2a.executor import LGAgentExecutor
        from coding_agent.a2a.server import run_server
        from coding_agent.agents.orchestrator.agent import OrchestratorAgent

        agent = OrchestratorAgent(auto_build=True)
        executor = LGAgentExecutor(graph=agent.graph)
        run_server(
            executor=executor,
            name="youngs75-coding-agent",
            description="Youngs75 Coding AI Agent (A2A)",
            host=args.host,
            port=args.port,
        )
        return

    config = config or CLIConfig()
    try:
        asyncio.run(_main_loop(config))
    finally:
        # 로컬 MCP 서버 정리
        global _local_mcp_process
        if _local_mcp_process is not None:
            _local_mcp_process.terminate()
            _local_mcp_process.wait(timeout=5)
            _local_mcp_process = None
