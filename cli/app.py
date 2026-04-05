"""대화형 CLI 메인 애플리케이션.

prompt-toolkit + rich 기반 대화형 루프.
에이전트 선택, 스트리밍 출력, 슬래시 커맨드를 지원한다.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from rich.console import Console

from youngs75_a2a.cli.commands import handle_command
from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession
from youngs75_a2a.core.abort_controller import AbortController, AbortReason
from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.context_manager import ContextManager
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.parallel_tool_executor import ParallelToolExecutor
from youngs75_a2a.core.project_context import ProjectContextLoader
from youngs75_a2a.core.skills.loader import SkillLoader
from youngs75_a2a.core.skills.registry import SkillRegistry
from youngs75_a2a.core.tool_permissions import ToolPermissionManager
from youngs75_a2a.eval_pipeline.observability.callback_handler import (
    AgentMetricsCollector,
    build_observed_config,
    create_langfuse_handler,
    safe_flush,
)

logger = logging.getLogger(__name__)

# 노드 이름 → 한국어 상태 레이블
_NODE_LABELS: dict[str, str] = {
    # coding_assistant
    "parse_request": "요청 분석",
    "retrieve_memory": "메모리 검색",
    "execute_code": "도구 호출 판단 (FAST)",
    "execute_tools": "도구 실행",
    "generate_final": "코드 생성 (STRONG)",
    "verify_result": "코드 검증",
    # deep_research
    "clarify_with_user": "질문 명확화",
    "write_research_brief": "연구 브리프 작성",
    "research_supervisor": "연구 수행",
    "final_report_generation": "보고서 작성",
    "record_episodic": "연구 이력 기록",
    # simple_react
    "react_agent": "처리",
    # apply_code
    "apply_code": "파일 저장",
    # planner
    "analyze_task": "태스크 분석",
    "explore_context": "프로젝트 탐색",
    "create_plan": "구현 계획 수립",
    # orchestrator
    "classify": "요청 분류",
    "plan": "계획 수립 (Planner Agent)",
    "delegate": "에이전트 위임",
    "respond": "응답 생성",
}

# Episodic 메모리 안전장치 상수 (Agent-as-a-Judge)
_EPISODIC_MAX_ITEMS = 5
_EPISODIC_MAX_CHARS = 200


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
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
        from youngs75_a2a.agents.coding_assistant.config import CodingConfig

        return await CodingAssistantAgent.create(
            config=CodingConfig(),
            checkpointer=checkpointer,
            memory_store=memory_store,
            skill_registry=skill_registry,
        )

    if name == "deep_research":
        from youngs75_a2a.agents.deep_research.agent import DeepResearchAgent
        from youngs75_a2a.agents.deep_research.config import ResearchConfig

        return await DeepResearchAgent.create(
            config=ResearchConfig(), checkpointer=checkpointer
        )

    if name == "simple_react":
        from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent
        from youngs75_a2a.agents.simple_react.config import SimpleReActConfig

        return await SimpleMCPReActAgent.create(
            config=SimpleReActConfig(), checkpointer=checkpointer
        )

    if name == "orchestrator":
        from youngs75_a2a.agents.orchestrator.agent import OrchestratorAgent
        from youngs75_a2a.agents.orchestrator.config import OrchestratorConfig

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
            return f"[{selected}] {agent_response}"
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
    passed = True
    token_streamed = False
    try:
        async for event in agent.graph.astream_events(
            input_state,
            config={**run_config, "recursion_limit": 50},
            version="v2",
        ):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # 노드 시작 시 스피너로 진행 상태 표시
            if kind == "on_chain_start" and node in _NODE_LABELS and node != last_node:
                if token_streamed:
                    renderer.flush_tokens()
                    token_streamed = False
                renderer.start_progress(_NODE_LABELS[node])
                metrics.record_node_start(node)
                last_node = node

            # LLM 토큰 스트리밍 및 사용량 수집
            # 사용자에게 보여줄 노드만 허용 (inclusion list)
            # - react_agent: simple_react 에이전트 응답
            # - respond: orchestrator 최종 응답
            # generate_final은 숨김 — 코드 전문 대신 apply_code에서 파일 요약 표시
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
                if chunk and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
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
                        tool_name, tool_input if isinstance(tool_input, dict) else None
                    )

            # 노드 완료 시 응답 데이터 수집
            elif kind == "on_chain_end" and node:
                metrics.record_node_end(node)
                output = event["data"].get("output")
                if isinstance(output, dict):
                    for key in (
                        "generated_code",
                        "verify_result",
                        "final_report",
                        "selected_agent",
                        "agent_response",
                        "exit_reason",
                        "written_files",
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

        if token_streamed:
            renderer.flush_tokens()

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

    renderer.welcome(session.info.agent_name)
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
    from dotenv import load_dotenv

    load_dotenv()
    config = config or CLIConfig()
    asyncio.run(_main_loop(config))
