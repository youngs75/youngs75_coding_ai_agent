"""대화형 CLI 메인 애플리케이션.

prompt-toolkit + rich 기반 대화형 루프.
에이전트 선택, 스트리밍 출력, 슬래시 커맨드를 지원한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console

from youngs75_a2a.cli.commands import handle_command
from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession
from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.skills.loader import SkillLoader
from youngs75_a2a.core.skills.registry import SkillRegistry
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
    "parse_request": "요청 분석 중...",
    "execute_code": "코드 생성 중...",
    "execute_tools": "도구 실행 중...",
    "verify_result": "코드 검증 중...",
    # deep_research
    "clarify_with_user": "질문 명확화...",
    "write_research_brief": "연구 브리프 작성...",
    "research_supervisor": "연구 수행 중...",
    "final_report_generation": "보고서 작성 중...",
    # simple_react
    "react_agent": "처리 중...",
    # orchestrator
    "classify": "요청 분류 중...",
    "delegate": "에이전트 위임 중...",
    "respond": "응답 생성 중...",
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
) -> BaseGraphAgent:
    """에이전트를 비동기로 생성한다."""
    if name == "coding_assistant":
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
        from youngs75_a2a.agents.coding_assistant.config import CodingConfig

        return await CodingAssistantAgent.create(
            config=CodingConfig(),
            checkpointer=checkpointer,
            memory_store=memory_store,
        )

    if name == "deep_research":
        from youngs75_a2a.agents.deep_research.agent import DeepResearchAgent
        from youngs75_a2a.agents.deep_research.config import ResearchConfig

        return await DeepResearchAgent.create(config=ResearchConfig(), checkpointer=checkpointer)

    if name == "simple_react":
        from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent
        from youngs75_a2a.agents.simple_react.config import SimpleReActConfig

        return await SimpleMCPReActAgent.create(config=SimpleReActConfig(), checkpointer=checkpointer)

    if name == "orchestrator":
        from youngs75_a2a.agents.orchestrator.agent import OrchestratorAgent
        from youngs75_a2a.agents.orchestrator.config import OrchestratorConfig

        return await OrchestratorAgent.create(config=OrchestratorConfig(), checkpointer=checkpointer)

    raise ValueError(f"알 수 없는 에이전트: {name}")


async def _get_or_create_agent(
    session: CLISession, renderer: CLIRenderer
) -> BaseGraphAgent | None:
    """세션에서 에이전트를 가져오거나 새로 생성한다."""
    name = session.info.agent_name
    cached = session.get_cached_agent(name)
    if cached is not None:
        return cached

    renderer.system_message(f"[{name}] 에이전트 초기화 중...")
    try:
        agent = await _create_agent(
            name,
            checkpointer=session.checkpointer,
            memory_store=session.memory,
        )
        session.cache_agent(name, agent)
        renderer.system_message(f"[{name}] 에이전트 준비 완료")
        return agent
    except Exception as e:
        logger.exception("에이전트 초기화 실패")
        renderer.error(f"에이전트 초기화 실패: {e}")
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
            MemoryType.EPISODIC, session_id=session.session_id,
        )
        recent = episodic_items[:_EPISODIC_MAX_ITEMS]
        if recent:
            state["episodic_log"] = [item.content for item in recent]
        # Procedural Memory 주입 — 학습된 코드 패턴 (Voyager식 누적)
        procedural_items = session.memory.retrieve_skills(
            query=user_input, limit=3,
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
        parts: list[str] = []
        code = data.get("generated_code", "")
        if code:
            parts.append(code)
        verify = data.get("verify_result")
        if isinstance(verify, dict):
            if verify.get("passed"):
                parts.append("\n검증 통과")
            else:
                issues = verify.get("issues", [])
                parts.append("\n검증 이슈:\n- " + "\n- ".join(issues) if issues else "\n검증 실패")
                suggestions = verify.get("suggestions", [])
                if suggestions:
                    parts.append("제안:\n- " + "\n- ".join(suggestions))
        return "\n".join(parts) if parts else data.get("last_ai_message", "응답을 생성하지 못했습니다.")

    if agent_name == "deep_research":
        return data.get("final_report") or data.get("last_ai_message", "보고서를 생성하지 못했습니다.")

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

    response_data: dict[str, Any] = {}
    last_node = ""
    passed = True
    token_streamed = False
    try:
        async for event in agent.graph.astream_events(
            input_state, config=run_config, version="v2",
        ):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node", "")

            # 노드 시작 시 진행 상태 표시
            if kind == "on_chain_start" and node in _NODE_LABELS and node != last_node:
                if token_streamed:
                    renderer.flush_tokens()
                    token_streamed = False
                renderer.system_message(f"  > {_NODE_LABELS[node]}")
                metrics.record_node_start(node)
                last_node = node

            # LLM 토큰 스트리밍 및 사용량 수집
            elif kind in ("on_chat_model_stream", "on_llm_stream"):
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
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

            # 노드 완료 시 응답 데이터 수집
            elif kind == "on_chain_end" and node:
                metrics.record_node_end(node)
                output = event["data"].get("output")
                if isinstance(output, dict):
                    for key in ("generated_code", "verify_result", "final_report",
                                "selected_agent", "agent_response"):
                        if key in output:
                            response_data[key] = output[key]
                    if "messages" in output:
                        for msg in output["messages"]:
                            if isinstance(msg, AIMessage) and msg.content:
                                response_data["last_ai_message"] = msg.content

        if token_streamed:
            renderer.flush_tokens()

    except Exception as e:
        if token_streamed:
            renderer.flush_tokens()
        metrics.record_error()
        logger.exception("에이전트 실행 오류")
        renderer.error(f"에이전트 실행 오류: {e}")
        return
    finally:
        # 메트릭 수집 완료 및 Langfuse flush
        metrics.finalize()
        if langfuse_handler is not None:
            logger.debug("에이전트 메트릭: %s", metrics.to_dict())
            safe_flush()

    # 검증 결과에서 passed 여부 추출
    verify = response_data.get("verify_result")
    if isinstance(verify, dict):
        passed = verify.get("passed", True)

    response = _extract_response(session.info.agent_name, response_data)

    # 토큰 스트리밍이 된 경우 이미 출력되었으므로 중복 출력 방지
    if not token_streamed:
        renderer.agent_message(response)

    session.add_message("assistant", response)

    # Episodic Memory 저장
    _save_episodic_memory(session, user_input, response, passed=passed)


def _init_skill_registry(skills_dir: str | None) -> SkillRegistry:
    """스킬 레지스트리를 초기화한다."""
    registry = SkillRegistry()
    if skills_dir:
        loader = SkillLoader(skills_dir)
        registry = SkillRegistry(loader=loader)
        discovered = registry.discover()
        if discovered:
            logger.info("스킬 %d개 발견: %s", len(discovered), discovered)
    return registry


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
    skill_registry = _init_skill_registry(config.skills_dir)
    checkpointer = _create_checkpointer(config)
    session = CLISession(
        agent_name=config.default_agent,
        skill_registry=skill_registry,
        checkpointer=checkpointer,
    )

    # Langfuse 콜백 핸들러 초기화 (설정으로 on/off 가능)
    langfuse_handler = None
    if config.langfuse_enabled:
        langfuse_handler = create_langfuse_handler()
        if langfuse_handler is not None:
            renderer.system_message("Langfuse 관측성 활성화됨")

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(config.history_file),
    )

    renderer.welcome(session.info.agent_name)

    while True:
        try:
            user_input = await asyncio.to_thread(
                prompt_session.prompt,
                f"[{session.info.agent_name}] > ",
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
            user_input, session, renderer, langfuse_handler=langfuse_handler,
        )


def run_cli(config: CLIConfig | None = None) -> None:
    """CLI 진입점."""
    config = config or CLIConfig()
    asyncio.run(_main_loop(config))
