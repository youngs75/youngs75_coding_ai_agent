"""CLI 유닛 테스트.

대화형 루프는 테스트하지 않고, 개별 컴포넌트를 검증한다.
"""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from rich.console import Console

from youngs75_a2a.cli.app import (
    _EPISODIC_MAX_ITEMS,
    _build_input_state,
    _create_checkpointer,
    _extract_response,
    _get_or_create_agent,
    _run_agent_turn,
    _save_episodic_memory,
)
from youngs75_a2a.cli.commands import handle_command
from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.eval_runner import (
    EvalResult,
    RemediationResult,
    format_eval_summary,
    format_remediation_summary,
    load_last_eval_results,
    load_last_remediation_report,
)
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.skills.registry import SkillRegistry
from youngs75_a2a.core.skills.schemas import Skill, SkillMetadata


def _make_renderer() -> CLIRenderer:
    return CLIRenderer(console=Console(file=StringIO(), force_terminal=True))


def _make_event(
    event_type: str,
    name: str = "",
    data: dict | None = None,
    node: str | None = None,
) -> dict:
    """테스트용 astream_events 이벤트 생성 헬퍼."""
    return {
        "event": event_type,
        "name": name,
        "data": data or {},
        "metadata": {"langgraph_node": node} if node else {},
        "tags": [],
        "run_id": "test-run",
        "parent_ids": [],
    }


# ── CLIConfig ──


class TestCLIConfig:
    def test_defaults(self):
        config = CLIConfig()
        assert config.default_agent == "coding_assistant"
        assert config.stream_output is True

    def test_checkpointer_default(self):
        config = CLIConfig()
        assert config.checkpointer_backend == "memory"

    def test_create_checkpointer_memory(self):
        from langgraph.checkpoint.memory import MemorySaver

        config = CLIConfig(checkpointer_backend="memory")
        cp = _create_checkpointer(config)
        assert isinstance(cp, MemorySaver)

    def test_create_checkpointer_sqlite_fallback(self):
        """sqlite 패키지 미설치 시 MemorySaver로 대체."""
        from langgraph.checkpoint.memory import MemorySaver

        config = CLIConfig(checkpointer_backend="sqlite")
        cp = _create_checkpointer(config)
        assert isinstance(cp, MemorySaver)


# ── CLISession ──


class TestCLISession:
    def test_session_id_generated(self):
        s = CLISession()
        assert len(s.session_id) == 12

    def test_add_message(self):
        s = CLISession()
        s.add_message("user", "hello")
        s.add_message("assistant", "hi")
        assert len(s.history) == 2
        assert s.info.message_count == 2

    def test_clear_history(self):
        s = CLISession()
        s.add_message("user", "test")
        s.clear_history()
        assert s.history == []
        assert s.info.message_count == 0

    def test_switch_agent(self):
        s = CLISession(agent_name="coding_assistant")
        s.switch_agent("deep_research")
        assert s.info.agent_name == "deep_research"

    def test_memory_store_available(self):
        s = CLISession()
        assert s.memory.total_count == 0

    def test_agent_cache(self):
        s = CLISession()
        assert s.get_cached_agent("coding_assistant") is None
        fake_agent = MagicMock()
        s.cache_agent("coding_assistant", fake_agent)
        assert s.get_cached_agent("coding_assistant") is fake_agent

    def test_agent_cache_multiple(self):
        s = CLISession()
        agent1 = MagicMock()
        agent2 = MagicMock()
        s.cache_agent("coding_assistant", agent1)
        s.cache_agent("deep_research", agent2)
        assert s.get_cached_agent("coding_assistant") is agent1
        assert s.get_cached_agent("deep_research") is agent2

    def test_skill_registry_default(self):
        s = CLISession()
        assert isinstance(s.skills, SkillRegistry)
        assert s.skills.list_skills() == []

    def test_skill_registry_custom(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(name="test_skill", description="테스트"),
            )
        )
        s = CLISession(skill_registry=registry)
        assert len(s.skills.list_skills()) == 1

    def test_checkpointer_default_none(self):
        s = CLISession()
        assert s.checkpointer is None

    def test_checkpointer_custom(self):
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        s = CLISession(checkpointer=cp)
        assert s.checkpointer is cp

    def test_thread_id_matches_session_id(self):
        s = CLISession()
        assert s.thread_id == s.session_id

    def test_get_history_summary_empty(self):
        s = CLISession()
        assert s.get_history_summary() == []

    def test_get_history_summary(self):
        s = CLISession()
        s.add_message("user", "hello")
        s.add_message("assistant", "hi")
        s.add_message("user", "bye")
        summary = s.get_history_summary(limit=2)
        assert len(summary) == 2
        assert summary[0]["content"] == "hi"
        assert summary[1]["content"] == "bye"

    def test_activate_skill_not_found(self):
        s = CLISession()
        assert s.activate_skill("nonexistent") is None

    def test_activate_skill_found(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(name="code_review", description="코드 리뷰"),
            )
        )
        s = CLISession(skill_registry=registry)
        assert s.activate_skill("code_review") == "code_review"


# ── CLIRenderer ──


class TestCLIRenderer:
    def test_welcome(self):
        r = _make_renderer()
        r.welcome("coding_assistant")  # 에러 없이 실행

    def test_messages(self):
        r = _make_renderer()
        r.user_message("hello")
        r.agent_message("response")
        r.system_message("info")
        r.error("oops")


# ── Commands ──


class TestCommands:
    def test_quit(self):
        result = handle_command("/quit", CLISession(), _make_renderer())
        assert result.should_quit

    def test_exit(self):
        result = handle_command("/exit", CLISession(), _make_renderer())
        assert result.should_quit

    def test_help(self):
        result = handle_command("/help", CLISession(), _make_renderer())
        assert result.handled

    def test_clear(self):
        session = CLISession()
        session.add_message("user", "test")
        result = handle_command("/clear", session, _make_renderer())
        assert result.handled
        assert session.info.message_count == 0

    def test_agent_switch(self):
        session = CLISession()
        result = handle_command("/agent coding_assistant", session, _make_renderer())
        assert result.handled
        assert session.info.agent_name == "coding_assistant"

    def test_agent_invalid(self):
        session = CLISession()
        result = handle_command("/agent nonexistent", session, _make_renderer())
        assert result.handled  # 에러 메시지 출력하지만 handled

    def test_agents_list(self):
        result = handle_command("/agents", CLISession(), _make_renderer())
        assert result.handled

    def test_session_info(self):
        result = handle_command("/session", CLISession(), _make_renderer())
        assert result.handled

    def test_memory_info(self):
        result = handle_command("/memory", CLISession(), _make_renderer())
        assert result.handled

    def test_unknown_command(self):
        result = handle_command("/unknown", CLISession(), _make_renderer())
        assert result.handled  # 에러 출력

    def test_non_command(self):
        result = handle_command("hello world", CLISession(), _make_renderer())
        assert not result.handled  # 일반 메시지로 처리

    # ── /skill 커맨드 ──

    def test_skill_list_empty(self):
        result = handle_command("/skill list", CLISession(), _make_renderer())
        assert result.handled

    def test_skill_list_with_skills(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(name="test_skill", description="테스트 스킬"),
            )
        )
        session = CLISession(skill_registry=registry)
        result = handle_command("/skill list", session, _make_renderer())
        assert result.handled

    def test_skill_activate_success(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(name="code_review", description="코드 리뷰"),
            )
        )
        session = CLISession(skill_registry=registry)
        result = handle_command(
            "/skill activate code_review", session, _make_renderer()
        )
        assert result.handled

    def test_skill_activate_not_found(self):
        result = handle_command(
            "/skill activate nonexistent", CLISession(), _make_renderer()
        )
        assert result.handled

    def test_skill_activate_no_name(self):
        result = handle_command("/skill activate", CLISession(), _make_renderer())
        assert result.handled

    def test_skill_unknown_sub(self):
        result = handle_command("/skill unknown", CLISession(), _make_renderer())
        assert result.handled

    # ── /history 커맨드 ──

    def test_history_empty(self):
        result = handle_command("/history", CLISession(), _make_renderer())
        assert result.handled

    def test_history_with_messages(self):
        session = CLISession()
        session.add_message("user", "hello")
        session.add_message("assistant", "hi there")
        result = handle_command("/history", session, _make_renderer())
        assert result.handled

    def test_history_clear(self):
        session = CLISession()
        session.add_message("user", "hello")
        result = handle_command("/history clear", session, _make_renderer())
        assert result.handled
        assert session.info.message_count == 0


# ── 에이전트 연동 헬퍼 ──


class TestBuildInputState:
    def test_coding_assistant_state(self):
        session = CLISession()
        state = _build_input_state(
            "coding_assistant", "피보나치 함수 작성해줘", session
        )
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "피보나치 함수 작성해줘"
        assert state["iteration"] == 0
        assert state["max_iterations"] == 3

    def test_simple_react_state(self):
        session = CLISession()
        state = _build_input_state("simple_react", "AI 트렌드 검색", session)
        assert len(state["messages"]) == 1
        assert "iteration" not in state

    def test_deep_research_state(self):
        session = CLISession()
        state = _build_input_state("deep_research", "양자 컴퓨팅 조사", session)
        assert len(state["messages"]) == 1
        assert "iteration" not in state

    def test_coding_with_semantic_memory(self):
        session = CLISession()
        session.memory.put(
            MemoryItem(
                type=MemoryType.SEMANTIC,
                content="프로젝트는 PEP 8 스타일 가이드를 따른다",
                tags=["convention"],
            )
        )
        session.memory.put(
            MemoryItem(
                type=MemoryType.SEMANTIC,
                content="모든 함수에 타입 힌트를 추가해야 한다",
                tags=["convention"],
            )
        )
        state = _build_input_state("coding_assistant", "함수 작성", session)
        assert "semantic_context" in state
        assert len(state["semantic_context"]) == 2
        contents = " ".join(state["semantic_context"])
        assert "PEP 8" in contents
        assert "타입 힌트" in contents

    def test_coding_without_semantic_memory(self):
        session = CLISession()
        state = _build_input_state("coding_assistant", "함수 작성", session)
        assert "semantic_context" not in state

    def test_coding_with_skill_context(self):
        registry = SkillRegistry()
        registry.register(
            Skill(
                metadata=SkillMetadata(
                    name="code_review",
                    description="코드 리뷰 수행",
                    tags=["review", "quality"],
                ),
            )
        )
        registry.register(
            Skill(
                metadata=SkillMetadata(
                    name="test_gen",
                    description="테스트 코드 자동 생성",
                    tags=["test"],
                ),
            )
        )
        session = CLISession(skill_registry=registry)
        state = _build_input_state("coding_assistant", "함수 작성", session)
        assert "skill_context" in state
        assert len(state["skill_context"]) == 2
        contents = " ".join(state["skill_context"])
        assert "code_review" in contents
        assert "test_gen" in contents

    def test_coding_without_skills(self):
        session = CLISession()
        state = _build_input_state("coding_assistant", "함수 작성", session)
        assert "skill_context" not in state

    def test_coding_with_procedural_memory(self):
        session = CLISession()
        session.memory.accumulate_skill(
            code="def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
            description="피보나치 함수 재귀 구현",
            tags=["python", "generate"],
        )
        state = _build_input_state("coding_assistant", "피보나치 함수 작성", session)
        assert "procedural_skills" in state
        assert len(state["procedural_skills"]) >= 1

    def test_coding_without_procedural_memory(self):
        session = CLISession()
        state = _build_input_state("coding_assistant", "함수 작성", session)
        assert "procedural_skills" not in state

    def test_coding_procedural_skills_content_format(self):
        """procedural_skills 항목이 description + code 형식인지 확인."""
        session = CLISession()
        session.memory.accumulate_skill(
            code="def add(a, b): return a + b",
            description="덧셈 함수 생성",
            tags=["python", "generate"],
        )
        state = _build_input_state("coding_assistant", "덧셈 함수 작성", session)
        assert "procedural_skills" in state
        skill_text = state["procedural_skills"][0]
        # MemoryItem.content = "{description}\n\n```\n{code}\n```" 형식
        assert "덧셈 함수 생성" in skill_text
        assert "def add(a, b)" in skill_text

    def test_coding_procedural_skills_limit(self):
        """retrieve_skills limit=3 으로 최대 3개 스킬만 주입되는지 확인."""
        session = CLISession()
        # 서로 다른 코드 패턴을 4개 이상 축적
        for i, (code, desc) in enumerate(
            [
                ("def func_a(): return 'a' * 100", "함수 A 생성"),
                (
                    "class Builder:\n    def build(self): return 'built_item'",
                    "빌더 클래스 생성",
                ),
                (
                    "async def fetch_data(url): return await client.get(url)",
                    "비동기 데이터 조회",
                ),
                (
                    "def parse_csv(path):\n    import csv\n    return list(csv.reader(open(path)))",
                    "CSV 파싱",
                ),
            ]
        ):
            session.memory.accumulate_skill(
                code=code, description=desc, tags=["python"]
            )
        state = _build_input_state("coding_assistant", "함수 작성", session)
        assert "procedural_skills" in state
        assert len(state["procedural_skills"]) <= 3

    def test_coding_procedural_roundtrip(self):
        """스킬 축적 후 다음 쿼리에서 해당 스킬이 검색되는 통합 확인."""
        session = CLISession()
        # 첫 번째 쿼리: 스킬 없음
        state1 = _build_input_state("coding_assistant", "정렬 함수 작성", session)
        assert "procedural_skills" not in state1

        # 스킬 축적 (에이전트 검증 통과 시 발생하는 동작 시뮬레이션)
        session.memory.accumulate_skill(
            code="def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr)//2\n    return merge(merge_sort(arr[:mid]), merge_sort(arr[mid:]))",
            description="머지소트 구현",
            tags=["generate", "python"],
        )

        # 두 번째 쿼리: 관련 스킬이 주입됨
        state2 = _build_input_state("coding_assistant", "정렬 sort 알고리즘", session)
        assert "procedural_skills" in state2
        assert any("merge_sort" in s for s in state2["procedural_skills"])

    def test_non_coding_agents_no_procedural(self):
        """coding_assistant 외 에이전트는 procedural_skills를 주입하지 않음."""
        session = CLISession()
        session.memory.accumulate_skill(
            code="def test(): pass",
            description="테스트 함수",
            tags=["python"],
        )
        for agent_name in ("simple_react", "deep_research"):
            state = _build_input_state(agent_name, "테스트", session)
            assert "procedural_skills" not in state


class TestExtractResponse:
    def test_coding_with_code_and_pass(self):
        data = {
            "generated_code": "def fib(n): ...",
            "verify_result": {"passed": True, "issues": [], "suggestions": []},
        }
        resp = _extract_response("coding_assistant", data)
        assert "def fib(n)" in resp
        assert "검증 통과" in resp

    def test_coding_with_code_and_fail(self):
        data = {
            "generated_code": "def fib(n): ...",
            "verify_result": {
                "passed": False,
                "issues": ["타입 힌트 누락"],
                "suggestions": [],
            },
        }
        resp = _extract_response("coding_assistant", data)
        assert "검증 이슈" in resp
        assert "타입 힌트 누락" in resp

    def test_coding_fallback_to_ai_message(self):
        data = {"last_ai_message": "코드 생성 결과입니다."}
        resp = _extract_response("coding_assistant", data)
        assert resp == "코드 생성 결과입니다."

    def test_coding_empty(self):
        resp = _extract_response("coding_assistant", {})
        assert "생성하지 못했습니다" in resp

    def test_deep_research(self):
        data = {"final_report": "# 보고서\n내용..."}
        resp = _extract_response("deep_research", data)
        assert "보고서" in resp

    def test_deep_research_empty(self):
        resp = _extract_response("deep_research", {})
        assert "생성하지 못했습니다" in resp

    def test_simple_react(self):
        data = {"last_ai_message": "검색 결과입니다."}
        resp = _extract_response("simple_react", data)
        assert resp == "검색 결과입니다."

    def test_orchestrator_with_agent_response(self):
        data = {
            "selected_agent": "coding_assistant",
            "agent_response": "코드 생성 완료",
        }
        resp = _extract_response("orchestrator", data)
        assert "[coding_assistant]" in resp
        assert "코드 생성 완료" in resp

    def test_orchestrator_fallback(self):
        data = {"last_ai_message": "요청을 처리했습니다."}
        resp = _extract_response("orchestrator", data)
        assert resp == "요청을 처리했습니다."

    def test_orchestrator_empty(self):
        resp = _extract_response("orchestrator", {})
        assert "생성하지 못했습니다" in resp


class TestGetOrCreateAgent:
    @pytest.mark.asyncio
    async def test_returns_cached_agent(self):
        session = CLISession()
        renderer = _make_renderer()
        fake_agent = MagicMock()
        session.cache_agent("coding_assistant", fake_agent)

        result = await _get_or_create_agent(session, renderer)
        assert result is fake_agent

    @pytest.mark.asyncio
    async def test_creates_and_caches_new_agent(self):
        session = CLISession()
        renderer = _make_renderer()
        fake_agent = MagicMock(spec=["graph"])
        # Phase 10 필드 추가 (에이전트 초기화 시 접근됨)
        fake_agent.project_context = None
        fake_agent.permission_manager = None
        fake_agent.tool_executor = None
        fake_agent.context_manager = None

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=fake_agent,
        ):
            result = await _get_or_create_agent(session, renderer)

        assert result is fake_agent
        assert session.get_cached_agent("coding_assistant") is fake_agent

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        session = CLISession()
        renderer = _make_renderer()

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("연결 실패"),
        ):
            result = await _get_or_create_agent(session, renderer)

        assert result is None


class TestRunAgentTurn:
    @pytest.mark.asyncio
    async def test_streams_agent_response(self):
        session = CLISession()
        renderer = _make_renderer()

        # 에이전트 그래프 astream_events mock
        ai_msg = AIMessage(content="피보나치 함수입니다.")
        fake_graph = MagicMock()

        async def fake_astream_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={
                    "output": {
                        "generated_code": "def fib(n): ...",
                        "messages": [ai_msg],
                    }
                },
                node="execute_code",
            )
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={
                    "output": {
                        "verify_result": {
                            "passed": True,
                            "issues": [],
                            "suggestions": [],
                        }
                    }
                },
                node="verify_result",
            )

        fake_graph.astream_events = fake_astream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("피보나치 함수 작성해줘", session, renderer)

        assert session.info.message_count == 2
        assert session.history[-1]["role"] == "assistant"
        assert "def fib(n)" in session.history[-1]["content"]

    @pytest.mark.asyncio
    async def test_token_streaming(self):
        """토큰 단위 스트리밍이 올바르게 동작하는지 검증."""
        from langchain_core.messages import AIMessageChunk

        session = CLISession()
        renderer = _make_renderer()
        ai_msg = AIMessage(content="Hello World")
        fake_graph = MagicMock()

        async def fake_astream_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "react_agent", node="react_agent")
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="Hello")},
                node="react_agent",
            )
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content=" World")},
                node="react_agent",
            )
            yield _make_event(
                "on_chain_end",
                "react_agent",
                data={"output": {"messages": [ai_msg]}},
                node="react_agent",
            )

        fake_graph.astream_events = fake_astream_events
        session.switch_agent("simple_react")
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("simple_react", fake_agent)

        await _run_agent_turn("test", session, renderer)

        assert session.info.message_count == 2
        assert session.history[-1]["content"] == "Hello World"

    @pytest.mark.asyncio
    async def test_handles_agent_error(self):
        session = CLISession()
        renderer = _make_renderer()

        fake_graph = MagicMock()

        async def failing_astream_events(input_state, config=None, version="v2"):
            raise RuntimeError("LLM API 오류")
            yield  # noqa: F841 — async generator 시그니처 유지

        fake_graph.astream_events = failing_astream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("test", session, renderer)

        # user 메시지만 추가되고 assistant 응답은 없어야 함
        assert session.info.message_count == 1
        assert session.history[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_returns_early_when_agent_unavailable(self):
        session = CLISession()
        renderer = _make_renderer()

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("초기화 실패"),
        ):
            await _run_agent_turn("test", session, renderer)

        # user 메시지만 추가
        assert session.info.message_count == 1


# ── Orchestrator ──


class TestOrchestratorCLI:
    def test_orchestrator_in_available_agents(self):
        from youngs75_a2a.cli.commands import AVAILABLE_AGENTS

        assert "orchestrator" in AVAILABLE_AGENTS

    def test_agent_switch_to_orchestrator(self):
        session = CLISession()
        result = handle_command("/agent orchestrator", session, _make_renderer())
        assert result.handled
        assert session.info.agent_name == "orchestrator"

    def test_orchestrator_input_state(self):
        session = CLISession()
        state = _build_input_state("orchestrator", "코드 작성해줘", session)
        assert len(state["messages"]) == 1
        assert state["selected_agent"] is None
        assert state["agent_response"] is None


# ── Episodic Memory ──


class TestEpisodicMemory:
    def test_episodic_memory_saved_after_turn(self):
        session = CLISession()
        _save_episodic_memory(
            session, "피보나치 함수 작성", "코드 생성 완료", passed=True
        )
        items = session.memory.list_by_type(
            MemoryType.EPISODIC, session_id=session.session_id
        )
        assert len(items) == 1
        assert "passed" in items[0].content
        assert "피보나치" in items[0].content

    def test_episodic_memory_failed_has_warning_tag(self):
        session = CLISession()
        _save_episodic_memory(session, "버그 수정", "타입 오류 발생", passed=False)
        items = session.memory.list_by_type(
            MemoryType.EPISODIC, session_id=session.session_id
        )
        assert len(items) == 1
        assert "주의" in items[0].tags
        assert "[주의] failed" in items[0].content

    def test_episodic_memory_injected(self):
        session = CLISession()
        _save_episodic_memory(session, "요청1", "결과1", passed=True)
        _save_episodic_memory(session, "요청2", "결과2", passed=False)
        state = _build_input_state("coding_assistant", "새 요청", session)
        assert "episodic_log" in state
        assert len(state["episodic_log"]) == 2

    def test_episodic_memory_limit(self):
        session = CLISession()
        for i in range(7):
            _save_episodic_memory(session, f"요청{i}", f"결과{i}", passed=True)
        state = _build_input_state("coding_assistant", "새 요청", session)
        assert "episodic_log" in state
        assert len(state["episodic_log"]) == _EPISODIC_MAX_ITEMS

    def test_episodic_memory_empty_returns_no_key(self):
        session = CLISession()
        state = _build_input_state("coding_assistant", "요청", session)
        assert "episodic_log" not in state

    def test_episodic_memory_session_scoped(self):
        session1 = CLISession()
        session2 = CLISession()
        _save_episodic_memory(session1, "세션1 요청", "세션1 결과", passed=True)
        _save_episodic_memory(session2, "세션2 요청", "세션2 결과", passed=True)
        state1 = _build_input_state("coding_assistant", "요청", session1)
        state2 = _build_input_state("coding_assistant", "요청", session2)
        assert len(state1["episodic_log"]) == 1
        assert len(state2["episodic_log"]) == 1
        assert state1["episodic_log"][0] != state2["episodic_log"][0]

    @pytest.mark.asyncio
    async def test_episodic_saved_after_agent_turn(self):
        session = CLISession()
        renderer = _make_renderer()
        ai_msg = AIMessage(content="결과입니다.")
        fake_graph = MagicMock()

        async def fake_astream_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"generated_code": "code", "messages": [ai_msg]}},
                node="execute_code",
            )
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={"output": {"verify_result": {"passed": True, "issues": []}}},
                node="verify_result",
            )

        fake_graph.astream_events = fake_astream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("함수 작성", session, renderer)

        items = session.memory.list_by_type(
            MemoryType.EPISODIC, session_id=session.session_id
        )
        assert len(items) == 1
        assert "passed" in items[0].content


# ── Procedural Memory CLI 통합 ──


class TestProceduralMemoryCLI:
    """Procedural Memory(Voyager 패턴) CLI 통합 테스트."""

    @pytest.mark.asyncio
    async def test_create_agent_receives_memory_store(self):
        """_create_agent가 memory_store를 에이전트에 전달하는지 확인."""
        session = CLISession()
        renderer = _make_renderer()
        fake_agent = MagicMock(spec=["graph"])

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=fake_agent,
        ) as mock_create:
            await _get_or_create_agent(session, renderer)
            mock_create.assert_called_once_with(
                "coding_assistant",
                checkpointer=session.checkpointer,
                memory_store=session.memory,
            )

    def test_session_memory_shared_between_turns(self):
        """동일 세션 내 MemoryStore가 턴 간 공유되는지 확인."""
        session = CLISession()
        store_ref = session.memory

        # 첫 번째 턴에서 스킬 축적
        store_ref.accumulate_skill(
            code="def validate(data): return bool(data)",
            description="데이터 유효성 검증 함수",
            tags=["python", "generate"],
        )

        # 두 번째 턴에서 같은 MemoryStore를 통해 스킬 조회
        state = _build_input_state("coding_assistant", "데이터 검증 validate", session)
        assert "procedural_skills" in state
        assert any("validate" in s for s in state["procedural_skills"])

    @pytest.mark.asyncio
    async def test_procedural_memory_persists_across_agent_turns(self):
        """에이전트 턴 실행 후 축적된 스킬이 다음 턴 input에 반영되는 통합 테스트."""
        session = CLISession()
        renderer = _make_renderer()

        # 첫 번째 턴 전에 스킬 미리 축적 (검증 통과 시 자동 축적 시뮬레이션)
        session.memory.accumulate_skill(
            code="def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot]) + [pivot] + quicksort([x for x in arr[1:] if x >= pivot])",
            description="퀵소트 정렬 알고리즘 구현",
            tags=["generate", "python"],
        )

        ai_msg = AIMessage(content="정렬 결과입니다.")
        fake_graph = MagicMock()

        captured_input_state = {}

        async def fake_astream_events(input_state, config=None, version="v2"):
            captured_input_state.update(input_state)
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={
                    "output": {
                        "generated_code": "def sort(arr): ...",
                        "messages": [ai_msg],
                    }
                },
                node="execute_code",
            )
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={"output": {"verify_result": {"passed": True, "issues": []}}},
                node="verify_result",
            )

        fake_graph.astream_events = fake_astream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("정렬 sort 함수 작성해줘", session, renderer)

        # 에이전트에 전달된 input_state에 procedural_skills가 포함되어야 함
        assert "procedural_skills" in captured_input_state
        assert any("quicksort" in s for s in captured_input_state["procedural_skills"])

    def test_procedural_memory_empty_query_no_crash(self):
        """빈 쿼리에서도 procedural 검색이 오류 없이 동작하는지 확인."""
        session = CLISession()
        session.memory.accumulate_skill(
            code="print('hello')",
            description="간단한 출력",
            tags=["python"],
        )
        # 빈 문자열 쿼리도 정상 처리
        state = _build_input_state("coding_assistant", "", session)
        # procedural_skills가 있을 수도 있고 없을 수도 있지만 오류는 없어야 함
        assert "messages" in state


# ── 토큰 스트리밍 + Checkpointer 통합 테스트 ──


class TestTokenStreamingIntegration:
    """토큰 스트리밍(astream_events v2) 통합 테스트."""

    @pytest.mark.asyncio
    async def test_multi_node_streaming_with_transitions(self):
        """여러 노드를 거치며 토큰 스트리밍이 올바르게 전환되는지 확인."""
        from langchain_core.messages import AIMessageChunk

        session = CLISession()
        buf = StringIO()
        renderer = CLIRenderer(console=Console(file=buf, force_terminal=True))
        ai_msg = AIMessage(content="생성된 코드입니다.")
        fake_graph = MagicMock()

        async def fake_astream_events(input_state, config=None, version="v2"):
            # parse 노드
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            # execute 노드 — 토큰 스트리밍
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="def ")},
                node="execute_code",
            )
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="hello():")},
                node="execute_code",
            )
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={
                    "output": {"generated_code": "def hello():", "messages": [ai_msg]}
                },
                node="execute_code",
            )
            # verify 노드
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={"output": {"verify_result": {"passed": True, "issues": []}}},
                node="verify_result",
            )

        fake_graph.astream_events = fake_astream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("함수 작성", session, renderer)

        output = buf.getvalue()
        # 노드 전환 상태 메시지가 출력됨 (스피너 포함)
        assert "요청 분석" in output
        assert "코드 생성" in output
        assert "코드 검증" in output
        # 스트리밍된 토큰이 출력됨
        assert "def " in output
        assert "hello():" in output

    @pytest.mark.asyncio
    async def test_streaming_with_tool_call_loop(self):
        """도구 호출 루프(execute → tools → execute → verify) 스트리밍 확인."""
        from langchain_core.messages import AIMessageChunk

        session = CLISession()
        renderer = _make_renderer()
        fake_graph = MagicMock()

        # tool_call을 포함한 AI 메시지
        tool_ai_msg = MagicMock(spec=AIMessage)
        tool_ai_msg.content = ""
        tool_ai_msg.tool_calls = [
            {"name": "read_file", "args": {"path": "test.py"}, "id": "tc1"}
        ]

        final_ai_msg = AIMessage(content="완료된 코드입니다.")

        async def fake_astream_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            # 첫 번째 execute — 도구 호출 결정
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"messages": [tool_ai_msg]}},
                node="execute_code",
            )
            # tools 노드
            yield _make_event("on_chain_start", "execute_tools", node="execute_tools")
            yield _make_event(
                "on_chain_end",
                "execute_tools",
                data={"output": {"messages": []}},
                node="execute_tools",
            )
            # 두 번째 execute — 실제 코드 생성 + 스트리밍
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="result = 42")},
                node="execute_code",
            )
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={
                    "output": {
                        "generated_code": "result = 42",
                        "messages": [final_ai_msg],
                    }
                },
                node="execute_code",
            )
            # verify
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={"output": {"verify_result": {"passed": True, "issues": []}}},
                node="verify_result",
            )

        fake_graph.astream_events = fake_astream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("코드 작성", session, renderer)

        assert session.info.message_count == 2
        assert session.history[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_empty_content_chunks_skipped(self):
        """빈 content 청크는 스트리밍에서 무시됨."""
        from langchain_core.messages import AIMessageChunk

        session = CLISession()
        buf = StringIO()
        renderer = CLIRenderer(console=Console(file=buf, force_terminal=True))
        ai_msg = AIMessage(content="결과")
        fake_graph = MagicMock()

        async def fake_astream_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            # 빈 content 청크
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="")},
                node="execute_code",
            )
            # None content 청크 (chunk 없음)
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": None},
                node="execute_code",
            )
            # 정상 content 청크
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="결과 코드")},
                node="execute_code",
            )
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"generated_code": "결과 코드", "messages": [ai_msg]}},
                node="execute_code",
            )

        fake_graph.astream_events = fake_astream_events
        session.switch_agent("simple_react")
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("simple_react", fake_agent)

        await _run_agent_turn("테스트", session, renderer)

        output = buf.getvalue()
        assert "결과 코드" in output

    @pytest.mark.asyncio
    async def test_streaming_error_midway_flushes_and_recovers(self):
        """스트리밍 도중 에러 발생 시 토큰 flush 후 에러 처리."""
        from langchain_core.messages import AIMessageChunk

        session = CLISession()
        buf = StringIO()
        renderer = CLIRenderer(console=Console(file=buf, force_terminal=True))
        fake_graph = MagicMock()

        async def failing_midstream(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="partial ")},
                node="execute_code",
            )
            raise RuntimeError("연결 끊김")

        fake_graph.astream_events = failing_midstream
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("테스트", session, renderer)

        output = buf.getvalue()
        # 부분 토큰이 출력된 후 에러 메시지 표시
        assert "partial " in output
        assert "에이전트 실행 오류" in output

    @pytest.mark.asyncio
    async def test_non_streaming_response_uses_agent_message(self):
        """토큰 스트리밍 없이 완료된 경우 agent_message로 응답 출력."""
        session = CLISession()
        buf = StringIO()
        renderer = CLIRenderer(console=Console(file=buf, force_terminal=True))
        ai_msg = AIMessage(content="최종 응답")
        fake_graph = MagicMock()

        async def no_stream_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            # 토큰 스트리밍 없이 바로 chain_end
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"generated_code": "x = 1", "messages": [ai_msg]}},
                node="execute_code",
            )
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={"output": {"verify_result": {"passed": True, "issues": []}}},
                node="verify_result",
            )

        fake_graph.astream_events = no_stream_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("테스트", session, renderer)

        # 토큰 스트리밍이 없으므로 agent_message()를 통해 출력됨
        assert session.info.message_count == 2
        assert "x = 1" in session.history[-1]["content"]


class TestCheckpointerIntegration:
    """LangGraph Checkpointer 통합 테스트."""

    @pytest.mark.asyncio
    async def test_checkpointer_passed_to_agent(self):
        """CLISession.checkpointer가 _create_agent()에 올바르게 전달됨."""
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        session = CLISession(checkpointer=cp)
        renderer = _make_renderer()
        fake_agent = MagicMock(spec=["graph"])

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=fake_agent,
        ) as mock_create:
            await _get_or_create_agent(session, renderer)
            mock_create.assert_called_once_with(
                "coding_assistant",
                checkpointer=cp,
                memory_store=session.memory,
            )

    @pytest.mark.asyncio
    async def test_run_config_contains_session_thread_id(self):
        """astream_events에 전달되는 config에 session thread_id가 포함됨."""
        session = CLISession()
        renderer = _make_renderer()
        ai_msg = AIMessage(content="응답")
        fake_graph = MagicMock()

        captured_config: dict = {}

        async def capture_config(input_state, config=None, version="v2"):
            captured_config.update(config or {})
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"messages": [ai_msg]}},
                node="execute_code",
            )

        fake_graph.astream_events = capture_config
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("테스트", session, renderer)

        assert "configurable" in captured_config
        assert captured_config["configurable"]["thread_id"] == session.thread_id

    @pytest.mark.asyncio
    async def test_different_sessions_have_different_thread_ids(self):
        """서로 다른 세션은 서로 다른 thread_id를 가짐 (상태 격리)."""
        session1 = CLISession()
        session2 = CLISession()

        assert session1.thread_id != session2.thread_id

        configs: list[dict] = []

        async def capture_config_factory():
            async def capture(input_state, config=None, version="v2"):
                configs.append(config or {})
                ai_msg = AIMessage(content="응답")
                yield _make_event(
                    "on_chain_end",
                    "react_agent",
                    data={"output": {"messages": [ai_msg]}},
                    node="react_agent",
                )

            return capture

        for sess in [session1, session2]:
            renderer = _make_renderer()
            fake_graph = MagicMock()
            fake_graph.astream_events = await capture_config_factory()
            sess.switch_agent("simple_react")
            fake_agent = MagicMock(graph=fake_graph)
            sess.cache_agent("simple_react", fake_agent)
            await _run_agent_turn("테스트", sess, renderer)

        assert len(configs) == 2
        tid1 = configs[0]["configurable"]["thread_id"]
        tid2 = configs[1]["configurable"]["thread_id"]
        assert tid1 != tid2

    def test_sqlite_config_path(self):
        """SQLite 체크포인터 경로 설정이 올바르게 적용됨."""
        config = CLIConfig(
            checkpointer_backend="sqlite",
            checkpointer_sqlite_path="/tmp/test_checkpoints.db",
        )
        assert config.checkpointer_sqlite_path == "/tmp/test_checkpoints.db"

    @pytest.mark.asyncio
    async def test_multi_turn_session_state(self):
        """동일 세션에서 멀티턴 실행 시 히스토리와 메모리가 누적됨."""
        session = CLISession()
        renderer = _make_renderer()

        ai_msgs = [AIMessage(content=f"응답 {i}") for i in range(3)]
        turn_count = 0

        async def make_events(input_state, config=None, version="v2"):
            nonlocal turn_count
            msg = ai_msgs[turn_count]
            turn_count += 1
            yield _make_event(
                "on_chain_end",
                "react_agent",
                data={"output": {"messages": [msg]}},
                node="react_agent",
            )

        session.switch_agent("simple_react")
        fake_graph = MagicMock()
        fake_graph.astream_events = make_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("simple_react", fake_agent)

        # 3턴 실행
        for i in range(3):
            await _run_agent_turn(f"질문 {i}", session, renderer)

        # 히스토리 누적 확인 (user + assistant 각 3쌍)
        assert session.info.message_count == 6
        assert len(session.history) == 6
        assert session.history[0]["role"] == "user"
        assert session.history[0]["content"] == "질문 0"
        assert session.history[1]["role"] == "assistant"
        assert "응답 0" in session.history[1]["content"]
        assert session.history[4]["role"] == "user"
        assert session.history[4]["content"] == "질문 2"


class TestStreamingCheckpointerE2E:
    """토큰 스트리밍 + Checkpointer 결합 E2E 테스트."""

    @pytest.mark.asyncio
    async def test_full_pipeline_streaming_checkpointer_memory(self):
        """스트리밍 + 체크포인터 + 메모리(episodic/procedural) 전체 파이프라인 E2E."""
        from langchain_core.messages import AIMessageChunk
        from langgraph.checkpoint.memory import MemorySaver

        cp = MemorySaver()
        session = CLISession(checkpointer=cp)
        buf = StringIO()
        renderer = CLIRenderer(console=Console(file=buf, force_terminal=True))

        # Procedural 스킬 축적
        session.memory.accumulate_skill(
            code="def quick_sort(arr): ...",
            description="퀵소트 구현",
            tags=["python", "generate"],
        )

        ai_msg = AIMessage(content="정렬 코드입니다.")
        fake_graph = MagicMock()

        captured_state: dict = {}

        async def full_pipeline_events(input_state, config=None, version="v2"):
            captured_state.update(input_state)
            # parse
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            # execute + 스트리밍
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            for token in ["def ", "sort", "(arr)", ":\n", "    ", "return sorted(arr)"]:
                yield _make_event(
                    "on_chat_model_stream",
                    "ChatOpenAI",
                    data={"chunk": AIMessageChunk(content=token)},
                    node="execute_code",
                )
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={
                    "output": {
                        "generated_code": "def sort(arr):\n    return sorted(arr)",
                        "messages": [ai_msg],
                    }
                },
                node="execute_code",
            )
            # verify
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={
                    "output": {
                        "verify_result": {
                            "passed": True,
                            "issues": [],
                            "suggestions": [],
                        }
                    }
                },
                node="verify_result",
            )

        fake_graph.astream_events = full_pipeline_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("정렬 sort 함수 작성", session, renderer)

        output = buf.getvalue()

        # 1. 스트리밍 토큰이 올바르게 출력됨
        assert "def " in output
        assert "sort" in output
        assert "return sorted(arr)" in output

        # 2. 노드 전환 메시지가 출력됨 (스피너 포함)
        assert "요청 분석" in output
        assert "코드 생성" in output
        assert "코드 검증" in output

        # 3. 세션 히스토리에 기록됨
        assert session.info.message_count == 2

        # 4. Procedural 스킬이 input_state에 포함됨
        assert "procedural_skills" in captured_state
        assert any("quick_sort" in s for s in captured_state["procedural_skills"])

        # 5. Episodic 메모리가 저장됨
        ep_items = session.memory.list_by_type(
            MemoryType.EPISODIC,
            session_id=session.session_id,
        )
        assert len(ep_items) == 1
        assert "passed" in ep_items[0].content

        # 6. 체크포인터가 세션에 설정되어 있음 (config 검증은 별도 테스트에서 수행)
        assert session.checkpointer is cp

    @pytest.mark.asyncio
    async def test_multi_turn_streaming_with_memory_accumulation(self):
        """멀티턴: 1턴에서 스킬 축적 → 2턴에서 procedural 스킬 주입 확인."""
        from langchain_core.messages import AIMessageChunk

        session = CLISession()
        renderer = _make_renderer()
        fake_graph = MagicMock()

        turn_states: list[dict] = []

        async def events_factory(input_state, config=None, version="v2"):
            turn_states.append(dict(input_state))
            ai_msg = AIMessage(content="코드 결과")
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chat_model_stream",
                "ChatOpenAI",
                data={"chunk": AIMessageChunk(content="코드 출력")},
                node="execute_code",
            )
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"generated_code": "x = 1", "messages": [ai_msg]}},
                node="execute_code",
            )
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={"output": {"verify_result": {"passed": True, "issues": []}}},
                node="verify_result",
            )

        fake_graph.astream_events = events_factory
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        # 1턴: procedural 스킬 없음
        await _run_agent_turn("함수 작성", session, renderer)
        assert "procedural_skills" not in turn_states[0]

        # 턴 사이에 스킬 축적 (에이전트 검증 통과 후 자동 축적 시뮬레이션)
        session.memory.accumulate_skill(
            code="def calculate(x, y): return x + y",
            description="계산 함수 생성",
            tags=["python", "generate"],
        )

        # 2턴: procedural 스킬이 주입됨
        await _run_agent_turn("계산 calculate 함수 작성", session, renderer)
        assert "procedural_skills" in turn_states[1]
        assert any("calculate" in s for s in turn_states[1]["procedural_skills"])

        # 히스토리 및 episodic 누적 확인
        assert session.info.message_count == 4
        ep_items = session.memory.list_by_type(
            MemoryType.EPISODIC,
            session_id=session.session_id,
        )
        assert len(ep_items) == 2

    @pytest.mark.asyncio
    async def test_verify_failed_turn_episodic_marked(self):
        """검증 실패 턴: episodic 메모리에 [주의] 태그가 포함됨."""
        session = CLISession()
        renderer = _make_renderer()
        ai_msg = AIMessage(content="코드")
        fake_graph = MagicMock()

        async def fail_verify_events(input_state, config=None, version="v2"):
            yield _make_event("on_chain_start", "parse_request", node="parse_request")
            yield _make_event(
                "on_chain_end",
                "parse_request",
                data={"output": {"parse_result": {"task_type": "generate"}}},
                node="parse_request",
            )
            yield _make_event("on_chain_start", "execute_code", node="execute_code")
            yield _make_event(
                "on_chain_end",
                "execute_code",
                data={"output": {"generated_code": "bad code", "messages": [ai_msg]}},
                node="execute_code",
            )
            yield _make_event("on_chain_start", "verify_result", node="verify_result")
            yield _make_event(
                "on_chain_end",
                "verify_result",
                data={
                    "output": {
                        "verify_result": {
                            "passed": False,
                            "issues": ["타입 오류"],
                            "suggestions": ["타입 힌트 추가"],
                        }
                    }
                },
                node="verify_result",
            )

        fake_graph.astream_events = fail_verify_events
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("코드 작성", session, renderer)

        ep_items = session.memory.list_by_type(
            MemoryType.EPISODIC,
            session_id=session.session_id,
        )
        assert len(ep_items) == 1
        assert "주의" in ep_items[0].tags
        assert "[주의] failed" in ep_items[0].content


# ── Eval Pipeline ──


class TestEvalCommands:
    def test_eval_command_handled(self):
        mock_result = EvalResult(success=False, error_message="테스트 환경")
        with patch(
            "youngs75_a2a.cli.eval_runner._run_evaluation_sync",
            return_value=mock_result,
        ):
            result = handle_command("/eval", CLISession(), _make_renderer())
        assert result.handled

    def test_eval_status_command_handled(self):
        with patch(
            "youngs75_a2a.cli.commands.load_last_eval_results",
            return_value=EvalResult(success=False, error_message="결과 없음"),
        ):
            result = handle_command("/eval status", CLISession(), _make_renderer())
        assert result.handled

    def test_eval_unknown_subcommand(self):
        result = handle_command("/eval unknown", CLISession(), _make_renderer())
        assert result.handled

    def test_eval_remediate_status_command_handled(self):
        with patch(
            "youngs75_a2a.cli.commands.load_last_remediation_report",
            return_value=RemediationResult(success=False, error_message="리포트 없음"),
        ):
            result = handle_command(
                "/eval remediate status", CLISession(), _make_renderer()
            )
        assert result.handled


class TestEvalRunner:
    def test_pass_rate(self):
        result = EvalResult(success=True, total=10, passed=7, failed=3)
        assert result.pass_rate == 0.7

    def test_pass_rate_zero(self):
        result = EvalResult(success=True, total=0, passed=0, failed=0)
        assert result.pass_rate == 0.0

    def test_format_success(self):
        result = EvalResult(
            success=True,
            total=3,
            passed=2,
            failed=1,
            results=[
                {"id": "a1", "input": "질문1", "scores": {"m": 0.9}, "passed": True},
                {"id": "a2", "input": "질문2", "scores": {"m": 0.8}, "passed": True},
                {"id": "a3", "input": "질문3", "scores": {"m": 0.3}, "passed": False},
            ],
        )
        summary = format_eval_summary(result)
        assert "총 3건" in summary
        assert "통과: 2건" in summary
        assert "66.7%" in summary

    def test_format_failure(self):
        result = EvalResult(success=False, error_message="파일 없음")
        assert "파일 없음" in format_eval_summary(result)

    def test_load_missing_file(self):
        with patch(
            "youngs75_a2a.cli.eval_runner._DEFAULT_RESULTS_FILE",
            Path("/nonexistent/eval_results.json"),
        ):
            result = load_last_eval_results()
        assert not result.success

    def test_load_valid_file(self):
        data = [
            {"id": "t1", "passed": True, "timestamp": "2026-01-01T00:00:00"},
            {"id": "t2", "passed": False, "timestamp": "2026-01-01T01:00:00"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            with patch("youngs75_a2a.cli.eval_runner._DEFAULT_RESULTS_FILE", tmp_path):
                result = load_last_eval_results()
            assert result.success
            assert result.total == 2
            assert result.passed == 1
        finally:
            tmp_path.unlink(missing_ok=True)


# ── Remediation Runner ──


class TestRemediationRunner:
    def test_remediation_result_success(self):
        result = RemediationResult(
            success=True, report="dummy", report_path="/tmp/r.json"
        )
        assert result.success
        assert result.report == "dummy"

    def test_remediation_result_failure(self):
        result = RemediationResult(success=False, error_message="의존성 없음")
        assert not result.success
        assert result.error_message == "의존성 없음"

    def test_format_remediation_failure(self):
        result = RemediationResult(success=False, error_message="파일 없음")
        assert "파일 없음" in format_remediation_summary(result)

    def test_format_remediation_success(self):
        # format_report를 갖는 mock 리포트
        mock_report = MagicMock()
        mock_report.format_report.return_value = "REMEDIATION REPORT\n요약: 테스트"
        result = RemediationResult(success=True, report=mock_report)
        summary = format_remediation_summary(result)
        assert "REMEDIATION REPORT" in summary

    def test_format_remediation_no_report(self):
        result = RemediationResult(success=True, report=None)
        assert "비어있습니다" in format_remediation_summary(result)

    def test_load_missing_report(self):
        with patch(
            "youngs75_a2a.cli.eval_runner._EVAL_RESULTS_DIR", Path("/nonexistent")
        ):
            result = load_last_remediation_report()
        assert not result.success

    def test_load_valid_report(self):
        data = {
            "summary": "테스트",
            "failure_analysis": {
                "total_evaluated": 10,
                "total_failed": 2,
                "failure_rate": 0.2,
                "categories": [],
            },
            "prompt_optimizations": [],
            "recommendations": [],
            "next_steps": [],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = Path(f.name)
        try:
            report_dir = tmp_path.parent
            with patch("youngs75_a2a.cli.eval_runner._EVAL_RESULTS_DIR", report_dir):
                # 파일명을 맞추기 위해 직접 경로로 테스트
                from youngs75_a2a.cli.eval_runner import (
                    load_last_remediation_report as _load,
                )

                # load_last_remediation_report는 remediation_report.json 파일명을 기대하므로
                # 임시 파일 대신 remediation_report.json으로 복사
                import shutil

                report_path = report_dir / "remediation_report.json"
                shutil.copy(tmp_path, report_path)
                result = _load()
            assert result.success
            assert result.report.summary == "테스트"
        finally:
            tmp_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)


# ── 프롬프트 버전 관리 ──


class TestPromptRegistry:
    def test_initial_prompts_registered(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        assert "parse" in registry.list_prompts()
        assert "execute" in registry.list_prompts()
        assert "verify" in registry.list_prompts()

    def test_get_prompt_latest(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get_prompt("parse")
        assert "요청을 분석" in prompt

    def test_get_prompt_v1(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get_prompt("parse", version="v1")
        assert "요청을 분석" in prompt

    def test_get_prompt_invalid_name(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        with pytest.raises(KeyError):
            registry.get_prompt("nonexistent")

    def test_get_prompt_invalid_version(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        with pytest.raises(ValueError):
            registry.get_prompt("parse", version="v99")

    def test_current_version(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        assert registry.get_current_version("parse") == "v1"

    def test_list_versions(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        assert registry.list_versions("execute") == ["v1"]

    def test_apply_remediation(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()

        changes = [
            {
                "target_prompt": "execute",
                "issue": "타입 힌트 누락",
                "change": "타입 힌트 필수 규칙 추가",
                "metric": "faithfulness +0.1",
            },
        ]
        updated = registry.apply_remediation(changes)
        assert "execute" in updated
        assert registry.get_current_version("execute") == "v2"
        assert registry.list_versions("execute") == ["v1", "v2"]

        # 새 버전에 개선 내용 포함
        new_prompt = registry.get_prompt("execute")
        assert "타입 힌트 필수 규칙 추가" in new_prompt

        # v1은 원본 유지
        v1_prompt = registry.get_prompt("execute", version="v1")
        assert "타입 힌트 필수 규칙 추가" not in v1_prompt

    def test_apply_remediation_multiple(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()

        changes = [
            {
                "target_prompt": "execute",
                "issue": "이슈1",
                "change": "변경1",
                "metric": "메트릭1",
            },
            {
                "target_prompt": "execute",
                "issue": "이슈2",
                "change": "변경2",
                "metric": "메트릭2",
            },
        ]
        updated = registry.apply_remediation(changes)
        assert "execute" in updated
        # 두 번 적용되므로 v3
        assert registry.get_current_version("execute") == "v3"

    def test_apply_remediation_unknown_target(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()

        changes = [
            {
                "target_prompt": "unknown_agent",
                "issue": "이슈",
                "change": "변경",
                "metric": "메트릭",
            },
        ]
        updated = registry.apply_remediation(changes)
        assert updated == []

    def test_apply_remediation_name_mapping(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()

        changes = [
            {
                "target_prompt": "verification",
                "issue": "검증 느슨함",
                "change": "기준 강화",
                "metric": "정확성 +10%",
            },
        ]
        updated = registry.apply_remediation(changes)
        assert "verify" in updated
        assert registry.get_current_version("verify") == "v2"

    def test_execute_prompt_contains_citation_guide(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get_prompt("execute")
        assert "인용 형식 규칙" in prompt
        assert "파일: path/to/file.py:42" in prompt
        assert "출처 URL" in prompt

    def test_verify_prompt_contains_citation_quality_check(self):
        from youngs75_a2a.agents.coding_assistant.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get_prompt("verify")
        assert "인용 품질" in prompt
        assert "파일 경로/라인 번호" in prompt

    def test_deep_research_prompts_contain_citation_rules(self):
        from youngs75_a2a.agents.deep_research.prompts import (
            RESEARCHER_SYSTEM_PROMPT,
            FINAL_REPORT_PROMPT,
        )

        assert "인용 형식 규칙" in RESEARCHER_SYSTEM_PROMPT
        assert "출처 인용 형식" in FINAL_REPORT_PROMPT
        assert "참고 자료" in FINAL_REPORT_PROMPT

    def test_citation_quality_eval_prompt_includes_code_policy(self):
        from youngs75_a2a.eval_pipeline.loop2_evaluation.prompts import (
            CITATION_QUALITY_PROMPT,
        )

        assert "Code-specific citation policy" in CITATION_QUALITY_PROMPT
        assert "file path and line number" in CITATION_QUALITY_PROMPT

    def test_singleton_registry(self):
        from youngs75_a2a.agents.coding_assistant.prompts import (
            get_prompt_registry,
            reset_prompt_registry,
        )

        reset_prompt_registry()
        r1 = get_prompt_registry()
        r2 = get_prompt_registry()
        assert r1 is r2
        reset_prompt_registry()  # 정리
