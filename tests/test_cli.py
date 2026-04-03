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
    _build_episodic_summary,
    _build_input_state,
    _extract_response,
    _get_or_create_agent,
    _run_agent_turn,
    _save_episodic_memory,
)
from youngs75_a2a.cli.commands import CommandResult, handle_command
from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.eval_runner import EvalResult, format_eval_summary, load_last_eval_results
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.skills.registry import SkillRegistry
from youngs75_a2a.core.skills.schemas import Skill, SkillMetadata


def _make_renderer() -> CLIRenderer:
    return CLIRenderer(console=Console(file=StringIO(), force_terminal=True))


# ── CLIConfig ──


class TestCLIConfig:
    def test_defaults(self):
        config = CLIConfig()
        assert config.default_agent == "coding_assistant"
        assert config.stream_output is True


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
        registry.register(Skill(
            metadata=SkillMetadata(name="test_skill", description="테스트"),
        ))
        s = CLISession(skill_registry=registry)
        assert len(s.skills.list_skills()) == 1


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


# ── 에이전트 연동 헬퍼 ──


class TestBuildInputState:
    def test_coding_assistant_state(self):
        session = CLISession()
        state = _build_input_state("coding_assistant", "피보나치 함수 작성해줘", session)
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
        session.memory.put(MemoryItem(
            type=MemoryType.SEMANTIC,
            content="프로젝트는 PEP 8 스타일 가이드를 따른다",
            tags=["convention"],
        ))
        session.memory.put(MemoryItem(
            type=MemoryType.SEMANTIC,
            content="모든 함수에 타입 힌트를 추가해야 한다",
            tags=["convention"],
        ))
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
        registry.register(Skill(
            metadata=SkillMetadata(
                name="code_review",
                description="코드 리뷰 수행",
                tags=["review", "quality"],
            ),
        ))
        registry.register(Skill(
            metadata=SkillMetadata(
                name="test_gen",
                description="테스트 코드 자동 생성",
                tags=["test"],
            ),
        ))
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
            "verify_result": {"passed": False, "issues": ["타입 힌트 누락"], "suggestions": []},
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
        data = {"selected_agent": "coding_assistant", "agent_response": "코드 생성 완료"}
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

        with patch("youngs75_a2a.cli.app._create_agent", new_callable=AsyncMock, return_value=fake_agent):
            result = await _get_or_create_agent(session, renderer)

        assert result is fake_agent
        assert session.get_cached_agent("coding_assistant") is fake_agent

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        session = CLISession()
        renderer = _make_renderer()

        with patch("youngs75_a2a.cli.app._create_agent", new_callable=AsyncMock, side_effect=RuntimeError("연결 실패")):
            result = await _get_or_create_agent(session, renderer)

        assert result is None


class TestRunAgentTurn:
    @pytest.mark.asyncio
    async def test_streams_agent_response(self):
        session = CLISession()
        renderer = _make_renderer()

        # 에이전트 그래프 astream mock
        ai_msg = AIMessage(content="피보나치 함수입니다.")
        fake_graph = MagicMock()

        async def fake_astream(input_state):
            yield {"parse_request": {"parse_result": {"task_type": "generate"}}}
            yield {"execute_code": {"generated_code": "def fib(n): ...", "messages": [ai_msg]}}
            yield {"verify_result": {"verify_result": {"passed": True, "issues": [], "suggestions": []}}}

        fake_graph.astream = fake_astream
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("피보나치 함수 작성해줘", session, renderer)

        assert session.info.message_count == 2
        assert session.history[-1]["role"] == "assistant"
        assert "def fib(n)" in session.history[-1]["content"]

    @pytest.mark.asyncio
    async def test_handles_agent_error(self):
        session = CLISession()
        renderer = _make_renderer()

        fake_graph = MagicMock()

        async def failing_astream(input_state):
            raise RuntimeError("LLM API 오류")
            yield  # noqa: unreachable — async generator 시그니처 유지

        fake_graph.astream = failing_astream
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

        with patch("youngs75_a2a.cli.app._create_agent", new_callable=AsyncMock, side_effect=RuntimeError("초기화 실패")):
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
        _save_episodic_memory(session, "피보나치 함수 작성", "코드 생성 완료", passed=True)
        items = session.memory.list_by_type(MemoryType.EPISODIC, session_id=session.session_id)
        assert len(items) == 1
        assert "passed" in items[0].content
        assert "피보나치" in items[0].content

    def test_episodic_memory_failed_has_warning_tag(self):
        session = CLISession()
        _save_episodic_memory(session, "버그 수정", "타입 오류 발생", passed=False)
        items = session.memory.list_by_type(MemoryType.EPISODIC, session_id=session.session_id)
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

        async def fake_astream(input_state):
            yield {"parse_request": {"parse_result": {"task_type": "generate"}}}
            yield {"execute_code": {"generated_code": "code", "messages": [ai_msg]}}
            yield {"verify_result": {"verify_result": {"passed": True, "issues": []}}}

        fake_graph.astream = fake_astream
        fake_agent = MagicMock(graph=fake_graph)
        session.cache_agent("coding_assistant", fake_agent)

        await _run_agent_turn("함수 작성", session, renderer)

        items = session.memory.list_by_type(MemoryType.EPISODIC, session_id=session.session_id)
        assert len(items) == 1
        assert "passed" in items[0].content


# ── Eval Pipeline ──


class TestEvalCommands:
    def test_eval_command_handled(self):
        mock_result = EvalResult(success=False, error_message="테스트 환경")
        with patch("youngs75_a2a.cli.eval_runner._run_evaluation_sync", return_value=mock_result):
            result = handle_command("/eval", CLISession(), _make_renderer())
        assert result.handled

    def test_eval_status_command_handled(self):
        with patch("youngs75_a2a.cli.commands.load_last_eval_results",
                    return_value=EvalResult(success=False, error_message="결과 없음")):
            result = handle_command("/eval status", CLISession(), _make_renderer())
        assert result.handled

    def test_eval_unknown_subcommand(self):
        result = handle_command("/eval unknown", CLISession(), _make_renderer())
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
            success=True, total=3, passed=2, failed=1,
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
        with patch("youngs75_a2a.cli.eval_runner._DEFAULT_RESULTS_FILE",
                    Path("/nonexistent/eval_results.json")):
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
