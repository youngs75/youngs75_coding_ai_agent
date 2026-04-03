"""CLI 유닛 테스트.

대화형 루프는 테스트하지 않고, 개별 컴포넌트를 검증한다.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from youngs75_a2a.cli.commands import CommandResult, handle_command
from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession


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
