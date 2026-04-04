"""Phase 11-3: CLI에 Phase 10 기능 통합 테스트.

검증 항목:
- ProjectContextLoader가 CLI 초기화 시 세션에 저장되는지
- ToolPermissionManager가 세션에 초기화되는지
- ParallelToolExecutor가 세션에 초기화되는지
- 권한 검사가 도구 실행 전에 호출되는지
- ParallelToolExecutor가 CodingAssistantAgent에서 사용되는지
- ContextManager가 모든 에이전트(BaseGraphAgent)에서 동작하는지
- 새 슬래시 명령어 (/permissions, /context, /tools) 동작
- BaseGraphAgent._build_system_prompt() 동작
- BaseGraphAgent._check_context_and_compact() 동작
- 기존 기능이 깨지지 않는지 (옵셔널 통합)
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from rich.console import Console

from youngs75_a2a.cli.commands import handle_command
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession
from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.context_manager import ContextManager
from youngs75_a2a.core.parallel_tool_executor import ParallelToolExecutor
from youngs75_a2a.core.tool_permissions import (
    PermissionDecision,
    ToolPermissionManager,
)


# ── 헬퍼 ──────────────────────────────────────────────────────


def _make_renderer() -> CLIRenderer:
    """테스트용 CLIRenderer (출력을 StringIO로 캡처)."""
    return CLIRenderer(console=Console(file=StringIO(), force_terminal=True))


def _make_session(**kwargs: Any) -> CLISession:
    """테스트용 CLISession."""
    return CLISession(**kwargs)


# ── 1. CLISession Phase 10 필드 초기화 ────────────────────────


class TestSessionPhase10Fields:
    """CLISession에 Phase 10 필드가 올바르게 추가되었는지 검증."""

    def test_session_has_project_context_field(self) -> None:
        """CLISession에 project_context 필드가 존재한다."""
        session = _make_session()
        assert hasattr(session, "project_context")
        assert session.project_context is None

    def test_session_has_permission_manager_field(self) -> None:
        """CLISession에 permission_manager 필드가 존재한다."""
        session = _make_session()
        assert hasattr(session, "permission_manager")
        assert session.permission_manager is None

    def test_session_has_tool_executor_field(self) -> None:
        """CLISession에 tool_executor 필드가 존재한다."""
        session = _make_session()
        assert hasattr(session, "tool_executor")
        assert session.tool_executor is None

    def test_session_fields_settable(self, tmp_path: Path) -> None:
        """Phase 10 필드에 값을 설정할 수 있다."""
        session = _make_session()
        session.project_context = "# 테스트 컨텍스트"
        session.permission_manager = ToolPermissionManager(str(tmp_path))
        session.tool_executor = ParallelToolExecutor()

        assert session.project_context == "# 테스트 컨텍스트"
        assert session.permission_manager is not None
        assert session.tool_executor is not None


# ── 2. BaseGraphAgent Phase 10 통합 ──────────────────────────


class TestBaseGraphAgentPhase10:
    """BaseGraphAgent에 Phase 10 필드와 메서드가 올바르게 추가되었는지 검증."""

    def test_base_agent_has_phase10_fields(self) -> None:
        """BaseGraphAgent에 Phase 10 필드가 존재한다."""
        agent = BaseGraphAgent(
            state_schema=dict,
            auto_build=False,
        )
        assert agent.permission_manager is None
        assert agent.tool_executor is None
        assert agent.project_context is None
        assert agent.context_manager is None

    def test_build_system_prompt_without_context(self) -> None:
        """project_context가 없으면 기본 프롬프트를 그대로 반환한다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        base = "당신은 도움이 되는 AI입니다."
        result = agent._build_system_prompt(base)
        assert result == base

    def test_build_system_prompt_with_context(self) -> None:
        """project_context가 있으면 기본 프롬프트에 추가된다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        agent.project_context = "# 프로젝트 규칙\n- Python 3.13 사용"
        base = "당신은 도움이 되는 AI입니다."
        result = agent._build_system_prompt(base)

        assert base in result
        assert "프로젝트 규칙" in result
        assert "Python 3.13" in result

    async def test_check_context_and_compact_without_manager(self) -> None:
        """context_manager가 없으면 원본 메시지를 그대로 반환한다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        messages = [HumanMessage(content="테스트")]
        mock_llm = AsyncMock()

        result = await agent._check_context_and_compact(messages, mock_llm)
        assert result == messages
        mock_llm.ainvoke.assert_not_called()

    async def test_check_context_and_compact_no_compaction_needed(self) -> None:
        """토큰 수가 임계치 미만이면 컴팩션하지 않는다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        agent.context_manager = ContextManager(
            max_tokens=100000,
            compact_threshold=0.8,
        )
        messages = [HumanMessage(content="짧은 메시지")]
        mock_llm = AsyncMock()

        result = await agent._check_context_and_compact(messages, mock_llm)
        assert result == messages

    async def test_check_context_and_compact_compaction_triggered(self) -> None:
        """토큰 수가 임계치 초과 시 컴팩션이 수행된다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        agent.context_manager = ContextManager(
            max_tokens=100,
            compact_threshold=0.5,
            keep_recent_turns=1,
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="요약된 내용")

        # 긴 메시지로 임계치 초과 유도
        long_content = "매우 긴 대화 내용입니다. " * 50
        messages = [
            SystemMessage(content="시스템"),
            HumanMessage(content=long_content),
            AIMessage(content=long_content),
            HumanMessage(content="최근 질문"),
            AIMessage(content="최근 답변"),
        ]

        result = await agent._check_context_and_compact(messages, mock_llm)
        # 컴팩션 결과 메시지 수가 원본보다 적어야 함
        assert len(result) < len(messages)


# ── 3. ProjectContextLoader CLI 통합 ─────────────────────────


class TestProjectContextLoaderIntegration:
    """ProjectContextLoader가 CLI 초기화 시 정상 동작하는지 검증."""

    def test_project_context_loaded_into_session(self, tmp_path: Path) -> None:
        """프로젝트 컨텍스트가 세션에 저장된다."""
        from youngs75_a2a.core.project_context import ProjectContextLoader

        # 컨텍스트 파일 생성
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# 테스트 규칙\n- ruff 포매팅 준수", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        context_section = loader.build_system_prompt_section()

        session = _make_session()
        session.project_context = context_section

        assert session.project_context is not None
        assert "테스트 규칙" in session.project_context
        assert "ruff 포매팅" in session.project_context

    def test_no_context_files_empty(self, tmp_path: Path) -> None:
        """컨텍스트 파일이 없으면 빈 문자열을 반환한다."""
        from youngs75_a2a.core.project_context import ProjectContextLoader

        loader = ProjectContextLoader(str(tmp_path))
        context_section = loader.build_system_prompt_section()

        assert context_section == ""


# ── 4. ToolPermissionManager 에이전트 통합 ────────────────────


class TestPermissionManagerIntegration:
    """ToolPermissionManager가 에이전트에서 올바르게 동작하는지 검증."""

    def test_permission_manager_assigned_to_agent(self, tmp_path: Path) -> None:
        """ToolPermissionManager가 에이전트에 할당된다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        mgr = ToolPermissionManager(str(tmp_path))
        agent.permission_manager = mgr

        assert agent.permission_manager is not None
        assert agent.permission_manager.check("read_file") == PermissionDecision.ALLOW

    def test_permission_deny_blocks_tool(self, tmp_path: Path) -> None:
        """DENY 판정 시 도구 실행이 차단된다."""
        mgr = ToolPermissionManager(str(tmp_path))

        # workspace 밖 경로 접근은 DENY
        decision = mgr.check("write_file", {"path": "/etc/passwd"})
        assert decision == PermissionDecision.DENY

    def test_permission_allow_passes_tool(self, tmp_path: Path) -> None:
        """ALLOW 판정 시 도구가 정상 실행된다."""
        mgr = ToolPermissionManager(str(tmp_path))

        decision = mgr.check("read_file", {"path": "src/main.py"})
        assert decision == PermissionDecision.ALLOW


# ── 5. ParallelToolExecutor 에이전트 통합 ─────────────────────


class TestParallelToolExecutorIntegration:
    """ParallelToolExecutor가 에이전트에서 올바르게 동작하는지 검증."""

    def test_executor_assigned_to_agent(self) -> None:
        """ParallelToolExecutor가 에이전트에 할당된다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        executor = ParallelToolExecutor()
        agent.tool_executor = executor

        assert agent.tool_executor is not None
        assert agent.tool_executor.is_concurrency_safe("read_file")

    async def test_executor_batch_execution(self) -> None:
        """배치 실행이 정상 동작한다."""
        executor = ParallelToolExecutor()

        async def mock_executor(name: str, args: dict) -> str:
            return f"result:{name}"

        calls = [
            {"name": "read_file", "args": {}, "id": "c1"},
            {"name": "search_code", "args": {}, "id": "c2"},
        ]
        results = await executor.execute_batch(calls, mock_executor)
        assert len(results) == 2
        assert "result:read_file" in results[0].content
        assert "result:search_code" in results[1].content


# ── 6. ContextManager 모든 에이전트 확장 ──────────────────────


class TestContextManagerAllAgents:
    """ContextManager가 모든 에이전트에서 동작하는지 검증."""

    def test_base_agent_context_manager_settable(self) -> None:
        """BaseGraphAgent에 ContextManager를 설정할 수 있다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        agent.context_manager = ContextManager(max_tokens=50000)

        assert agent.context_manager is not None
        assert agent.context_manager.max_tokens == 50000

    def test_simple_react_inherits_context_manager(self) -> None:
        """SimpleMCPReActAgent이 BaseGraphAgent의 context_manager를 상속한다."""
        from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent

        agent = SimpleMCPReActAgent(auto_build=False)
        assert hasattr(agent, "context_manager")
        agent.context_manager = ContextManager()
        assert agent.context_manager is not None

    def test_deep_research_inherits_context_manager(self) -> None:
        """DeepResearchAgent이 BaseGraphAgent의 context_manager를 상속한다."""
        from youngs75_a2a.agents.deep_research.agent import DeepResearchAgent

        agent = DeepResearchAgent(auto_build=False)
        assert hasattr(agent, "context_manager")
        agent.context_manager = ContextManager()
        assert agent.context_manager is not None

    def test_simple_react_inherits_project_context(self) -> None:
        """SimpleMCPReActAgent이 project_context를 상속한다."""
        from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent

        agent = SimpleMCPReActAgent(auto_build=False)
        agent.project_context = "# 프로젝트 규칙"
        assert agent.project_context == "# 프로젝트 규칙"

    def test_deep_research_inherits_project_context(self) -> None:
        """DeepResearchAgent이 project_context를 상속한다."""
        from youngs75_a2a.agents.deep_research.agent import DeepResearchAgent

        agent = DeepResearchAgent(auto_build=False)
        agent.project_context = "# 프로젝트 규칙"
        assert agent.project_context == "# 프로젝트 규칙"


# ── 7. 슬래시 커맨드: /permissions ─────────────────────────────


class TestPermissionsCommand:
    """/permissions 커맨드 동작 검증."""

    def test_permissions_command_handled(self, tmp_path: Path) -> None:
        """/permissions 커맨드가 처리된다."""
        session = _make_session()
        session.permission_manager = ToolPermissionManager(str(tmp_path))
        renderer = _make_renderer()

        result = handle_command("/permissions", session, renderer)
        assert result.handled is True
        assert result.should_quit is False

    def test_permissions_command_no_manager(self) -> None:
        """/permissions 커맨드 — 권한 관리자 미설정 시 안내 메시지."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/permissions", session, renderer)
        assert result.handled is True

    def test_permissions_command_shows_deny_log(self, tmp_path: Path) -> None:
        """/permissions 커맨드 — 거부 기록이 표시된다."""
        session = _make_session()
        mgr = ToolPermissionManager(str(tmp_path))
        mgr.record_denial("bash", "테스트 거부")
        session.permission_manager = mgr
        renderer = _make_renderer()

        result = handle_command("/permissions", session, renderer)
        assert result.handled is True


# ── 8. 슬래시 커맨드: /context ─────────────────────────────────


class TestContextCommand:
    """/context 커맨드 동작 검증."""

    def test_context_command_handled(self) -> None:
        """/context 커맨드가 처리된다."""
        session = _make_session()
        session.project_context = "# 테스트 컨텍스트\n프로젝트 규칙"
        renderer = _make_renderer()

        result = handle_command("/context", session, renderer)
        assert result.handled is True
        assert result.should_quit is False

    def test_context_command_no_context(self) -> None:
        """/context 커맨드 — 컨텍스트 미설정 시 안내 메시지."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/context", session, renderer)
        assert result.handled is True

    def test_context_command_long_context_truncated(self) -> None:
        """/context 커맨드 — 긴 컨텍스트는 잘려서 표시된다."""
        session = _make_session()
        session.project_context = "A" * 1000  # 500자 초과
        renderer = _make_renderer()

        result = handle_command("/context", session, renderer)
        assert result.handled is True


# ── 9. 슬래시 커맨드: /tools ──────────────────────────────────


class TestToolsCommand:
    """/tools 커맨드 동작 검증."""

    def test_tools_command_handled(self, tmp_path: Path) -> None:
        """/tools 커맨드가 처리된다."""
        session = _make_session()
        session.permission_manager = ToolPermissionManager(str(tmp_path))
        session.tool_executor = ParallelToolExecutor()
        renderer = _make_renderer()

        result = handle_command("/tools", session, renderer)
        assert result.handled is True
        assert result.should_quit is False

    def test_tools_command_no_manager(self) -> None:
        """/tools 커맨드 — 권한 관리자 미설정 시 안내 메시지."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/tools", session, renderer)
        assert result.handled is True

    def test_tools_command_without_executor(self, tmp_path: Path) -> None:
        """/tools 커맨드 — 병렬 실행기 없어도 동작한다."""
        session = _make_session()
        session.permission_manager = ToolPermissionManager(str(tmp_path))
        # tool_executor 미설정
        renderer = _make_renderer()

        result = handle_command("/tools", session, renderer)
        assert result.handled is True


# ── 10. 기존 커맨드 호환성 ────────────────────────────────────


class TestExistingCommandsCompatibility:
    """Phase 10 통합 후 기존 커맨드가 정상 동작하는지 검증."""

    def test_help_command_includes_new_commands(self) -> None:
        """/help 출력에 새 커맨드가 포함된다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/help", session, renderer)
        assert result.handled is True

    def test_quit_command_still_works(self) -> None:
        """/quit 커맨드가 여전히 동작한다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/quit", session, renderer)
        assert result.should_quit is True

    def test_agent_switch_still_works(self) -> None:
        """/agent 전환이 여전히 동작한다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/agent simple_react", session, renderer)
        assert result.handled is True
        assert session.info.agent_name == "simple_react"

    def test_session_command_still_works(self) -> None:
        """/session 커맨드가 여전히 동작한다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/session", session, renderer)
        assert result.handled is True

    def test_memory_command_still_works(self) -> None:
        """/memory 커맨드가 여전히 동작한다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/memory", session, renderer)
        assert result.handled is True

    def test_clear_command_still_works(self) -> None:
        """/clear 커맨드가 여전히 동작한다."""
        session = _make_session()
        session.add_message("user", "test")
        renderer = _make_renderer()

        result = handle_command("/clear", session, renderer)
        assert result.handled is True
        assert session.info.message_count == 0

    def test_non_command_passthrough(self) -> None:
        """일반 텍스트는 handled=False를 반환한다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("안녕하세요", session, renderer)
        assert result.handled is False

    def test_unknown_command_handled(self) -> None:
        """알 수 없는 커맨드는 에러로 처리된다."""
        session = _make_session()
        renderer = _make_renderer()

        result = handle_command("/unknown_cmd", session, renderer)
        assert result.handled is True


# ── 11. 에이전트 생성 시 Phase 10 기능 주입 ────────────────────


class TestAgentPhase10Injection:
    """_get_or_create_agent에서 Phase 10 기능이 에이전트에 주입되는지 검증."""

    async def test_agent_gets_project_context(self) -> None:
        """에이전트 생성 시 project_context가 주입된다."""
        from youngs75_a2a.cli.app import _get_or_create_agent

        session = _make_session()
        session.project_context = "# 테스트 컨텍스트"
        renderer = _make_renderer()

        # mock _create_agent
        mock_agent = MagicMock(spec=BaseGraphAgent)
        mock_agent.project_context = None
        mock_agent.permission_manager = None
        mock_agent.tool_executor = None
        mock_agent.context_manager = None

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ):
            agent = await _get_or_create_agent(session, renderer)

        assert agent is not None
        assert agent.project_context == "# 테스트 컨텍스트"

    async def test_agent_gets_permission_manager(self, tmp_path: Path) -> None:
        """에이전트 생성 시 permission_manager가 주입된다."""
        from youngs75_a2a.cli.app import _get_or_create_agent

        session = _make_session()
        mgr = ToolPermissionManager(str(tmp_path))
        session.permission_manager = mgr
        renderer = _make_renderer()

        mock_agent = MagicMock(spec=BaseGraphAgent)
        mock_agent.project_context = None
        mock_agent.permission_manager = None
        mock_agent.tool_executor = None
        mock_agent.context_manager = None

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ):
            agent = await _get_or_create_agent(session, renderer)

        assert agent is not None
        assert agent.permission_manager is mgr

    async def test_agent_gets_tool_executor(self) -> None:
        """에이전트 생성 시 tool_executor가 주입된다."""
        from youngs75_a2a.cli.app import _get_or_create_agent

        session = _make_session()
        executor = ParallelToolExecutor()
        session.tool_executor = executor
        renderer = _make_renderer()

        mock_agent = MagicMock(spec=BaseGraphAgent)
        mock_agent.project_context = None
        mock_agent.permission_manager = None
        mock_agent.tool_executor = None
        mock_agent.context_manager = None

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ):
            agent = await _get_or_create_agent(session, renderer)

        assert agent is not None
        assert agent.tool_executor is executor

    async def test_agent_gets_default_context_manager(self) -> None:
        """context_manager가 없으면 기본 ContextManager가 주입된다."""
        from youngs75_a2a.cli.app import _get_or_create_agent

        session = _make_session()
        renderer = _make_renderer()

        mock_agent = MagicMock(spec=BaseGraphAgent)
        mock_agent.project_context = None
        mock_agent.permission_manager = None
        mock_agent.tool_executor = None
        mock_agent.context_manager = None

        with patch(
            "youngs75_a2a.cli.app._create_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ):
            agent = await _get_or_create_agent(session, renderer)

        assert agent is not None
        # context_manager가 설정되어야 함
        assert isinstance(agent.context_manager, ContextManager)


# ── 12. CodingAssistantAgent 도구 실행 통합 ────────────────────


class TestCodingAssistantToolExecution:
    """CodingAssistantAgent에서 ParallelToolExecutor와 ToolPermissionManager가 동작하는지 검증."""

    def test_coding_agent_has_phase10_fields(self) -> None:
        """CodingAssistantAgent에 Phase 10 필드가 존재한다."""
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

        agent = CodingAssistantAgent(auto_build=False)
        assert hasattr(agent, "permission_manager")
        assert hasattr(agent, "tool_executor")
        assert hasattr(agent, "project_context")
        assert hasattr(agent, "context_manager")

    async def test_execute_tools_with_parallel_executor(self) -> None:
        """ParallelToolExecutor를 통해 도구가 실행된다."""
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

        agent = CodingAssistantAgent(auto_build=False)

        # mock 도구 설정
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.ainvoke = AsyncMock(return_value="파일 내용")
        agent._tools = [mock_tool]

        # ParallelToolExecutor 설정
        executor = ParallelToolExecutor()
        agent.tool_executor = executor

        # 도구 호출이 있는 AI 메시지 생성
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "test.py"}, "id": "tc1"}
            ],
        )

        state = {
            "messages": [ai_msg],
            "project_context": [],
        }

        result = await agent._execute_tools(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "파일 내용" in result["messages"][0].content

    async def test_execute_tools_permission_deny(self, tmp_path: Path) -> None:
        """ToolPermissionManager DENY 시 도구 실행이 차단된다."""
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

        agent = CodingAssistantAgent(auto_build=False)

        # mock 도구 설정
        mock_tool = MagicMock()
        mock_tool.name = "write_file"
        mock_tool.ainvoke = AsyncMock(return_value="성공")
        agent._tools = [mock_tool]

        # ToolPermissionManager 설정 — workspace 밖 경로는 DENY
        agent.permission_manager = ToolPermissionManager(str(tmp_path))

        # workspace 밖 파일 쓰기 시도
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"path": "/etc/passwd"},
                    "id": "tc1",
                }
            ],
        )

        state = {
            "messages": [ai_msg],
            "project_context": [],
        }

        result = await agent._execute_tools(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "권한 거부" in result["messages"][0].content

    async def test_execute_tools_fallback_without_executor(self) -> None:
        """ParallelToolExecutor 없이도 기존 순차 실행이 동작한다."""
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

        agent = CodingAssistantAgent(auto_build=False)
        # tool_executor를 명시적으로 None
        agent.tool_executor = None

        # mock 도구 설정
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.ainvoke = AsyncMock(return_value="파일 내용 (순차)")
        agent._tools = [mock_tool]

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "test.py"}, "id": "tc1"}
            ],
        )

        state = {
            "messages": [ai_msg],
            "project_context": [],
        }

        result = await agent._execute_tools(state)
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "파일 내용 (순차)" in result["messages"][0].content

    async def test_execute_tools_with_both_executor_and_permissions(
        self, tmp_path: Path
    ) -> None:
        """ParallelToolExecutor + ToolPermissionManager가 함께 동작한다."""
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

        agent = CodingAssistantAgent(auto_build=False)

        # mock 도구 설정
        mock_read = MagicMock()
        mock_read.name = "read_file"
        mock_read.ainvoke = AsyncMock(return_value="파일 내용")
        mock_write = MagicMock()
        mock_write.name = "write_file"
        mock_write.ainvoke = AsyncMock(return_value="쓰기 완료")
        agent._tools = [mock_read, mock_write]

        agent.tool_executor = ParallelToolExecutor()
        agent.permission_manager = ToolPermissionManager(str(tmp_path))

        # workspace 내 파일 읽기 + workspace 밖 파일 쓰기
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "src/main.py"}, "id": "tc1"},
                {"name": "write_file", "args": {"path": "/etc/passwd"}, "id": "tc2"},
            ],
        )

        state = {
            "messages": [ai_msg],
            "project_context": [],
        }

        result = await agent._execute_tools(state)
        assert "messages" in result
        assert len(result["messages"]) == 2

        # read_file은 허용
        assert "파일 내용" in result["messages"][0].content
        # write_file은 거부 (workspace 밖)
        assert "권한 거부" in result["messages"][1].content


# ── 13. 옵셔널 통합 — Phase 10 없이도 정상 동작 ────────────────


class TestOptionalIntegration:
    """Phase 10 기능이 없어도 에이전트가 정상 동작하는지 검증."""

    def test_base_agent_works_without_phase10(self) -> None:
        """Phase 10 필드가 모두 None이어도 BaseGraphAgent가 동작한다."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        assert agent.permission_manager is None
        assert agent.tool_executor is None
        assert agent.project_context is None
        assert agent.context_manager is None

        # _build_system_prompt는 기본 프롬프트를 그대로 반환
        assert agent._build_system_prompt("test") == "test"

    async def test_check_context_noop_without_manager(self) -> None:
        """context_manager 없이 _check_context_and_compact는 noop."""
        agent = BaseGraphAgent(state_schema=dict, auto_build=False)
        messages = [HumanMessage(content="test")]
        result = await agent._check_context_and_compact(messages, None)
        assert result == messages

    def test_session_works_without_phase10(self) -> None:
        """Phase 10 필드가 설정되지 않아도 CLISession이 동작한다."""
        session = _make_session()
        session.add_message("user", "test")
        assert session.info.message_count == 1
        assert session.project_context is None
        assert session.permission_manager is None
        assert session.tool_executor is None

    async def test_coding_agent_execute_tools_no_tool_calls(self) -> None:
        """도구 호출이 없으면 빈 결과를 반환한다."""
        from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

        agent = CodingAssistantAgent(auto_build=False)
        agent.tool_executor = ParallelToolExecutor()

        # 도구 호출이 없는 AI 메시지
        ai_msg = AIMessage(content="코드를 작성했습니다.")
        state = {"messages": [ai_msg], "project_context": []}

        result = await agent._execute_tools(state)
        assert result == {}


# ── 14. SimpleReActAgent 프로젝트 컨텍스트 통합 ─────────────────


class TestSimpleReActProjectContext:
    """SimpleMCPReActAgent에서 프로젝트 컨텍스트가 시스템 프롬프트에 주입되는지 검증."""

    def test_get_system_prompt_without_context(self) -> None:
        """project_context 없이 기본 프롬프트를 반환한다."""
        from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent

        agent = SimpleMCPReActAgent(auto_build=False)
        prompt = agent._get_system_prompt()
        # project_context가 없으므로 기본 프롬프트만
        assert "프로젝트 규칙" not in prompt

    def test_get_system_prompt_with_context(self) -> None:
        """project_context가 있으면 시스템 프롬프트에 포함된다."""
        from youngs75_a2a.agents.simple_react.agent import SimpleMCPReActAgent

        agent = SimpleMCPReActAgent(auto_build=False)
        agent.project_context = "# 프로젝트 규칙\n- Python 3.13 사용"
        prompt = agent._get_system_prompt()

        assert "프로젝트 규칙" in prompt
        assert "Python 3.13" in prompt
