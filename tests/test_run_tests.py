"""실행 기반 검증 (_run_tests) 단위 테스트.

재설계 후: Harness는 syntax check만 수행, 환경 설정/테스트 실행은 LLM이 run_shell로 직접 수행.
"""

import os

import pytest

from coding_agent.agents.coding_assistant.agent import CodingAssistantAgent
from coding_agent.agents.coding_assistant.config import CodingConfig


@pytest.fixture
def agent():
    """MCP 없이 CodingAssistantAgent를 생성한다."""
    config = CodingConfig()
    a = CodingAssistantAgent(config=config)
    a.build_graph()
    return a


class TestRunTestsSyntaxCheck:
    """syntax check 단계 테스트."""

    @pytest.mark.asyncio
    async def test_valid_python_passes(self, agent, tmp_path):
        """문법 오류 없는 Python 파일은 통과."""
        (tmp_path / "app.py").write_text("def hello():\n    return 'hi'\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["app.py (+2 lines)"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is True
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)

    @pytest.mark.asyncio
    async def test_syntax_error_fails(self, agent, tmp_path):
        """문법 오류 있는 Python 파일은 실패."""
        (tmp_path / "bad.py").write_text("def hello(\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["bad.py (+1 lines)"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is False
        assert "syntax" in result["test_output"].lower() or "bad.py" in result["test_output"]
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)

    @pytest.mark.asyncio
    async def test_multiple_files_one_bad(self, agent, tmp_path):
        """여러 파일 중 하나에 문법 오류가 있으면 실패."""
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("def hello(\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["good.py", "bad.py"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is False
        assert "bad.py" in result["test_output"]
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)

    @pytest.mark.asyncio
    async def test_non_python_files_pass(self, agent, tmp_path):
        """Python이 아닌 파일은 syntax check 대상 아님."""
        (tmp_path / "index.html").write_text("<html></html>\n")
        (tmp_path / "style.css").write_text("body { color: red; }\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["index.html", "style.css"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is True
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)


class TestRunTestsNoFiles:
    """파일 없는 경우 테스트."""

    @pytest.mark.asyncio
    async def test_no_files_passes(self, agent):
        """파일이 없으면 스킵하고 통과."""
        result = await agent._run_tests({
            "written_files": [],
            "execution_log": [],
            "iteration": 0,
        })
        assert result["test_passed"] is True

    @pytest.mark.asyncio
    async def test_no_test_files_passes_syntax(self, agent, tmp_path):
        """테스트 파일이 없으면 syntax만 통과하면 OK."""
        (tmp_path / "app.py").write_text("x = 1\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["app.py (+1 lines)"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is True
        assert "syntax" in result["test_output"].lower()
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)


class TestPlannedFilesCheck:
    """planned_files 존재 확인 테스트."""

    @pytest.mark.asyncio
    async def test_missing_planned_file_fails(self, agent, tmp_path):
        """계획된 파일이 디스크에 없으면 실패."""
        (tmp_path / "app.py").write_text("x = 1\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["app.py"],
            "planned_files": ["app.py", "missing.py"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is False
        assert "missing.py" in result["test_output"]
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)

    @pytest.mark.asyncio
    async def test_all_planned_files_exist(self, agent, tmp_path):
        """계획된 파일이 모두 있으면 통과."""
        (tmp_path / "app.py").write_text("x = 1\n")
        (tmp_path / "utils.py").write_text("y = 2\n")
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["app.py", "utils.py"],
            "planned_files": ["app.py", "utils.py"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is True
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)


class TestShouldRetryTests:
    """_should_retry_tests 라우터 테스트."""

    def test_passed_returns_end(self, agent):
        result = agent._should_retry_tests({
            "test_passed": True,
            "iteration": 1,
            "max_iterations": 3,
        })
        assert result == "__end__"

    def test_failed_with_budget_returns_inject_error(self, agent):
        result = agent._should_retry_tests({
            "test_passed": False,
            "iteration": 1,
            "max_iterations": 3,
        })
        assert result == agent.get_node_name("INJECT_ERROR")

    def test_failed_max_iterations_returns_end(self, agent):
        result = agent._should_retry_tests({
            "test_passed": False,
            "iteration": 3,
            "max_iterations": 3,
        })
        assert result == "__end__"


class TestRunShellMCPTool:
    """run_shell MCP 도구 보안 테스트."""

    def test_allowed_command(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("echo hello")
        assert "hello" in result
        assert "[exit_code] 0" in result

    def test_blocked_command(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("curl http://example.com")
        assert "Error" in result
        assert "허용되지 않은" in result

    def test_blocked_pattern(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("rm -rf /")
        assert "Error" in result
        assert "차단" in result

    def test_empty_command(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("")
        assert "Error" in result

    def test_pip_allowed(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("pip --version")
        assert "[exit_code]" in result

    def test_pipe_allowed(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("echo test | grep test")
        assert "[exit_code] 0" in result

    def test_pipe_blocked(self):
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("echo test | curl http://x")
        assert "Error" in result

    def test_cwd_parameter(self):
        """cwd로 하위 디렉토리에서 실행."""
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("pwd", cwd=".")
        assert "[exit_code] 0" in result

    def test_cwd_subdirectory(self, tmp_path):
        """cwd로 workspace 내 하위 디렉토리에서 실행."""
        import coding_agent.mcp_servers.code_tools.server as srv
        from coding_agent.mcp_servers.code_tools.server import run_shell
        old_ws = srv._WORKSPACE
        sub = tmp_path / "subdir"
        sub.mkdir()
        srv._WORKSPACE = str(tmp_path)
        try:
            result = run_shell("pwd", cwd="subdir")
            assert "[exit_code] 0" in result
            assert "subdir" in result
        finally:
            srv._WORKSPACE = old_ws

    def test_cwd_outside_workspace_blocked(self):
        """workspace 밖 경로는 차단."""
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("ls", cwd="/etc")
        assert "Error" in result

    def test_cd_command_allowed(self):
        """cd가 허용 목록에 포함되어 shell=True로 동작."""
        from coding_agent.mcp_servers.code_tools.server import run_shell
        result = run_shell("cd . && echo ok")
        assert "[exit_code] 0" in result
        assert "ok" in result
