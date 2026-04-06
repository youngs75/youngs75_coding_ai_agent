"""실행 기반 검증 (_run_tests) 단위 테스트."""

import os
import tempfile

import pytest

from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
from youngs75_a2a.agents.coding_assistant.config import CodingConfig


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


class TestRunTestsWithPytest:
    """pytest 실행 단계 테스트."""

    @pytest.mark.asyncio
    async def test_passing_test_passes(self, agent, tmp_path):
        """통과하는 테스트는 test_passed=True."""
        (tmp_path / "utils.py").write_text("def add(a, b): return a + b\n")
        (tmp_path / "test_utils.py").write_text(
            "from utils import add\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["utils.py (+1 lines)", "test_utils.py (+3 lines)"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is True
        os.environ.pop("CODE_TOOLS_WORKSPACE", None)

    @pytest.mark.asyncio
    async def test_failing_test_fails(self, agent, tmp_path):
        """실패하는 테스트는 test_passed=False + 에러 출력."""
        (tmp_path / "utils.py").write_text("def add(a, b): return a - b\n")  # 일부러 버그
        (tmp_path / "test_utils.py").write_text(
            "from utils import add\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
        os.environ["CODE_TOOLS_WORKSPACE"] = str(tmp_path)

        result = await agent._run_tests({
            "written_files": ["utils.py (+1 lines)", "test_utils.py (+3 lines)"],
            "execution_log": [],
            "iteration": 0,
        })

        assert result["test_passed"] is False
        assert "failed" in result["test_output"].lower() or "FAILED" in result["test_output"]
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
        assert "테스트 파일 없음" in result["test_output"]
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

    def test_failed_with_budget_returns_generate_final(self, agent):
        result = agent._should_retry_tests({
            "test_passed": False,
            "iteration": 1,
            "max_iterations": 3,
        })
        assert result == agent.get_node_name("GENERATE_FINAL")

    def test_failed_max_iterations_returns_end(self, agent):
        result = agent._should_retry_tests({
            "test_passed": False,
            "iteration": 3,
            "max_iterations": 3,
        })
        assert result == "__end__"


class TestInstallDependencies:
    """의존성 자동 설치 테스트."""

    @pytest.mark.asyncio
    async def test_requirements_txt_with_venv(self, agent, tmp_path):
        """requirements.txt + venv_python이 있으면 pip install 실행 시도."""
        (tmp_path / "requirements.txt").write_text("requests\n")
        log: list[str] = []
        # 실제 venv_python 경로 전달 (존재 여부에 관계없이 로그만 확인)
        fake_python = str(tmp_path / ".venv" / "bin" / "python")
        await agent._install_dependencies(str(tmp_path), fake_python, log)
        assert any("install" in entry for entry in log)

    @pytest.mark.asyncio
    async def test_no_dep_file_no_log(self, agent, tmp_path):
        """의존성 파일이 없으면 로그 없음."""
        log: list[str] = []
        await agent._install_dependencies(str(tmp_path), None, log)
        assert len(log) == 0

    @pytest.mark.asyncio
    async def test_package_json_detected(self, agent, tmp_path):
        """package.json이 있으면 npm install 시도."""
        (tmp_path / "package.json").write_text('{"name": "test", "dependencies": {}}\n')
        log: list[str] = []
        await agent._install_dependencies(str(tmp_path), None, log)
        assert any("npm" in entry for entry in log)


class TestDetectRuntimes:
    """런타임 감지 테스트."""

    @pytest.mark.asyncio
    async def test_python_detected(self, agent):
        log: list[str] = []
        result = await agent._detect_runtimes(["python"], log)
        assert result["python"] is True
        assert any("python" in entry and "✓" in entry for entry in log)

    @pytest.mark.asyncio
    async def test_missing_runtime(self, agent):
        log: list[str] = []
        result = await agent._detect_runtimes(["rust"], log)
        # Rust가 설치 안 되어 있을 수도 있으므로 결과 자체만 확인
        assert "rust" in result


class TestSetupProjectEnv:
    """venv 생성 테스트."""

    @pytest.mark.asyncio
    async def test_creates_venv(self, agent, tmp_path):
        """requirements.txt가 있으면 venv 생성."""
        (tmp_path / "requirements.txt").write_text("flask\n")
        log: list[str] = []
        venv_python = await agent._setup_project_env(str(tmp_path), log)
        if venv_python:
            assert ".venv" in venv_python
            assert os.path.exists(venv_python)
            assert any("✓" in entry for entry in log)

    @pytest.mark.asyncio
    async def test_no_deps_returns_none(self, agent, tmp_path):
        """의존성 파일 없으면 venv 생성 안 함."""
        log: list[str] = []
        result = await agent._setup_project_env(str(tmp_path), log)
        assert result is None

    @pytest.mark.asyncio
    async def test_reuses_existing_venv(self, agent, tmp_path):
        """이미 venv가 있으면 재사용."""
        venv_dir = tmp_path / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        (venv_dir / "python").write_text("#!/bin/sh\n")
        (venv_dir / "python").chmod(0o755)
        log: list[str] = []
        result = await agent._setup_project_env(str(tmp_path), log)
        assert result is not None
        assert "재사용" in log[0]


class TestCheckCommandExists:
    """_check_command_exists 테스트."""

    @pytest.mark.asyncio
    async def test_python_exists(self):
        """python은 반드시 존재."""
        result = await CodingAssistantAgent._check_command_exists("python3")
        assert result is True

    @pytest.mark.asyncio
    async def test_nonexistent_command(self):
        """존재하지 않는 명령어는 False."""
        result = await CodingAssistantAgent._check_command_exists("totally_fake_command_xyz")
        assert result is False


class TestRunShellMCPTool:
    """run_shell MCP 도구 보안 테스트."""

    def test_allowed_command(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("echo hello")
        assert "hello" in result
        assert "[exit_code] 0" in result

    def test_blocked_command(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("curl http://example.com")
        assert "Error" in result
        assert "허용되지 않은" in result

    def test_blocked_pattern(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("rm -rf /")
        assert "Error" in result
        assert "차단" in result

    def test_empty_command(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("")
        assert "Error" in result

    def test_pip_allowed(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("pip --version")
        assert "[exit_code]" in result

    def test_pipe_allowed(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("echo test | grep test")
        assert "[exit_code] 0" in result

    def test_pipe_blocked(self):
        from youngs75_a2a.mcp_servers.code_tools.server import run_shell
        result = run_shell("echo test | curl http://x")
        assert "Error" in result
