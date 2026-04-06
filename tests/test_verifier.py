"""VerificationAgent 단위 테스트."""

import pytest

from youngs75_a2a.agents.verifier.agent import (
    VerificationAgent,
    _build_lint_command,
    _build_test_command,
    _filter_lintable_files,
    _find_tool,
    _skip_result,
)
from youngs75_a2a.agents.verifier.config import VerifierConfig
from youngs75_a2a.agents.verifier.schemas import (
    CheckResult,
    VerificationResult,
    VerificationState,
)


class TestVerifierConfig:
    """VerifierConfig 기본값 테스트."""

    def test_defaults(self):
        config = VerifierConfig()
        assert config.enable_lint is True
        assert config.enable_test is True
        assert config.enable_llm_review is True
        assert config.lint_timeout == 30
        assert config.test_timeout == 60
        assert config.review_timeout == 30
        assert config.max_review_chars == 4000

    def test_disable_checks(self):
        config = VerifierConfig(
            enable_lint=False,
            enable_test=False,
            enable_llm_review=True,
        )
        assert config.enable_lint is False
        assert config.enable_test is False
        assert config.enable_llm_review is True


class TestSkipResult:
    """_skip_result 헬퍼 테스트."""

    def test_skip_result_passed(self):
        result = _skip_result("lint", "비활성화됨")
        assert result["check_type"] == "lint"
        assert result["passed"] is True
        assert result["output"] == "비활성화됨"
        assert result["issues"] == []


class TestFindTool:
    """_find_tool 헬퍼 테스트."""

    def test_find_existing_tool(self):
        class MockTool:
            name = "run_python"

        tools = [MockTool()]
        assert _find_tool(tools, "run_python") is tools[0]

    def test_find_missing_tool(self):
        assert _find_tool([], "run_python") is None


class TestVerificationSchemas:
    """스키마 테스트."""

    def test_check_result_fields(self):
        result = CheckResult(
            check_type="lint",
            passed=True,
            output="OK",
            issues=[],
        )
        assert result["check_type"] == "lint"
        assert result["passed"] is True

    def test_verification_result_fields(self):
        result = VerificationResult(
            passed=True,
            checks=[],
            issues=[],
            suggestions=[],
            summary="✓ lint | ✓ test | ✓ llm_review",
        )
        assert result["passed"] is True
        assert "lint" in result["summary"]


class TestVerificationAgentInit:
    """VerificationAgent 초기화 테스트."""

    def test_node_names(self):
        assert "LINT" in VerificationAgent.NODE_NAMES
        assert "TEST" in VerificationAgent.NODE_NAMES
        assert "REVIEW" in VerificationAgent.NODE_NAMES
        assert "AGGREGATE" in VerificationAgent.NODE_NAMES


class TestLintCheckDisabled:
    """lint 비활성화 시 스킵 테스트."""

    @pytest.mark.asyncio
    async def test_lint_disabled_skips(self):
        config = VerifierConfig(enable_lint=False)
        agent = VerificationAgent(config=config)
        result = await agent._lint_check({})
        assert result["lint_result"]["passed"] is True
        assert "비활성화" in result["lint_result"]["output"]


class TestTestCheckDisabled:
    """test 비활성화 시 스킵 테스트."""

    @pytest.mark.asyncio
    async def test_test_disabled_skips(self):
        config = VerifierConfig(enable_test=False)
        agent = VerificationAgent(config=config)
        result = await agent._test_check({})
        assert result["test_result"]["passed"] is True
        assert "비활성화" in result["test_result"]["output"]


class TestLLMReviewDisabled:
    """LLM 리뷰 비활성화 시 스킵 테스트."""

    @pytest.mark.asyncio
    async def test_review_disabled_skips(self):
        config = VerifierConfig(enable_llm_review=False)
        agent = VerificationAgent(config=config)
        result = await agent._llm_review({})
        assert result["review_result"]["passed"] is True


class TestAggregateResults:
    """결과 집계 테스트."""

    @pytest.mark.asyncio
    async def test_all_passed(self):
        config = VerifierConfig()
        agent = VerificationAgent(config=config)
        state = {
            "lint_result": CheckResult(
                check_type="lint", passed=True, output="OK", issues=[]
            ),
            "test_result": CheckResult(
                check_type="test", passed=True, output="OK", issues=[]
            ),
            "review_result": CheckResult(
                check_type="llm_review", passed=True, output='{"passed":true,"issues":[],"suggestions":[]}', issues=[]
            ),
        }
        result = await agent._aggregate_results(state)
        vr = result["verification_result"]
        assert vr["passed"] is True
        assert len(vr["checks"]) == 3
        assert "✓" in vr["summary"]

    @pytest.mark.asyncio
    async def test_one_failed(self):
        config = VerifierConfig()
        agent = VerificationAgent(config=config)
        state = {
            "lint_result": CheckResult(
                check_type="lint", passed=False, output="에러", issues=["syntax error"]
            ),
            "test_result": CheckResult(
                check_type="test", passed=True, output="OK", issues=[]
            ),
            "review_result": CheckResult(
                check_type="llm_review", passed=True, output='{"passed":true}', issues=[]
            ),
        }
        result = await agent._aggregate_results(state)
        vr = result["verification_result"]
        assert vr["passed"] is False
        assert "syntax error" in vr["issues"]


class TestMultiLanguageHelpers:
    """멀티언어 헬퍼 함수 테스트."""

    def test_filter_python_files(self):
        files = ["app.py", "main.go", "style.css"]
        assert _filter_lintable_files(files, "python") == ["app.py"]

    def test_filter_go_files(self):
        files = ["app.py", "main.go", "handler.go"]
        assert _filter_lintable_files(files, "go") == ["main.go", "handler.go"]

    def test_filter_js_files(self):
        files = ["index.js", "app.tsx", "main.py"]
        assert _filter_lintable_files(files, "javascript") == ["index.js"]
        assert _filter_lintable_files(files, "typescript") == ["app.tsx"]

    def test_filter_rust_files(self):
        files = ["main.rs", "lib.rs", "README.md"]
        assert _filter_lintable_files(files, "rust") == ["main.rs", "lib.rs"]

    def test_filter_unknown_language_returns_all(self):
        files = ["a.py", "b.go"]
        assert _filter_lintable_files(files, "haskell") == ["a.py", "b.go"]

    def test_build_lint_python(self):
        cmd = _build_lint_command("app.py", "python")
        assert "py_compile" in cmd

    def test_build_lint_javascript(self):
        cmd = _build_lint_command("index.js", "javascript")
        assert "node" in cmd and "--check" in cmd

    def test_build_lint_go(self):
        cmd = _build_lint_command("main.go", "go")
        assert "go" in cmd and "vet" in cmd

    def test_build_lint_rust(self):
        cmd = _build_lint_command("main.rs", "rust")
        assert "cargo" in cmd and "check" in cmd

    def test_build_lint_java(self):
        cmd = _build_lint_command("App.java", "java")
        assert "javac" in cmd

    def test_build_lint_unsupported(self):
        cmd = _build_lint_command("main.hs", "haskell")
        assert "OK" in cmd

    def test_build_test_python(self):
        cmd = _build_test_command(["test_app.py"], "python")
        assert "pytest" in cmd

    def test_build_test_javascript(self):
        cmd = _build_test_command(["app.test.js"], "javascript")
        assert "jest" in cmd

    def test_build_test_go(self):
        cmd = _build_test_command(["app_test.go"], "go")
        assert "go" in cmd and "test" in cmd

    def test_build_test_rust(self):
        cmd = _build_test_command(["test_lib.rs"], "rust")
        assert "cargo" in cmd and "test" in cmd


class TestDetectLanguage:
    """orchestrator의 _detect_language 테스트."""

    def test_detect_python(self):
        from youngs75_a2a.agents.orchestrator.agent import _detect_language
        assert _detect_language(["app.py", "models.py"]) == "python"

    def test_detect_go(self):
        from youngs75_a2a.agents.orchestrator.agent import _detect_language
        assert _detect_language(["main.go", "handler.go"]) == "go"

    def test_detect_mixed_majority(self):
        from youngs75_a2a.agents.orchestrator.agent import _detect_language
        assert _detect_language(["app.py", "index.js", "utils.py"]) == "python"

    def test_detect_empty_defaults_python(self):
        from youngs75_a2a.agents.orchestrator.agent import _detect_language
        assert _detect_language([]) == "python"

    def test_detect_rust(self):
        from youngs75_a2a.agents.orchestrator.agent import _detect_language
        assert _detect_language(["main.rs", "lib.rs"]) == "rust"
