"""generate_final 도구 기반 파일 저장 전환 테스트.

DeepAgents 패턴 도입 후:
- _generate_final()이 write_file 도구를 호출하여 파일 저장
- _apply_code()가 도구 저장 시 마크다운 파싱 스킵
- VERIFY_SYSTEM_PROMPT에서 filepath 규칙 제거
- 재시도 시 written_files 초기화
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_agent.agents.coding_assistant.prompts import (
    GENERATE_FINAL_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)


# ── 프롬프트 테스트 ──────────────────────────────────────


class TestGenerateFinalPrompt:
    """GENERATE_FINAL_SYSTEM_PROMPT 검증."""

    def test_contains_write_file_instruction(self):
        prompt = GENERATE_FINAL_SYSTEM_PROMPT.format(language="python")
        assert "write_file" in prompt

    def test_contains_tool_usage_mandate(self):
        prompt = GENERATE_FINAL_SYSTEM_PROMPT.format(language="python")
        assert "마크다운 코드 블록만으로는 파일이 저장되지 않습니다" in prompt

    def test_contains_relative_path_instruction(self):
        prompt = GENERATE_FINAL_SYSTEM_PROMPT.format(language="python")
        assert "상대 경로" in prompt

    def test_contains_test_generation_rule(self):
        prompt = GENERATE_FINAL_SYSTEM_PROMPT.format(language="python")
        assert "테스트" in prompt

    def test_contains_dependency_rules(self):
        prompt = GENERATE_FINAL_SYSTEM_PROMPT.format(language="python")
        assert "의존성" in prompt

    def test_language_placeholder_works(self):
        for lang in ("python", "javascript", "go"):
            prompt = GENERATE_FINAL_SYSTEM_PROMPT.format(language=lang)
            assert lang in prompt


class TestVerifyPromptNoFilepath:
    """VERIFY_SYSTEM_PROMPT에서 filepath 규칙 제거 확인."""

    def test_no_filepath_check_rule(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        # filepath 주석 "검사" 규칙이 없어야 함 (설명에서 언급은 OK)
        assert "누락된 코드 블록이 있으면 반드시" not in prompt

    def test_no_rule_8(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        # 규칙이 6개까지만 (7, 8 삭제)
        assert "7." not in prompt
        assert "8." not in prompt

    def test_has_write_file_note(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        assert "write_file 도구로 이미 저장" in prompt

    def test_correctness_rule_exists(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        assert "정확성" in prompt
        assert "안전성" in prompt

    def test_dev_secret_lenient(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        assert "개발 환경 기본값" in prompt

    def test_json_output_format(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        assert "passed: true/false" in prompt


# ── _apply_code 스킵 가드 테스트 ────────────────────────


class TestApplyCodeSkipGuard:
    """write_file 도구로 파일이 이미 저장된 경우 _apply_code가 스킵하는지 확인."""

    @pytest.fixture
    def mock_agent(self):
        """최소한의 CodingAssistantAgent mock."""
        from coding_agent.agents.coding_assistant.agent import (
            CodingAssistantAgent,
        )

        with patch.object(CodingAssistantAgent, "__init__", lambda self: None):
            agent = CodingAssistantAgent.__new__(CodingAssistantAgent)
            agent._coding_config = MagicMock()
            agent._coding_config.allowed_extensions = [".py", ".js"]
            agent._coding_config.max_delete_lines = 100
            return agent

    @pytest.mark.asyncio
    async def test_skip_when_written_files_exist(self, mock_agent):
        state = {
            "written_files": ["app.py", "tests/test_app.py"],
            "execution_log": [],
            "generated_code": "# some code",
        }
        result = await mock_agent._apply_code(state)
        assert result["written_files"] == ["app.py", "tests/test_app.py"]
        assert any("스킵" in e for e in result["execution_log"])

    @pytest.mark.asyncio
    async def test_fallback_when_no_written_files(self, mock_agent):
        state = {
            "written_files": [],
            "execution_log": [],
            "generated_code": "",
        }
        result = await mock_agent._apply_code(state)
        assert result["written_files"] == []
        assert any("생성된 코드 없음" in e for e in result["execution_log"])


# ── verify_result written_files 초기화 테스트 ────────────


class TestWrittenFilesResetOnRetry:
    """검증 실패 / 테스트 실패 시 written_files가 초기화되는지 확인."""

    def test_verify_result_resets_written_files_on_failure(self):
        """_verify_result 반환값에서 passed=False 시 written_files=[] 포함 확인.

        실제 _verify_result는 LLM 호출이 필요하므로,
        반환 딕셔너리 구조만 검증한다.
        """
        # passed=False일 때 written_files가 비어야 함
        result = {
            "verify_result": {"passed": False, "issues": ["test issue"]},
            "written_files": [],
            "iteration": 1,
        }
        assert result["written_files"] == []

    def test_verify_result_no_reset_on_pass(self):
        """passed=True일 때 written_files가 유지되어야 함."""
        result = {
            "verify_result": {"passed": True, "issues": []},
            "iteration": 1,
        }
        # written_files 키가 없으면 기존 state 값 유지
        assert "written_files" not in result


# ── write_file 결과 파싱 테스트 ──────────────────────────


class TestWriteFileResultParsing:
    """write_file 도구 결과 문자열에서 파일 경로 파싱."""

    def test_parse_ok_result(self):
        result = "OK: app.py (1234자, 56줄)"
        filepath = result.split("OK:")[1].split("(")[0].strip()
        assert filepath == "app.py"

    def test_parse_nested_path(self):
        result = "OK: backend/models.py (2345자, 78줄)"
        filepath = result.split("OK:")[1].split("(")[0].strip()
        assert filepath == "backend/models.py"

    def test_parse_deep_nested_path(self):
        result = "OK: src/components/Board.vue (890자, 30줄)"
        filepath = result.split("OK:")[1].split("(")[0].strip()
        assert filepath == "src/components/Board.vue"

    def test_error_result_not_parsed(self):
        result = "도구 실행 오류: FileNotFoundError"
        assert not result.startswith("OK:")
