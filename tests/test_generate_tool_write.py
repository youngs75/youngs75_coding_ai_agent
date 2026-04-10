"""generate_final 도구 기반 파일 저장 전환 테스트.

DeepAgents 패턴 도입 후:
- _generate_final()이 write_file 도구를 호출하여 파일 저장
- _generate_code()가 마크다운 폴백 수행
- VERIFY_SYSTEM_PROMPT에서 filepath 규칙 제거
- 재시도 시 written_files 초기화
"""

from __future__ import annotations

import re

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

    def test_no_rule_9(self):
        prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=100, allowed_extensions=".py, .js"
        )
        # 검증 영역이 8개까지만 (9 이상 없어야 함)
        assert "9." not in prompt

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
