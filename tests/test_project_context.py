"""프로젝트 컨텍스트 주입 및 도구 권한 모델 테스트.

테스트 항목:
- ProjectContextLoader: 컨텍스트 파일 발견, 로드, 토큰 예산, 시스템 프롬프트 섹션
- ToolPermissionManager: 기본 규칙, workspace 경로 제한, 민감 파일, 프로젝트 설정 오버라이드
"""

from __future__ import annotations

from pathlib import Path

import pytest

from youngs75_a2a.core.project_context import (
    ProjectContextLoader,
    _estimate_tokens,
)
from youngs75_a2a.core.tool_permissions import (
    PermissionDecision,
    ToolPermissionManager,
    _is_sensitive_path,
)


# ── ProjectContextLoader 테스트 ──


class TestProjectContextLoaderDiscover:
    """컨텍스트 파일 발견 테스트."""

    def test_discover_in_project_root(self, tmp_path: Path) -> None:
        """프로젝트 루트에 AGENTS.md가 있으면 발견한다."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# Project Rules\nUse Python 3.13", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        found = loader.discover()

        assert len(found) == 1
        assert found[0] == agents_md.resolve()

    def test_discover_agent_context_md(self, tmp_path: Path) -> None:
        """프로젝트 전용 .agent/context.md를 발견한다."""
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        ctx_file = agent_dir / "context.md"
        ctx_file.write_text("# Context\nCustom rules", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        found = loader.discover()

        assert len(found) == 1
        assert found[0] == ctx_file.resolve()

    def test_discover_priority_order(self, tmp_path: Path) -> None:
        """여러 컨텍스트 파일이 있으면 우선순위 순으로 반환한다."""
        # .agent/context.md (최우선)
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        ctx1 = agent_dir / "context.md"
        ctx1.write_text("context1", encoding="utf-8")

        # AGENTS.md (차순위)
        ctx2 = tmp_path / "AGENTS.md"
        ctx2.write_text("context2", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        found = loader.discover()

        assert len(found) == 2
        assert found[0] == ctx1.resolve()
        assert found[1] == ctx2.resolve()

    def test_discover_in_parent_directory(self, tmp_path: Path) -> None:
        """상위 디렉토리의 AGENTS.md를 발견한다."""
        # 상위에 컨텍스트 파일 생성
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# Root context", encoding="utf-8")

        # 하위 디렉토리를 workspace로 사용
        child_dir = tmp_path / "subproject"
        child_dir.mkdir()

        loader = ProjectContextLoader(str(child_dir))
        found = loader.discover()

        assert len(found) == 1
        assert found[0] == agents_md.resolve()

    def test_discover_no_context_files(self, tmp_path: Path) -> None:
        """컨텍스트 파일이 없으면 빈 리스트를 반환한다."""
        loader = ProjectContextLoader(str(tmp_path))
        found = loader.discover()

        assert found == []

    def test_discover_no_duplicate(self, tmp_path: Path) -> None:
        """같은 파일이 중복 발견되지 않는다."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("content", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        found = loader.discover()

        # 같은 파일이 여러 번 반환되면 안 됨
        assert len(found) == len(set(found))


class TestProjectContextLoaderLoad:
    """컨텍스트 파일 로드 테스트."""

    def test_load_single_file(self, tmp_path: Path) -> None:
        """단일 컨텍스트 파일의 내용을 반환한다."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# Rules\nFollow PEP 8", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        content = loader.load()

        assert "Rules" in content
        assert "Follow PEP 8" in content

    def test_load_no_files_returns_empty(self, tmp_path: Path) -> None:
        """컨텍스트 파일이 없으면 빈 문자열을 반환한다."""
        loader = ProjectContextLoader(str(tmp_path))
        content = loader.load()

        assert content == ""

    def test_load_truncate_on_token_budget(self, tmp_path: Path) -> None:
        """토큰 예산 초과 시 truncate한다."""
        agents_md = tmp_path / "AGENTS.md"
        # 큰 내용 생성 (약 2000 토큰 = 8000자)
        large_content = "A" * 10000
        agents_md.write_text(large_content, encoding="utf-8")

        # 매우 작은 토큰 예산 설정
        loader = ProjectContextLoader(str(tmp_path), max_context_tokens=100)
        content = loader.load()

        # truncate 되어야 함
        assert len(content) < len(large_content)
        assert "토큰 예산 초과로 생략" in content

    def test_load_multiple_files_merged(self, tmp_path: Path) -> None:
        """여러 컨텍스트 파일이 합쳐져 반환된다."""
        agent_dir = tmp_path / ".agent"
        agent_dir.mkdir()
        (agent_dir / "context.md").write_text("Context A", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("Context B", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        content = loader.load()

        assert "Context A" in content
        assert "Context B" in content


class TestProjectContextLoaderBuildSection:
    """시스템 프롬프트 섹션 생성 테스트."""

    def test_build_system_prompt_section(self, tmp_path: Path) -> None:
        """시스템 프롬프트 섹션이 올바른 형식으로 생성된다."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# My Project Rules", encoding="utf-8")

        loader = ProjectContextLoader(str(tmp_path))
        section = loader.build_system_prompt_section()

        assert "# 프로젝트 컨텍스트" in section
        assert "아래는 현재 프로젝트의 규칙과 컨텍스트입니다" in section
        assert "---" in section
        assert "# My Project Rules" in section

    def test_build_system_prompt_section_empty(self, tmp_path: Path) -> None:
        """컨텍스트 파일이 없으면 빈 문자열을 반환한다."""
        loader = ProjectContextLoader(str(tmp_path))
        section = loader.build_system_prompt_section()

        assert section == ""


class TestEstimateTokens:
    """토큰 추정 테스트."""

    def test_estimate_tokens_basic(self) -> None:
        """기본 토큰 추정이 동작한다."""
        # 4글자 = 1토큰
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("abcdefgh") == 2

    def test_estimate_tokens_minimum(self) -> None:
        """빈 문자열도 최소 1토큰."""
        assert _estimate_tokens("") == 1
        assert _estimate_tokens("a") == 1


# ── ToolPermissionManager 테스트 ──


class TestToolPermissionManagerDefaults:
    """기본 권한 규칙 테스트."""

    def test_read_tools_allow(self, tmp_path: Path) -> None:
        """읽기 도구는 항상 ALLOW."""
        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.check("read_file") == PermissionDecision.ALLOW
        assert mgr.check("search_code") == PermissionDecision.ALLOW
        assert mgr.check("list_directory") == PermissionDecision.ALLOW

    def test_write_tools_allow(self, tmp_path: Path) -> None:
        """쓰기 도구는 workspace 내에서 ALLOW."""
        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.check("write_file") == PermissionDecision.ALLOW
        assert mgr.check("str_replace") == PermissionDecision.ALLOW

    def test_execute_tools_ask(self, tmp_path: Path) -> None:
        """실행 도구(bash)는 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.check("bash") == PermissionDecision.ASK
        assert mgr.check("execute_python") == PermissionDecision.ASK

    def test_delete_tool_ask(self, tmp_path: Path) -> None:
        """삭제 도구는 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.check("delete_file") == PermissionDecision.ASK

    def test_unknown_tool_ask(self, tmp_path: Path) -> None:
        """등록되지 않은 도구는 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.check("unknown_tool") == PermissionDecision.ASK


class TestToolPermissionManagerWorkspace:
    """Workspace 경로 제한 테스트."""

    def test_workspace_outside_deny(self, tmp_path: Path) -> None:
        """workspace 밖 경로 접근 시 DENY."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("write_file", {"path": "/etc/passwd"})
        assert result == PermissionDecision.DENY

    def test_workspace_inside_allow(self, tmp_path: Path) -> None:
        """workspace 안 경로 접근 시 정상 판정."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("write_file", {"path": "src/main.py"})
        assert result == PermissionDecision.ALLOW

    def test_workspace_traversal_deny(self, tmp_path: Path) -> None:
        """.. 경로 트래버설로 workspace 밖 접근 시 DENY."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("write_file", {"path": "../../etc/passwd"})
        assert result == PermissionDecision.DENY


class TestToolPermissionManagerSensitive:
    """민감 파일 패턴 테스트."""

    def test_env_file_ask(self, tmp_path: Path) -> None:
        """.env 파일 쓰기 시 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("write_file", {"path": ".env"})
        assert result == PermissionDecision.ASK

    def test_env_local_file_ask(self, tmp_path: Path) -> None:
        """.env.local 파일 쓰기 시 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("write_file", {"path": ".env.local"})
        assert result == PermissionDecision.ASK

    def test_credentials_file_ask(self, tmp_path: Path) -> None:
        """credentials 파일 쓰기 시 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("write_file", {"path": "config/credentials.json"})
        assert result == PermissionDecision.ASK

    def test_pem_file_ask(self, tmp_path: Path) -> None:
        """.pem 파일 삭제 시 ASK."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("delete_file", {"path": "certs/server.pem"})
        assert result == PermissionDecision.ASK

    def test_normal_file_not_sensitive(self, tmp_path: Path) -> None:
        """일반 파일은 민감 파일로 판정되지 않는다."""
        assert not _is_sensitive_path("src/main.py")
        assert not _is_sensitive_path("README.md")
        assert not _is_sensitive_path("tests/test_app.py")

    def test_read_sensitive_still_allow(self, tmp_path: Path) -> None:
        """읽기 도구로 민감 파일 접근은 ALLOW (쓰기만 ASK)."""
        mgr = ToolPermissionManager(str(tmp_path))

        result = mgr.check("read_file", {"path": ".env"})
        assert result == PermissionDecision.ALLOW


class TestToolPermissionManagerProjectConfig:
    """프로젝트 설정 파일 오버라이드 테스트."""

    def test_project_config_override(self, tmp_path: Path) -> None:
        """프로젝트 설정 파일로 권한을 오버라이드한다."""
        config_path = tmp_path / "permissions.yaml"
        config_path.write_text(
            "permissions:\n  bash: allow\n  delete_file: deny\n",
            encoding="utf-8",
        )

        mgr = ToolPermissionManager(str(tmp_path), project_config=config_path)

        assert mgr.check("bash") == PermissionDecision.ALLOW
        assert mgr.check("delete_file") == PermissionDecision.DENY

    def test_project_config_missing_no_error(self, tmp_path: Path) -> None:
        """프로젝트 설정 파일이 없어도 에러 없이 기본값 사용."""
        missing = tmp_path / "nonexistent.yaml"
        mgr = ToolPermissionManager(str(tmp_path), project_config=missing)

        # 기본값이 적용되어야 함
        assert mgr.check("read_file") == PermissionDecision.ALLOW
        assert mgr.check("bash") == PermissionDecision.ASK


class TestToolPermissionManagerEnvOverride:
    """환경변수 오버라이드 테스트."""

    def test_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """환경변수로 권한을 오버라이드한다."""
        monkeypatch.setenv("TOOL_PERM_BASH", "allow")
        monkeypatch.setenv("TOOL_PERM_READ_FILE", "deny")

        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.check("bash") == PermissionDecision.ALLOW
        assert mgr.check("read_file") == PermissionDecision.DENY

    def test_env_override_invalid_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """잘못된 환경변수 값은 무시된다."""
        monkeypatch.setenv("TOOL_PERM_BASH", "invalid_value")

        mgr = ToolPermissionManager(str(tmp_path))

        # 기본값이 유지되어야 함
        assert mgr.check("bash") == PermissionDecision.ASK


class TestToolPermissionManagerDenialLog:
    """거부 기록 테스트."""

    def test_record_denial(self, tmp_path: Path) -> None:
        """거부 기록이 저장된다."""
        mgr = ToolPermissionManager(str(tmp_path))

        mgr.record_denial("bash", "위험한 명령어")

        log = mgr.denial_log
        assert len(log) == 1
        assert log[0]["tool_name"] == "bash"
        assert log[0]["reason"] == "위험한 명령어"
        assert "timestamp" in log[0]

    def test_denial_log_empty_initially(self, tmp_path: Path) -> None:
        """초기 거부 기록은 비어있다."""
        mgr = ToolPermissionManager(str(tmp_path))

        assert mgr.denial_log == []

    def test_workspace_deny_auto_records(self, tmp_path: Path) -> None:
        """workspace 밖 접근 시 자동으로 거부 기록이 저장된다."""
        mgr = ToolPermissionManager(str(tmp_path))

        mgr.check("write_file", {"path": "/etc/shadow"})

        log = mgr.denial_log
        assert len(log) == 1
        assert "workspace" in log[0]["reason"]

    def test_multiple_denials_recorded(self, tmp_path: Path) -> None:
        """여러 거부 기록이 순서대로 저장된다."""
        mgr = ToolPermissionManager(str(tmp_path))

        mgr.record_denial("tool_a", "사유 1")
        mgr.record_denial("tool_b", "사유 2")

        log = mgr.denial_log
        assert len(log) == 2
        assert log[0]["tool_name"] == "tool_a"
        assert log[1]["tool_name"] == "tool_b"

    def test_denial_log_is_copy(self, tmp_path: Path) -> None:
        """denial_log는 내부 리스트의 복사본이다 (외부 수정 방지)."""
        mgr = ToolPermissionManager(str(tmp_path))

        mgr.record_denial("test", "사유")
        log = mgr.denial_log
        log.clear()

        # 내부 리스트는 영향 없어야 함
        assert len(mgr.denial_log) == 1


# ── 통합 테스트 ──


class TestIntegration:
    """프로젝트 컨텍스트와 도구 권한의 통합 시나리오."""

    def test_context_loader_with_prompt_injection(self, tmp_path: Path) -> None:
        """컨텍스트 로더가 프롬프트에 올바르게 주입된다."""
        from youngs75_a2a.agents.coding_assistant.prompts import (
            PARSE_SYSTEM_PROMPT,
            inject_project_context,
        )

        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# Custom Rules\nAlways test first", encoding="utf-8")

        result = inject_project_context(PARSE_SYSTEM_PROMPT, workspace=str(tmp_path))

        assert PARSE_SYSTEM_PROMPT in result
        assert "Custom Rules" in result
        assert "Always test first" in result

    def test_context_injection_no_workspace(self) -> None:
        """workspace가 None이면 원본 프롬프트를 그대로 반환한다."""
        from youngs75_a2a.agents.coding_assistant.prompts import (
            PARSE_SYSTEM_PROMPT,
            inject_project_context,
        )

        result = inject_project_context(PARSE_SYSTEM_PROMPT, workspace=None)
        assert result == PARSE_SYSTEM_PROMPT

    def test_permission_manager_with_context_loader(self, tmp_path: Path) -> None:
        """권한 관리자와 컨텍스트 로더가 같은 workspace를 공유한다."""
        loader = ProjectContextLoader(str(tmp_path))
        mgr = ToolPermissionManager(str(tmp_path))

        assert loader.workspace == mgr.workspace
