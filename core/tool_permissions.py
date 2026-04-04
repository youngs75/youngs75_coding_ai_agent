"""도구 실행 권한 관리.

Claude Code의 계층적 권한 + Codex의 정책 엔진을 Python으로 간결하게 구현.

계층적 권한 설정:
1. 기본 규칙 (DEFAULT_PERMISSIONS)
2. 프로젝트 설정 (.agent/permissions.yaml)
3. 환경변수 오버라이드
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PermissionDecision(str, Enum):
    """도구 실행 권한 판정 결과."""

    ALLOW = "allow"  # 자동 허용
    ASK = "ask"  # 사용자 확인 필요
    DENY = "deny"  # 거부


# 민감 파일 패턴 (수정 시 사용자 확인 필요)
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\.env($|\.)"),  # .env, .env.local 등
    re.compile(r"credentials", re.IGNORECASE),
    re.compile(r"secrets?\.ya?ml", re.IGNORECASE),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
    re.compile(r"id_rsa"),
    re.compile(r"\.ssh/"),
    re.compile(r"token\.json", re.IGNORECASE),
]


def _is_sensitive_path(path: str) -> bool:
    """경로가 민감 파일 패턴에 해당하는지 확인한다."""
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.search(path):
            return True
    return False


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """YAML 설정 파일을 로드한다.

    PyYAML이 없으면 빈 딕셔너리를 반환한다.
    """
    if not path.is_file():
        return {}
    try:
        import yaml

        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        return data if isinstance(data, dict) else {}
    except ImportError:
        logger.debug("PyYAML 미설치 — 프로젝트 권한 설정 파일 무시: %s", path)
        return {}
    except Exception as e:
        logger.warning("권한 설정 파일 로드 실패: %s — %s", path, e)
        return {}


class ToolPermissionManager:
    """도구 실행 권한 관리자.

    계층적 권한 설정:
    1. 기본 규칙 (DEFAULT_PERMISSIONS)
    2. 프로젝트 설정 (.agent/permissions.yaml)
    3. 환경변수 오버라이드 (TOOL_PERM_{TOOL_NAME}=allow|ask|deny)
    """

    DEFAULT_PERMISSIONS: dict[str, PermissionDecision] = {
        # 읽기 도구 — 항상 허용
        "read_file": PermissionDecision.ALLOW,
        "search_code": PermissionDecision.ALLOW,
        "list_directory": PermissionDecision.ALLOW,
        # 쓰기 도구 — 허용 (workspace 내)
        "write_file": PermissionDecision.ALLOW,
        "str_replace": PermissionDecision.ALLOW,
        "apply_patch": PermissionDecision.ALLOW,
        # 실행 도구 — 사용자 확인
        "execute_python": PermissionDecision.ASK,
        "bash": PermissionDecision.ASK,
        # 삭제 — 사용자 확인
        "delete_file": PermissionDecision.ASK,
    }

    def __init__(
        self,
        workspace: str,
        project_config: Path | None = None,
    ):
        self._workspace = Path(workspace).resolve()
        self._denials: list[dict[str, Any]] = []

        # 계층 1: 기본 규칙 복사
        self._permissions: dict[str, PermissionDecision] = dict(
            self.DEFAULT_PERMISSIONS
        )

        # 계층 2: 프로젝트 설정 (.agent/permissions.yaml)
        config_path = project_config or (
            self._workspace / ".agent" / "permissions.yaml"
        )
        self._apply_project_config(config_path)

        # 계층 3: 환경변수 오버라이드
        self._apply_env_overrides()

    @property
    def workspace(self) -> Path:
        """워크스페이스 경로를 반환한다."""
        return self._workspace

    def _apply_project_config(self, config_path: Path) -> None:
        """프로젝트 설정 파일에서 권한을 적용한다."""
        config = _load_yaml_config(config_path)
        permissions = config.get("permissions", {})
        if not isinstance(permissions, dict):
            return

        for tool_name, decision_str in permissions.items():
            try:
                decision = PermissionDecision(str(decision_str).lower())
                self._permissions[str(tool_name)] = decision
                logger.debug(
                    "프로젝트 설정 권한 적용: %s → %s", tool_name, decision.value
                )
            except ValueError:
                logger.warning("잘못된 권한 값 무시: %s=%s", tool_name, decision_str)

    def _apply_env_overrides(self) -> None:
        """환경변수에서 권한 오버라이드를 적용한다.

        형식: TOOL_PERM_{TOOL_NAME}=allow|ask|deny
        예시: TOOL_PERM_BASH=allow
        """
        prefix = "TOOL_PERM_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            tool_name = key[len(prefix) :].lower()
            try:
                decision = PermissionDecision(value.lower())
                self._permissions[tool_name] = decision
                logger.debug(
                    "환경변수 권한 오버라이드: %s → %s", tool_name, decision.value
                )
            except ValueError:
                logger.warning("잘못된 환경변수 권한 값 무시: %s=%s", key, value)

    def _is_within_workspace(self, path: str) -> bool:
        """경로가 workspace 내에 있는지 확인한다."""
        try:
            resolved = Path(self._workspace, path).resolve()
            return str(resolved).startswith(str(self._workspace))
        except (OSError, ValueError):
            return False

    def _extract_path_from_input(self, tool_input: dict[str, Any]) -> str | None:
        """도구 입력에서 파일 경로를 추출한다."""
        # 일반적인 경로 파라미터명들
        for key in ("path", "file_path", "filepath", "filename", "file", "target"):
            if key in tool_input and isinstance(tool_input[key], str):
                return tool_input[key]
        return None

    def check(
        self, tool_name: str, tool_input: dict[str, Any] | None = None
    ) -> PermissionDecision:
        """도구 사용 권한을 판정한다.

        추가 검사:
        - workspace 밖 경로 접근 → DENY
        - 위험 패턴 (.env, credentials 등 민감 파일 수정) → ASK

        Args:
            tool_name: 도구 이름
            tool_input: 도구 입력 파라미터 딕셔너리

        Returns:
            PermissionDecision (ALLOW, ASK, DENY)
        """
        if tool_input is None:
            tool_input = {}

        # 경로 기반 검사
        path = self._extract_path_from_input(tool_input)
        if path is not None:
            # workspace 밖 경로 → DENY
            if not self._is_within_workspace(path):
                reason = f"workspace 밖 경로 접근 시도: {path}"
                logger.warning("권한 거부: %s — %s", tool_name, reason)
                self.record_denial(tool_name, reason)
                return PermissionDecision.DENY

            # 쓰기/삭제 도구 + 민감 파일 → ASK
            write_tools = {
                "write_file",
                "str_replace",
                "apply_patch",
                "delete_file",
            }
            if tool_name in write_tools and _is_sensitive_path(path):
                logger.info(
                    "민감 파일 접근 — 사용자 확인 필요: %s → %s",
                    tool_name,
                    path,
                )
                return PermissionDecision.ASK

        # 기본 규칙 조회
        decision = self._permissions.get(tool_name, PermissionDecision.ASK)
        return decision

    def record_denial(self, tool_name: str, reason: str) -> None:
        """거부 기록을 저장한다.

        Args:
            tool_name: 거부된 도구 이름
            reason: 거부 사유
        """
        entry = {
            "tool_name": tool_name,
            "reason": reason,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self._denials.append(entry)
        logger.info("도구 거부 기록: %s — %s", tool_name, reason)

    @property
    def denial_log(self) -> list[dict[str, Any]]:
        """거부 기록을 반환한다.

        Returns:
            거부 기록 리스트 (각 항목: tool_name, reason, timestamp)
        """
        return list(self._denials)
