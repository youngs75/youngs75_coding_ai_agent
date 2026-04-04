"""Safety Envelope — 에이전트 출력을 실행 전에 검증.

AutoHarness 논문 교훈: 검증기는 에이전트 바깥에 있어야 한다.
"심판이 선수를 겸하면 안 된다."

프레임워크 차원에서 강제하므로 개별 에이전트 코드 수정 없이 적용 가능.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationStatus(Enum):
    """검증 결과 상태."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class ValidationResult:
    """개별 규칙 검증 결과."""

    rule_name: str
    status: ValidationStatus
    message: str


@dataclass
class ValidationReport:
    """전체 검증 리포트."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return all(r.status != ValidationStatus.BLOCK for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.status == ValidationStatus.WARN for r in self.results)

    @property
    def blocked_rules(self) -> list[str]:
        return [r.rule_name for r in self.results if r.status == ValidationStatus.BLOCK]

    def summary(self) -> str:
        if self.is_safe and not self.has_warnings:
            return "모든 검증 통과"
        parts = []
        for r in self.results:
            if r.status != ValidationStatus.PASS:
                parts.append(f"[{r.status.value}] {r.rule_name}: {r.message}")
        return "\n".join(parts)


# ── 검증 규칙 ──────────────────────────────────────────

# 시크릿 패턴
_SECRET_PATTERNS = [
    re.compile(
        r"""(?i)(api[_-]?key|secret|password|token|credential)\s*[=:]\s*['"][^'"]{8,}['"]"""
    ),
    re.compile(r"""(?i)sk-[a-zA-Z0-9]{20,}"""),
]

# 위험 명령 패턴
_DANGEROUS_COMMANDS = [
    re.compile(r"""rm\s+-rf\s+/"""),
    re.compile(r"""(?i)drop\s+(table|database)"""),
    re.compile(r"""(?i)truncate\s+table"""),
    re.compile(r""">\s*/dev/sd"""),
    re.compile(r"""(?i)format\s+[a-z]:"""),
]


class ActionValidator:
    """에이전트 출력을 실행 전에 검증하는 Safety Envelope."""

    def __init__(
        self,
        *,
        allowed_extensions: list[str] | None = None,
        max_delete_lines: int = 100,
        allowed_directories: list[str] | None = None,
    ) -> None:
        self.allowed_extensions = allowed_extensions or [
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".toml",
        ]
        self.max_delete_lines = max_delete_lines
        self.allowed_directories = allowed_directories

    def validate(self, code: str, **context: Any) -> ValidationReport:
        """코드에 대해 모든 규칙을 검사한다."""
        report = ValidationReport()
        report.results.append(self._check_secrets(code))
        report.results.append(self._check_dangerous_commands(code))
        report.results.append(self._check_delete_volume(code))

        if target_files := context.get("target_files"):
            report.results.append(self._check_file_extensions(target_files))
            if self.allowed_directories:
                report.results.append(self._check_directory_scope(target_files))

        return report

    def _check_secrets(self, code: str) -> ValidationResult:
        """시크릿 노출 방지."""
        for pattern in _SECRET_PATTERNS:
            if pattern.search(code):
                return ValidationResult(
                    rule_name="secret_exposure",
                    status=ValidationStatus.BLOCK,
                    message="코드에 시크릿(API 키, 패스워드 등)이 하드코딩되어 있습니다",
                )
        return ValidationResult(
            rule_name="secret_exposure",
            status=ValidationStatus.PASS,
            message="시크릿 미발견",
        )

    def _check_dangerous_commands(self, code: str) -> ValidationResult:
        """위험 명령 차단."""
        for pattern in _DANGEROUS_COMMANDS:
            if pattern.search(code):
                return ValidationResult(
                    rule_name="dangerous_command",
                    status=ValidationStatus.BLOCK,
                    message=f"위험 명령 감지: {pattern.pattern}",
                )
        return ValidationResult(
            rule_name="dangerous_command",
            status=ValidationStatus.PASS,
            message="위험 명령 미발견",
        )

    def _check_delete_volume(self, code: str) -> ValidationResult:
        """대량 삭제 감지."""
        delete_indicators = (
            code.count("- ") + code.count("deleted") + code.count("remove")
        )
        if delete_indicators > self.max_delete_lines:
            return ValidationResult(
                rule_name="delete_volume",
                status=ValidationStatus.WARN,
                message=f"대량 삭제 감지 (지표: {delete_indicators}줄 이상)",
            )
        return ValidationResult(
            rule_name="delete_volume",
            status=ValidationStatus.PASS,
            message="삭제 규모 정상",
        )

    def _check_file_extensions(self, target_files: list[str]) -> ValidationResult:
        """허용 파일 확장자 검사."""
        for f in target_files:
            ext = "." + f.rsplit(".", 1)[-1] if "." in f else ""
            if ext and ext not in self.allowed_extensions:
                return ValidationResult(
                    rule_name="file_extension",
                    status=ValidationStatus.BLOCK,
                    message=f"허용되지 않은 파일 확장자: {ext} ({f})",
                )
        return ValidationResult(
            rule_name="file_extension",
            status=ValidationStatus.PASS,
            message="파일 확장자 검증 통과",
        )

    def _check_directory_scope(self, target_files: list[str]) -> ValidationResult:
        """허용 디렉토리 범위 검사."""
        for f in target_files:
            if not any(f.startswith(d) for d in self.allowed_directories or []):
                return ValidationResult(
                    rule_name="directory_scope",
                    status=ValidationStatus.BLOCK,
                    message=f"허용 범위 밖 파일: {f}",
                )
        return ValidationResult(
            rule_name="directory_scope",
            status=ValidationStatus.PASS,
            message="디렉토리 범위 검증 통과",
        )
