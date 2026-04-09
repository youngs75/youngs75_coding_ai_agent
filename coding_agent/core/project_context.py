"""프로젝트 컨텍스트 동적 주입.

Claude Code의 CLAUDE.md 패턴을 구현:
- 프로젝트 루트에서 시작하여 상위 디렉토리까지 컨텍스트 파일 검색
- 발견된 모든 컨텍스트 파일을 합쳐서 시스템 프롬프트에 주입
- 토큰 예산 내에서만 포함 (max_context_tokens 설정)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 컨텍스트 파일 검색 경로 (우선순위 순)
CONTEXT_FILE_NAMES = [
    ".agent/context.md",  # 프로젝트 전용
    "AGENTS.md",  # 기존 호환
    ".ai_agent.md",  # 대체 이름
]

# 상위 디렉토리 검색 최대 깊이
_MAX_PARENT_DEPTH = 3

# 토큰 추정: 평균 4글자(한국어 포함) ≈ 1토큰
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """텍스트의 토큰 수를 간단히 추정한다.

    정확한 토크나이저 없이도 합리적인 예산 관리를 위해
    평균 4글자 = 1토큰으로 추정한다.
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)


class ProjectContextLoader:
    """프로젝트 컨텍스트 파일을 검색하고 시스템 프롬프트에 주입한다.

    Claude Code의 CLAUDE.md 패턴:
    - 프로젝트 루트에서 시작하여 상위 디렉토리까지 검색 (최대 3단계)
    - 발견된 모든 컨텍스트 파일을 합쳐서 시스템 프롬프트에 주입
    - 토큰 예산 내에서만 포함 (max_context_tokens 설정)
    """

    def __init__(self, workspace: str, max_context_tokens: int = 4000):
        self._workspace = Path(workspace).resolve()
        self._max_context_tokens = max_context_tokens

    @property
    def workspace(self) -> Path:
        """워크스페이스 경로를 반환한다."""
        return self._workspace

    @property
    def max_context_tokens(self) -> int:
        """최대 컨텍스트 토큰 수를 반환한다."""
        return self._max_context_tokens

    def discover(self) -> list[Path]:
        """컨텍스트 파일을 검색한다.

        프로젝트 루트에서 시작하여 상위 디렉토리까지 검색한다.
        각 디렉토리에서 CONTEXT_FILE_NAMES를 순서대로 확인한다.
        발견된 파일은 프로젝트 루트 → 상위 순서로 반환한다.

        Returns:
            발견된 컨텍스트 파일 경로 리스트 (프로젝트 루트 우선)
        """
        found: list[Path] = []
        current = self._workspace

        for depth in range(_MAX_PARENT_DEPTH + 1):
            for name in CONTEXT_FILE_NAMES:
                candidate = current / name
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if resolved not in found:
                        found.append(resolved)
                        logger.debug(
                            "컨텍스트 파일 발견 (depth=%d): %s", depth, resolved
                        )

            parent = current.parent
            # 루트 디렉토리에 도달하면 중단
            if parent == current:
                break
            current = parent

        return found

    def load(self) -> str:
        """발견된 컨텍스트 파일을 합쳐서 반환한다.

        토큰 예산 초과 시 truncate한다.
        파일이 없으면 빈 문자열을 반환한다.

        Returns:
            합쳐진 컨텍스트 문자열
        """
        files = self.discover()
        if not files:
            return ""

        parts: list[str] = []
        total_tokens = 0

        for filepath in files:
            try:
                content = filepath.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("컨텍스트 파일 읽기 실패: %s — %s", filepath, e)
                continue

            content_tokens = _estimate_tokens(content)

            if total_tokens + content_tokens > self._max_context_tokens:
                # 남은 예산만큼만 포함
                remaining_tokens = self._max_context_tokens - total_tokens
                if remaining_tokens <= 0:
                    break
                # 남은 토큰에 해당하는 글자 수만큼 자르기
                max_chars = remaining_tokens * _CHARS_PER_TOKEN
                content = content[:max_chars] + "\n... (토큰 예산 초과로 생략)"
                parts.append(content)
                break

            parts.append(content)
            total_tokens += content_tokens

        return "\n\n".join(parts)

    def build_system_prompt_section(self) -> str:
        """시스템 프롬프트에 삽입할 섹션을 생성한다.

        컨텍스트 파일이 없으면 빈 문자열을 반환한다.

        Returns:
            시스템 프롬프트에 삽입할 섹션 문자열
        """
        content = self.load()
        if not content:
            return ""

        return (
            "\n\n# 프로젝트 컨텍스트\n"
            "아래는 현재 프로젝트의 규칙과 컨텍스트입니다. "
            "이 내용을 따라 작업하세요.\n\n"
            "---\n"
            f"{content}\n"
            "---"
        )
