"""Semantic Memory 자동 로더.

프로젝트 컨벤션/규칙을 파일에서 읽어 Semantic Memory에 로딩한다.

소스:
  - AGENTS.md: 프로젝트 규칙, 커밋 규칙, 기술 스택
  - pyproject.toml: 의존성, 프로젝트 메타데이터

사용 예:
    loader = SemanticMemoryLoader(workspace="/path/to/project", store=memory_store)
    count = loader.load_all()
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from coding_agent.core.memory.schemas import MemoryItem, MemoryType
from coding_agent.core.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Semantic Memory 태그 상수
_TAG_CONVENTION = "convention"
_TAG_TECH_STACK = "tech_stack"
_TAG_RULE = "rule"
_TAG_STRUCTURE = "structure"


class SemanticMemoryLoader:
    """프로젝트 컨벤션을 Semantic Memory로 자동 로딩한다."""

    def __init__(
        self,
        workspace: str | Path,
        store: MemoryStore,
    ) -> None:
        self._workspace = Path(workspace)
        self._store = store

    def load_all(self) -> int:
        """모든 소스에서 Semantic Memory를 로딩한다.

        Returns:
            로딩된 항목 수
        """
        # 중복 방지: 기존 semantic 메모리 초기화
        self._store.clear(MemoryType.SEMANTIC)

        count = 0
        count += self._load_from_agents_md()
        count += self._load_from_pyproject()
        logger.info("Semantic Memory 로딩 완료: %d건", count)
        return count

    def _load_from_agents_md(self) -> int:
        """AGENTS.md에서 프로젝트 규칙/컨벤션을 추출한다."""
        agents_md = self._workspace / "AGENTS.md"
        if not agents_md.exists():
            logger.debug("AGENTS.md 없음, 스킵")
            return 0

        text = agents_md.read_text(encoding="utf-8")
        items: list[MemoryItem] = []

        # 섹션별 파싱: ## 헤딩 기준으로 분할
        sections = re.split(r"\n## ", text)
        for section in sections:
            lines = section.strip().split("\n")
            if not lines:
                continue

            heading = lines[0].lstrip("# ").strip()
            body = "\n".join(lines[1:]).strip()

            if not body:
                continue

            # 핵심 섹션만 추출
            if heading in (
                "커뮤니케이션 규칙",
                "커밋 및 PR 규칙",
                "개발 및 검증 규칙",
            ):
                items.append(
                    MemoryItem(
                        type=MemoryType.SEMANTIC,
                        content=f"[{heading}] {body}",
                        tags=[_TAG_RULE, _TAG_CONVENTION],
                        metadata={"source": "AGENTS.md", "section": heading},
                    )
                )
            elif heading == "주요 기술 스택":
                items.append(
                    MemoryItem(
                        type=MemoryType.SEMANTIC,
                        content=f"[기술 스택] {body}",
                        tags=[_TAG_TECH_STACK],
                        metadata={"source": "AGENTS.md", "section": heading},
                    )
                )
            elif heading == "프로젝트 구조":
                items.append(
                    MemoryItem(
                        type=MemoryType.SEMANTIC,
                        content=f"[프로젝트 구조] {body}",
                        tags=[_TAG_STRUCTURE],
                        metadata={"source": "AGENTS.md", "section": heading},
                    )
                )

        for item in items:
            self._store.put(item)

        return len(items)

    def _load_from_pyproject(self) -> int:
        """pyproject.toml에서 프로젝트 메타데이터를 추출한다."""
        pyproject = self._workspace / "pyproject.toml"
        if not pyproject.exists():
            logger.debug("pyproject.toml 없음, 스킵")
            return 0

        text = pyproject.read_text(encoding="utf-8")
        items: list[MemoryItem] = []

        # 프로젝트명과 버전 추출
        name_match = re.search(r'^name\s*=\s*"(.+?)"', text, re.MULTILINE)
        version_match = re.search(r'^version\s*=\s*"(.+?)"', text, re.MULTILINE)
        python_match = re.search(r'^requires-python\s*=\s*"(.+?)"', text, re.MULTILINE)

        meta_parts = []
        if name_match:
            meta_parts.append(f"프로젝트: {name_match.group(1)}")
        if version_match:
            meta_parts.append(f"버전: {version_match.group(1)}")
        if python_match:
            meta_parts.append(f"Python: {python_match.group(1)}")

        if meta_parts:
            items.append(
                MemoryItem(
                    type=MemoryType.SEMANTIC,
                    content=f"[프로젝트 메타] {', '.join(meta_parts)}",
                    tags=[_TAG_CONVENTION],
                    metadata={"source": "pyproject.toml"},
                )
            )

        # 주요 의존성 추출
        deps_match = re.search(
            r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL
        )
        if deps_match:
            deps_text = deps_match.group(1)
            deps = re.findall(r'"([^"]+)"', deps_text)
            if deps:
                dep_names = [
                    d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
                    for d in deps
                ]
                items.append(
                    MemoryItem(
                        type=MemoryType.SEMANTIC,
                        content=f"[주요 의존성] {', '.join(dep_names)}",
                        tags=[_TAG_TECH_STACK],
                        metadata={"source": "pyproject.toml", "count": len(dep_names)},
                    )
                )

        for item in items:
            self._store.put(item)

        return len(items)
