"""메모리 주입 미들웨어.

에이전트의 메모리(semantic, episodic, procedural, user_profile, domain_knowledge)를
시스템 프롬프트에 주입한다. AGENTS.md와 같은 파일 기반 메모리도 지원한다.
"""

from __future__ import annotations

import logging
from typing import Any

from coding_agent.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
    append_to_system_message,
)

logger = logging.getLogger(__name__)

# state 키 → 섹션 제목 매핑
_MEMORY_SECTIONS: list[tuple[str, str]] = [
    ("semantic_context", "Semantic"),
    ("episodic_log", "Episodic"),
    ("procedural_skills", "Procedural"),
    ("user_profile_context", "User Profile"),
    ("domain_knowledge_context", "Domain Knowledge"),
]


class MemoryMiddleware(AgentMiddleware):
    """에이전트 메모리를 시스템 프롬프트에 주입하는 미들웨어."""

    def __init__(
        self,
        memory_store: Any | None = None,
        memory_sources: list[str] | None = None,
    ) -> None:
        self._memory_store = memory_store
        self._memory_sources = memory_sources or []

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        state = request.state or {}

        # 각 메모리 유형에서 비어있지 않은 섹션만 수집
        sections: list[str] = []
        for key, title in _MEMORY_SECTIONS:
            content = state.get(key)
            if not content:
                continue

            # 리스트면 항목별, 문자열이면 그대로
            if isinstance(content, list):
                items = "\n".join(f"- {item}" for item in content if item)
            else:
                items = str(content)

            if items.strip():
                sections.append(f"### {title}\n{items}")

        if not sections:
            return await handler(request)

        # 메모리 컨텍스트 블록 구성
        memory_block = "## 메모리 컨텍스트\n" + "\n\n".join(sections)
        new_system = append_to_system_message(request.system_message, memory_block)

        logger.debug("[MemoryMiddleware] %d개 메모리 섹션 주입", len(sections))
        return await handler(request.override(system_message=new_system))
