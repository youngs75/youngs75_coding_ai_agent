"""메모리 주입 + 자동 축적 미들웨어.

에이전트의 메모리(semantic, episodic, procedural, user_profile, domain_knowledge)를
시스템 프롬프트에 주입한다.

자동 축적 기능:
  after 단계에서 SLM(FAST 모델)으로 "이 대화에서 새로 학습할 지식이 있는가?" 판단 후
  있으면 accumulate_domain_knowledge() 또는 accumulate_user_profile() 호출.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from langchain_core.messages import BaseMessage, SystemMessage

from coding_agent.core.memory.schemas import MemoryType
from coding_agent.core.memory.store import MemoryStore
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

# purpose → 검색할 MemoryType 매핑
_PURPOSE_MEMORY_MAP: dict[str, list[MemoryType]] = {
    "generation": [
        MemoryType.PROCEDURAL,
        MemoryType.SEMANTIC,
        MemoryType.DOMAIN_KNOWLEDGE,
    ],
    "planning": [
        MemoryType.SEMANTIC,
        MemoryType.DOMAIN_KNOWLEDGE,
        MemoryType.USER_PROFILE,
    ],
    "verification": [
        MemoryType.SEMANTIC,
        MemoryType.DOMAIN_KNOWLEDGE,
    ],
    "parsing": [
        MemoryType.PROCEDURAL,
    ],
    "tool_planning": [
        MemoryType.PROCEDURAL,
        MemoryType.SEMANTIC,
    ],
    "default": [
        MemoryType.SEMANTIC,
        MemoryType.DOMAIN_KNOWLEDGE,
        MemoryType.USER_PROFILE,
    ],
}

# 지식 추출 판단 프롬프트
_KNOWLEDGE_EXTRACTION_PROMPT = (
    "다음 대화 응답에서 새로 학습할 도메인 지식이나 사용자 선호가 있는지 판단하라.\n"
    "있으면 JSON 형식으로 반환: "
    '{{"type": "domain_knowledge"|"user_profile", "content": "...", "tags": [...]}}\n'
    "없으면 빈 문자열을 반환하라.\n\n"
    "응답:\n{response_text}"
)

# SLM 호출 타입: (messages) → str
SLMInvoker = Callable[[list[BaseMessage]], Awaitable[str]]


class MemoryMiddleware(AgentMiddleware):
    """메모리 주입 + 자동 축적 미들웨어.

    before: state 기반 메모리 주입 + MemoryStore 검색 결과 주입.
    after: SLM으로 응답에서 새로운 지식/선호 감지 → 자동 축적.

    Args:
        memory_store: MemoryStore 인스턴스 (검색/축적에 사용).
            None이면 state 기반 주입만 수행.
        memory_sources: 추가 메모리 소스 (미래 확장용).
        slm_invoker: 지식 추출 판단용 SLM(FAST 모델) 호출 함수.
            None이면 자동 축적을 수행하지 않는다.
        memory_limit: purpose별 최대 검색 결과 수.
        auto_accumulate: after 단계에서 자동 축적 활성화 여부.
    """

    def __init__(
        self,
        memory_store: MemoryStore | None = None,
        memory_sources: list[str] | None = None,
        slm_invoker: SLMInvoker | None = None,
        memory_limit: int = 5,
        auto_accumulate: bool = True,
    ) -> None:
        self._memory_store = memory_store
        self._memory_sources = memory_sources or []
        self._slm_invoker = slm_invoker
        self._memory_limit = memory_limit
        self._auto_accumulate = auto_accumulate

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        """메모리 주입 → LLM 호출 → 자동 축적.

        Args:
            request: LLM 호출 요청.
            handler: 다음 미들웨어 또는 최종 LLM 호출.

        Returns:
            LLM 응답.
        """
        # before: 메모리 주입
        request = self._inject_state_memories(request)
        request = self._inject_store_memories(request)

        # LLM 호출
        response = await handler(request)

        # after: 자동 축적
        if self._auto_accumulate and self._slm_invoker and self._memory_store:
            await self._try_accumulate(response)

        return response

    def _inject_state_memories(self, request: ModelRequest) -> ModelRequest:
        """state 딕셔너리에서 메모리 섹션을 system prompt에 주입한다.

        Args:
            request: LLM 호출 요청.

        Returns:
            메모리가 주입된 새 요청.
        """
        state = request.state or {}
        sections: list[str] = []

        for key, title in _MEMORY_SECTIONS:
            content = state.get(key)
            if not content:
                continue
            if isinstance(content, list):
                items = "\n".join(f"- {item}" for item in content if item)
            else:
                items = str(content)
            if items.strip():
                sections.append(f"### {title}\n{items}")

        if not sections:
            return request

        memory_block = "## 메모리 컨텍스트\n" + "\n\n".join(sections)
        new_system = append_to_system_message(request.system_message, memory_block)
        logger.debug("[MemoryMiddleware] %d개 state 메모리 섹션 주입", len(sections))
        return request.override(system_message=new_system)

    def _inject_store_memories(self, request: ModelRequest) -> ModelRequest:
        """MemoryStore에서 purpose 기반 검색하여 system prompt에 주입한다.

        Args:
            request: LLM 호출 요청.

        Returns:
            메모리가 주입된 새 요청.
        """
        if not self._memory_store:
            return request

        purpose = request.metadata.get("purpose", "default")
        memory_types = _PURPOSE_MEMORY_MAP.get(
            purpose, _PURPOSE_MEMORY_MAP["default"]
        )

        query = self._extract_query(request.messages)
        if not query:
            return request

        retrieved = []
        for mem_type in memory_types:
            results = self._memory_store.search(
                query,
                memory_type=mem_type,
                limit=self._memory_limit,
            )
            retrieved.extend(results)

        if not retrieved:
            return request

        memory_text = _format_memories(retrieved)
        new_system = append_to_system_message(request.system_message, memory_text)
        logger.debug(
            "[MemoryMiddleware] %d개 store 메모리 주입 (purpose=%s)",
            len(retrieved),
            purpose,
        )
        return request.override(system_message=new_system)

    async def _try_accumulate(self, response: ModelResponse) -> None:
        """SLM으로 응답에서 지식 추출을 시도한다.

        축적 실패는 LLM 응답 흐름을 방해하지 않는다.

        Args:
            response: LLM 응답.
        """
        try:
            response_text = _extract_response_text(response)
            if not response_text:
                return

            prompt = _KNOWLEDGE_EXTRACTION_PROMPT.format(
                response_text=response_text[:2000],
            )
            slm_messages = [SystemMessage(content=prompt)]
            result = await self._slm_invoker(slm_messages)  # type: ignore[misc]

            if not result or not result.strip():
                return

            self._process_extraction(result)
        except Exception as exc:
            logger.warning("[MemoryMiddleware] 자동 축적 실패 (무시): %s", exc)

    def _process_extraction(self, raw_result: str) -> None:
        """SLM 추출 결과를 파싱하여 메모리에 저장한다.

        Args:
            raw_result: SLM이 반환한 JSON 문자열.
        """
        try:
            data = json.loads(raw_result.strip())
        except json.JSONDecodeError:
            return

        if not isinstance(data, dict) or "type" not in data:
            return

        content = data.get("content", "")
        tags = data.get("tags", [])
        if not content or not self._memory_store:
            return

        if data["type"] == "domain_knowledge":
            self._memory_store.accumulate_domain_knowledge(
                content=content, tags=tags, source="auto"
            )
            logger.info("[MemoryMiddleware] 도메인 지식 자동 축적: %s", content[:50])
        elif data["type"] == "user_profile":
            self._memory_store.accumulate_user_profile(
                content=content, tags=tags, source="auto"
            )
            logger.info("[MemoryMiddleware] 사용자 프로필 자동 축적: %s", content[:50])

    @staticmethod
    def _extract_query(messages: list[BaseMessage]) -> str:
        """메시지 리스트에서 검색 쿼리를 추출한다.

        Args:
            messages: LLM 메시지 리스트.

        Returns:
            마지막 메시지의 content (최대 500자).
        """
        for msg in reversed(messages):
            content = msg.content if isinstance(msg.content, str) else ""
            if content.strip():
                return content[:500]
        return ""


def _format_memories(items: list[Any]) -> str:
    """메모리 항목들을 system prompt 텍스트로 포맷한다.

    Args:
        items: 검색된 MemoryItem 리스트.

    Returns:
        포맷된 텍스트.
    """
    lines = ["## 검색된 메모리"]
    for item in items:
        tag_str = ", ".join(item.tags) if item.tags else ""
        prefix = f"[{item.type.value}]"
        if tag_str:
            prefix += f" ({tag_str})"
        lines.append(f"- {prefix}: {item.content[:200]}")
    return "\n".join(lines)


def _extract_response_text(response: ModelResponse) -> str:
    """ModelResponse에서 텍스트를 추출한다.

    Args:
        response: LLM 응답.

    Returns:
        텍스트 문자열.
    """
    if hasattr(response.message, "content"):
        content = response.message.content
        return content if isinstance(content, str) else str(content)
    return ""
