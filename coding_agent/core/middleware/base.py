"""미들웨어 베이스 프로토콜.

DeepAgents의 AgentMiddleware 패턴을 차용하여,
LLM 호출 전후에 메시지/도구/시스템프롬프트를 가공하는 composable 미들웨어를 정의한다.

양파(Onion) 패턴:
    Request → [MW1] → [MW2] → [MW3] → LLM 호출
    Response ← [MW1] ← [MW2] ← [MW3] ← LLM 응답
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.messages import BaseMessage, SystemMessage


# ── 요청/응답 타입 ──


@dataclass
class ModelRequest:
    """LLM 호출 요청 — 미들웨어가 수정 가능.

    불변 패턴: override()로 수정된 복사본을 생성한다.
    """

    system_message: str
    messages: list[BaseMessage]
    tools: list[Any] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def override(self, **kwargs: Any) -> ModelRequest:
        """지정된 필드만 교체한 새 ModelRequest를 반환한다."""
        current = {
            "system_message": self.system_message,
            "messages": self.messages,
            "tools": self.tools,
            "state": self.state,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }
        current.update(kwargs)
        return ModelRequest(**current)

    @property
    def all_messages(self) -> list[BaseMessage]:
        """SystemMessage + messages를 합친 전체 메시지 리스트."""
        result: list[BaseMessage] = []
        if self.system_message:
            result.append(SystemMessage(content=self.system_message))
        result.extend(self.messages)
        return result


@dataclass
class ModelResponse:
    """LLM 호출 응답."""

    message: BaseMessage
    state_update: dict[str, Any] | None = None


# ── 핸들러 타입 ──

Handler = Callable[[ModelRequest], Awaitable[ModelResponse]]


# ── 미들웨어 베이스 ──


class AgentMiddleware(ABC):
    """미들웨어 베이스 클래스.

    서브클래스는 wrap_model_call()을 구현하여
    LLM 호출 전후에 request/response를 가공한다.

    사용 예:
        class MyMiddleware(AgentMiddleware):
            async def wrap_model_call(self, request, handler):
                # before: request 수정
                modified = request.override(system_message=request.system_message + "\\n추가 지시")
                # LLM 호출
                response = await handler(modified)
                # after: response 수정
                return response
    """

    @abstractmethod
    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        """handler(request)를 호출하되, 전후로 request/response를 가공한다.

        반드시 await handler(request)를 호출해야 체인이 이어진다.
        """
        ...


# ── 유틸리티 ──


def append_to_system_message(system_message: str, text: str) -> str:
    """시스템 메시지에 텍스트를 안전하게 추가한다.

    DeepAgents의 append_to_system_message 패턴을 적용:
    - 기존 내용이 있으면 \\n\\n 구분자 추가
    - None/빈 문자열이면 텍스트만 반환
    """
    if not system_message:
        return text
    return f"{system_message}\n\n{text}"
