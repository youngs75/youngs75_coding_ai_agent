"""미들웨어 레이어 — LLM 호출 전후 composable 가공 파이프라인.

DeepAgents의 미들웨어 패턴을 차용하여 구현.
"""

from .base import AgentMiddleware, Handler, ModelRequest, ModelResponse, append_to_system_message
from .chain import MiddlewareChain
from .memory import MemoryMiddleware
from .message_window import MessageWindowMiddleware
from .skill import SkillMiddleware
from .summarization import SummarizationMiddleware

__all__ = [
    "AgentMiddleware",
    "Handler",
    "MemoryMiddleware",
    "MessageWindowMiddleware",
    "MiddlewareChain",
    "ModelRequest",
    "ModelResponse",
    "SkillMiddleware",
    "SummarizationMiddleware",
    "append_to_system_message",
]
