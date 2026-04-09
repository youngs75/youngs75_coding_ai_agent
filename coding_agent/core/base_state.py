"""LangGraph 에이전트 공통 상태 스키마."""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BaseGraphState(TypedDict):
    """모든 LangGraph 에이전트의 기본 상태.

    messages 필드는 add_messages reducer를 사용하여
    메시지를 누적 관리한다.
    """

    messages: Annotated[list[BaseMessage], add_messages]
