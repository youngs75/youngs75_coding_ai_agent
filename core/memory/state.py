"""CoALA 메모리 체계를 반영한 확장 상태.

Working Memory는 BaseGraphState.messages로 이미 제공된다.
이 모듈은 Semantic 메모리를 상태에 통합하여
에이전트가 프로젝트 규칙/컨벤션을 참조할 수 있게 한다.

Agent-as-a-Judge 교훈:
  Episodic/Procedural은 오류 전파 위험이 있으므로
  초기에는 semantic만 상태에 포함한다.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from youngs75_a2a.core.reducers import override_reducer


class MemoryAwareState(TypedDict, total=False):
    """CoALA 메모리 체계를 반영한 에이전트 상태.

    Attributes:
        messages: Working Memory — 현재 대화 컨텍스트 (add_messages 누적)
        semantic_context: Semantic Memory — 프로젝트 규칙/컨벤션 (덮어쓰기 가능)
        episodic_log: Episodic Memory — 실행 결과 이력 (덮어쓰기 가능, 실험적)
    """

    messages: Annotated[list[BaseMessage], add_messages]
    semantic_context: Annotated[list[str], override_reducer]
    episodic_log: Annotated[list[str], override_reducer]
