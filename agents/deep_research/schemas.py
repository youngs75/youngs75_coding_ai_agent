"""Deep Research 에이전트 구조화 출력 스키마 및 상태 정의."""

from __future__ import annotations

from typing import Annotated, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from youngs75_a2a.core.base_state import BaseGraphState
from youngs75_a2a.core.reducers import override_reducer


# --- 구조화 출력 스키마 ---

class ClarifyWithUser(BaseModel):
    """사용자 질문 명확화 결과."""
    need_clarification: bool = Field(description="추가 정보가 필요한지 여부")
    question: str = Field(description="명확화가 필요한 경우 질문")
    verification: str = Field(description="명확화 판단의 근거")


class ResearchQuestion(BaseModel):
    """연구 브리프."""
    research_brief: str = Field(description="구체적인 연구 질문")


class ConductResearch(BaseModel):
    """연구 작업 실행 도구."""
    research_topic: str = Field(description="연구 주제")


class ResearchComplete(BaseModel):
    """연구 완료 신호 도구."""
    pass


# --- 에이전트 상태 ---

class AgentState(BaseGraphState):
    """Deep Research 에이전트 메인 상태."""
    supervisor_messages: Annotated[list[BaseMessage], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    final_report: str


class HITLAgentState(AgentState):
    """HITL 지원 에이전트 상태."""
    revision_count: int
    human_feedback: Optional[str]


# --- Researcher 서브그래프 상태 ---

class ResearcherInputState(TypedDict):
    researcher_messages: Annotated[list[BaseMessage], add_messages]
    research_topic: str


class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: list[str]


class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[BaseMessage], add_messages]
    research_topic: str
    compressed_research: str
    raw_notes: list[str]
    tool_call_iterations: int


# --- Supervisor 서브그래프 상태 ---

class SupervisorInputState(TypedDict):
    supervisor_messages: Annotated[list[BaseMessage], override_reducer]
    research_brief: str


class SupervisorOutputState(TypedDict):
    supervisor_messages: Annotated[list[BaseMessage], override_reducer]
    research_iterations: int
    notes: list[str]
    raw_notes: list[str]


class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[BaseMessage], override_reducer]
    research_brief: str
    research_iterations: int
    notes: Annotated[list[str], override_reducer]
    raw_notes: Annotated[list[str], override_reducer]
