"""동적 SubAgent 스키마.

Puppeteer 논문 원칙: "효율성은 제거에서 온다"
— 모든 에이전트를 항상 활성화하지 않고, 태스크에 필요한 것만 동적 선택한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SubAgentStatus(str, Enum):
    """서브에이전트 상태."""

    AVAILABLE = "available"
    BUSY = "busy"
    FAILED = "failed"
    DISABLED = "disabled"


class SubAgentSpec(BaseModel):
    """서브에이전트 사양 — 능력, 비용, 상태를 기술한다."""

    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    endpoint: str | None = None  # A2A 엔드포인트 또는 None(로컬)
    model_tier: str = "default"
    cost_weight: float = 1.0  # 비용 가중치 (높을수록 비쌈)
    status: SubAgentStatus = SubAgentStatus.AVAILABLE
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentUsageRecord(BaseModel):
    """서브에이전트 사용 기록."""

    agent_name: str
    task_type: str
    success: bool
    duration_ms: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SelectionResult(BaseModel):
    """에이전트 선택 결과."""

    agent: SubAgentSpec
    score: float
    reason: str
