"""에이전트 시스템 구조화된 예외 계층.

모든 에이전트 관련 예외는 AgentError를 상속하며,
도메인별로 세분화된 예외 타입을 제공한다.

계층 구조:
    AgentError
    ├── SubAgentError          — SubAgent 실행 관련
    ├── MemoryError            — 메모리 시스템
    ├── ToolCallError          — 도구 호출
    └── ResilienceError        — 복원력 메커니즘
        ├── StallDetectedError       — 무진전 루프 감지
        ├── BudgetExceededError      — 토큰/턴 예산 초과
        └── ModelFallbackExhaustedError — 모든 fallback 모델 실패
"""

from __future__ import annotations


class AgentError(Exception):
    """에이전트 시스템 최상위 예외.

    모든 에이전트 관련 예외의 기반 클래스. 추가 컨텍스트를
    details 딕셔너리로 전달할 수 있다.

    Args:
        message: 오류 메시지.
        **kwargs: 추가 컨텍스트 (details에 저장).
    """

    def __init__(self, message: str = "", **kwargs):
        self.details = kwargs
        super().__init__(message)


class SubAgentError(AgentError):
    """SubAgent 실행 관련 예외.

    Args:
        agent_id: 실패한 SubAgent의 식별자.
        message: 오류 메시지.
        cause: 원인 예외 (있는 경우).
    """

    def __init__(
        self,
        agent_id: str,
        message: str = "",
        *,
        cause: Exception | None = None,
    ):
        self.agent_id = agent_id
        self.cause = cause
        super().__init__(message, agent_id=agent_id)

    def __str__(self) -> str:
        base = f"SubAgent [{self.agent_id}]: {self.args[0]}"
        if self.cause:
            base += f" (원인: {self.cause})"
        return base


class MemoryError(AgentError):
    """메모리 시스템 예외.

    Args:
        message: 오류 메시지.
        memory_type: 관련 메모리 타입 (있는 경우).
    """

    def __init__(self, message: str = "", *, memory_type: str | None = None):
        self.memory_type = memory_type
        super().__init__(message, memory_type=memory_type)


class ToolCallError(AgentError):
    """도구 호출 예외.

    Args:
        tool_name: 실패한 도구 이름.
        message: 오류 메시지.
        call_id: 도구 호출 ID (있는 경우).
    """

    def __init__(
        self,
        tool_name: str,
        message: str = "",
        *,
        call_id: str | None = None,
    ):
        self.tool_name = tool_name
        self.call_id = call_id
        super().__init__(message, tool_name=tool_name, call_id=call_id)

    def __str__(self) -> str:
        return f"ToolCallError [{self.tool_name}]: {self.args[0]}"


class ResilienceError(AgentError):
    """복원력 메커니즘 예외.

    재시도, fallback, 무진전 루프 감지 등 복원력 관련
    예외의 기반 클래스.
    """


class StallDetectedError(ResilienceError):
    """무진전 루프 감지.

    에이전트가 진전 없이 동일한 행동을 반복할 때 발생한다.

    Args:
        message: 오류 메시지.
        turns: 감지 시점까지의 반복 턴 수.
    """

    def __init__(self, message: str = "무진전 루프 감지", *, turns: int = 0):
        self.turns = turns
        super().__init__(message, turns=turns)


class BudgetExceededError(ResilienceError):
    """토큰/턴 예산 초과.

    Args:
        message: 오류 메시지.
        budget_type: 초과된 예산 유형 ("token" 또는 "turn").
        limit: 설정된 한도.
        actual: 실제 사용량.
    """

    def __init__(
        self,
        message: str = "예산 초과",
        *,
        budget_type: str = "token",
        limit: int = 0,
        actual: int = 0,
    ):
        self.budget_type = budget_type
        self.limit = limit
        self.actual = actual
        super().__init__(
            message, budget_type=budget_type, limit=limit, actual=actual
        )


class ModelFallbackExhaustedError(ResilienceError):
    """모든 fallback 모델 실패.

    Args:
        message: 오류 메시지.
        tried_models: 시도한 모델 목록.
    """

    def __init__(
        self,
        message: str = "모든 fallback 모델 소진",
        *,
        tried_models: list[str] | None = None,
    ):
        self.tried_models = tried_models or []
        super().__init__(message, tried_models=self.tried_models)
