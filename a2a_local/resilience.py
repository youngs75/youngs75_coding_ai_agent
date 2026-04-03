"""A2A 에이전트 에러 핸들링 및 복원력 모듈.

에이전트 간 통신의 안정성을 보장하기 위한 패턴을 제공한다:
  - RetryPolicy: 재시도 정책 (지수 백오프)
  - CircuitBreaker: 서킷 브레이커 (연속 실패 시 차단)
  - AgentMonitor: 에이전트 상태 모니터링
  - ResilientA2AClient: 복원력이 내장된 A2A 클라이언트 래퍼

사용 예:
    client = ResilientA2AClient(
        url="http://localhost:8080",
        retry_policy=RetryPolicy(max_retries=3),
    )
    response = await client.send_message("Hello")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import httpx

from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    TaskState,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── 재시도 정책 ──────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """재시도 정책.

    지수 백오프(exponential backoff)를 적용한 재시도를 수행한다.

    Attributes:
        max_retries: 최대 재시도 횟수
        base_delay: 기본 대기 시간 (초)
        max_delay: 최대 대기 시간 (초)
        exponential_base: 지수 백오프 기저값
        retryable_exceptions: 재시도 대상 예외 타입 목록
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        asyncio.TimeoutError,
        ConnectionError,
        OSError,
    )

    def compute_delay(self, attempt: int) -> float:
        """현재 시도에 대한 대기 시간 계산 (지수 백오프)."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    def is_retryable(self, error: Exception) -> bool:
        """해당 예외가 재시도 대상인지 확인."""
        return isinstance(error, self.retryable_exceptions)


# ── 서킷 브레이커 ────────────────────────────────────────────


class CircuitState(str, Enum):
    """서킷 브레이커 상태."""

    CLOSED = "closed"      # 정상 — 요청 허용
    OPEN = "open"          # 차단 — 요청 거부
    HALF_OPEN = "half_open"  # 반개방 — 제한적 요청 허용 (테스트)


@dataclass
class CircuitBreaker:
    """서킷 브레이커 패턴.

    연속 실패가 임계치를 초과하면 일시적으로 요청을 차단하여
    과부하된 서비스에 복구 시간을 제공한다.

    Attributes:
        failure_threshold: 서킷 개방 임계치 (연속 실패 횟수)
        recovery_timeout: 서킷 반개방까지 대기 시간 (초)
        half_open_max_calls: 반개방 상태에서 허용할 최대 시도 횟수
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1

    # 내부 상태
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """현재 서킷 상태 (시간 기반 자동 전이 포함)."""
        if self._state == CircuitState.OPEN:
            # 복구 대기 시간 경과 → 반개방으로 전이
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("서킷 브레이커: OPEN → HALF_OPEN 전이")
        return self._state

    def can_execute(self) -> bool:
        """현재 요청 실행 가능 여부."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        if current_state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max_calls
        # OPEN 상태
        return False

    def record_success(self) -> None:
        """성공 기록 — 서킷을 닫는다."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("서킷 브레이커: HALF_OPEN → CLOSED 전이 (성공)")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0

    def record_failure(self) -> None:
        """실패 기록 — 임계치 초과 시 서킷을 연다."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # 반개방 상태에서 실패 → 다시 개방
            self._state = CircuitState.OPEN
            logger.info("서킷 브레이커: HALF_OPEN → OPEN 전이 (실패)")
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"서킷 브레이커: CLOSED → OPEN 전이 "
                f"(연속 실패 {self._failure_count}회)"
            )

    def reset(self) -> None:
        """서킷 브레이커를 초기 상태(CLOSED)로 리셋."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0


class CircuitOpenError(Exception):
    """서킷이 열려 있어 요청이 차단되었을 때 발생하는 예외."""

    def __init__(self, url: str):
        self.url = url
        super().__init__(f"서킷이 열려 있어 요청 차단됨: {url}")


# ── 에이전트 상태 모니터링 ───────────────────────────────────


@dataclass
class AgentHealthStats:
    """에이전트 헬스 통계."""

    url: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None
    last_error_time: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED

    @property
    def success_rate(self) -> float:
        """성공률 (0.0 ~ 1.0)."""
        if self.total_requests == 0:
            return 1.0  # 요청 없으면 건강으로 간주
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """평균 응답 시간 (밀리초)."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests


class AgentMonitor:
    """에이전트 상태 모니터링.

    각 에이전트별 요청 성공/실패, 응답 시간, 서킷 상태를 추적한다.
    """

    def __init__(self) -> None:
        self._stats: dict[str, AgentHealthStats] = {}

    def get_stats(self, url: str) -> AgentHealthStats:
        """에이전트별 통계 조회 (없으면 생성)."""
        if url not in self._stats:
            self._stats[url] = AgentHealthStats(url=url)
        return self._stats[url]

    def record_success(self, url: str, latency_ms: float) -> None:
        """성공 기록."""
        stats = self.get_stats(url)
        stats.total_requests += 1
        stats.successful_requests += 1
        stats.total_latency_ms += latency_ms

    def record_failure(self, url: str, error: str) -> None:
        """실패 기록."""
        stats = self.get_stats(url)
        stats.total_requests += 1
        stats.failed_requests += 1
        stats.last_error = error
        stats.last_error_time = time.time()

    def update_circuit_state(self, url: str, state: CircuitState) -> None:
        """서킷 브레이커 상태 업데이트."""
        self.get_stats(url).circuit_state = state

    def get_all_stats(self) -> dict[str, AgentHealthStats]:
        """모든 에이전트 통계."""
        return dict(self._stats)

    def get_healthy_urls(self, min_success_rate: float = 0.5) -> list[str]:
        """건강한 에이전트 URL 목록."""
        return [
            url for url, stats in self._stats.items()
            if stats.success_rate >= min_success_rate
            and stats.circuit_state != CircuitState.OPEN
        ]


# ── 복원력 내장 A2A 클라이언트 ────────────────────────────────


class ResilientA2AClient:
    """복원력이 내장된 A2A 클라이언트 래퍼.

    - 타임아웃 처리
    - 재시도 (지수 백오프)
    - 서킷 브레이커
    - 폴백 에이전트 지원
    - 상태 모니터링 연동

    사용 예:
        client = ResilientA2AClient(
            url="http://localhost:8080",
            retry_policy=RetryPolicy(max_retries=3),
        )
        response = await client.send_message("Hello")
    """

    def __init__(
        self,
        url: str,
        *,
        retry_policy: RetryPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        monitor: AgentMonitor | None = None,
        fallback_urls: list[str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.url = url
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.monitor = monitor or AgentMonitor()
        self.fallback_urls = fallback_urls or []
        self.timeout = timeout

    async def send_message(
        self,
        content: str,
        *,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> SendMessageResponse:
        """메시지를 에이전트에 전송한다 (복원력 내장).

        Args:
            content: 전송할 텍스트 내용
            context_id: 대화 컨텍스트 ID
            task_id: 태스크 ID

        Returns:
            A2A SendMessageResponse

        Raises:
            CircuitOpenError: 서킷이 열려 있을 때
            Exception: 모든 재시도 및 폴백이 실패했을 때
        """
        # 1차: 메인 에이전트에 시도
        try:
            return await self._send_with_retry(self.url, content)
        except CircuitOpenError:
            logger.warning(f"메인 에이전트 서킷 열림: {self.url}")
        except Exception as e:
            logger.warning(f"메인 에이전트 요청 실패: {self.url} - {e}")

        # 2차: 폴백 에이전트에 시도
        for fallback_url in self.fallback_urls:
            try:
                logger.info(f"폴백 에이전트 시도: {fallback_url}")
                return await self._send_with_retry(fallback_url, content)
            except Exception as e:
                logger.warning(f"폴백 에이전트 실패: {fallback_url} - {e}")

        raise ConnectionError(
            f"모든 에이전트 요청 실패: 메인={self.url}, "
            f"폴백={self.fallback_urls}"
        )

    async def send_message_streaming(
        self,
        content: str,
    ) -> AsyncStreamingResponse:
        """스트리밍 메시지를 에이전트에 전송한다 (복원력 내장).

        Args:
            content: 전송할 텍스트 내용

        Returns:
            스트리밍 응답을 감싸는 AsyncStreamingResponse

        Raises:
            CircuitOpenError: 서킷이 열려 있을 때
        """
        if not self.circuit_breaker.can_execute():
            self.monitor.update_circuit_state(self.url, self.circuit_breaker.state)
            raise CircuitOpenError(self.url)

        msg = create_text_message_object(content=content)
        request = SendStreamingMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=msg),
        )

        start_time = time.time()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout)
            ) as hc:
                client = A2AClient(httpx_client=hc, url=self.url)
                stream = client.send_message_streaming(request)
                return AsyncStreamingResponse(
                    stream=stream,
                    url=self.url,
                    circuit_breaker=self.circuit_breaker,
                    monitor=self.monitor,
                    start_time=start_time,
                )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.circuit_breaker.record_failure()
            self.monitor.record_failure(self.url, str(e))
            self.monitor.update_circuit_state(self.url, self.circuit_breaker.state)
            raise

    async def _send_with_retry(
        self,
        url: str,
        content: str,
    ) -> SendMessageResponse:
        """재시도 로직이 포함된 메시지 전송.

        Args:
            url: 대상 에이전트 URL
            content: 전송할 텍스트

        Returns:
            A2A SendMessageResponse
        """
        # 서킷 브레이커 체크
        if not self.circuit_breaker.can_execute():
            self.monitor.update_circuit_state(url, self.circuit_breaker.state)
            raise CircuitOpenError(url)

        last_error: Exception | None = None

        for attempt in range(self.retry_policy.max_retries + 1):
            start_time = time.time()
            try:
                msg = create_text_message_object(content=content)
                request = SendMessageRequest(
                    id=str(uuid.uuid4()),
                    params=MessageSendParams(message=msg),
                )

                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout)
                ) as hc:
                    client = A2AClient(httpx_client=hc, url=url)
                    response = await client.send_message(request)

                # 성공
                latency_ms = (time.time() - start_time) * 1000
                self.circuit_breaker.record_success()
                self.monitor.record_success(url, latency_ms)
                self.monitor.update_circuit_state(url, self.circuit_breaker.state)
                return response

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                last_error = e

                # 재시도 가능 여부 판단
                if not self.retry_policy.is_retryable(e):
                    self.circuit_breaker.record_failure()
                    self.monitor.record_failure(url, str(e))
                    self.monitor.update_circuit_state(url, self.circuit_breaker.state)
                    raise

                # 마지막 시도가 아니면 대기 후 재시도
                if attempt < self.retry_policy.max_retries:
                    delay = self.retry_policy.compute_delay(attempt)
                    logger.info(
                        f"재시도 {attempt + 1}/{self.retry_policy.max_retries} "
                        f"({url}): {delay:.1f}초 후 재시도"
                    )
                    await asyncio.sleep(delay)

        # 모든 재시도 실패
        self.circuit_breaker.record_failure()
        self.monitor.record_failure(url, str(last_error))
        self.monitor.update_circuit_state(url, self.circuit_breaker.state)
        raise last_error  # type: ignore[misc]


class AsyncStreamingResponse:
    """스트리밍 응답 래퍼.

    서킷 브레이커 및 모니터링과 연동하여
    스트리밍 응답의 성공/실패를 추적한다.
    """

    def __init__(
        self,
        stream: Any,
        url: str,
        circuit_breaker: CircuitBreaker,
        monitor: AgentMonitor,
        start_time: float,
    ) -> None:
        self._stream = stream
        self._url = url
        self._circuit_breaker = circuit_breaker
        self._monitor = monitor
        self._start_time = start_time
        self._completed = False

    async def __aiter__(self):
        """스트리밍 응답을 비동기 이터레이션."""
        try:
            async for event in self._stream:
                yield event
            # 스트리밍 완료
            self._completed = True
            latency_ms = (time.time() - self._start_time) * 1000
            self._circuit_breaker.record_success()
            self._monitor.record_success(self._url, latency_ms)
        except Exception as e:
            self._circuit_breaker.record_failure()
            self._monitor.record_failure(self._url, str(e))
            raise
