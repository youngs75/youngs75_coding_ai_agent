"""A2A 스트리밍 응답 지원 모듈.

에이전트 간 실시간 스트리밍 통신을 위한 유틸리티를 제공한다.

주요 컴포넌트:
  - StreamingResponseCollector: 스트리밍 응답 수집/변환
  - stream_agent_response: 에이전트 스트리밍 응답을 수집하는 헬퍼

사용 예:
    collector = StreamingResponseCollector()
    async for chunk in collector.collect_stream(url, "질문"):
        print(chunk)
    final = collector.get_final_text()
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import (
    MessageSendParams,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """스트리밍 청크 데이터."""

    text: str
    # 전체 누적 텍스트
    accumulated: str
    # 태스크 상태
    state: str = "working"
    # 청크 인덱스
    index: int = 0
    # 시간 정보 (밀리초)
    elapsed_ms: float = 0.0


class StreamingResponseCollector:
    """A2A 스트리밍 응답 수집기.

    스트리밍 응답을 청크 단위로 수집하면서
    누적 텍스트와 상태를 관리한다.
    """

    def __init__(
        self,
        *,
        timeout: float = 300.0,
        chunk_callback: Any | None = None,
    ) -> None:
        """
        Args:
            timeout: 전체 스트리밍 타임아웃 (초)
            chunk_callback: 각 청크 수신 시 호출할 콜백 (async callable)
        """
        self._timeout = timeout
        self._chunk_callback = chunk_callback
        self._accumulated_text = ""
        self._chunks: list[StreamChunk] = []
        self._final_text: str | None = None
        self._start_time: float = 0.0

    @property
    def accumulated_text(self) -> str:
        """현재까지 누적된 텍스트."""
        return self._accumulated_text

    @property
    def chunks(self) -> list[StreamChunk]:
        """수집된 청크 목록."""
        return list(self._chunks)

    def get_final_text(self) -> str:
        """최종 결과 텍스트."""
        return self._final_text or self._accumulated_text

    async def collect_stream(
        self,
        url: str,
        content: str,
    ) -> AsyncIterator[StreamChunk]:
        """에이전트에 스트리밍 요청을 보내고 청크를 수집한다.

        Args:
            url: 에이전트 URL
            content: 전송할 메시지

        Yields:
            StreamChunk — 각 청크 데이터
        """
        self._start_time = time.time()
        self._accumulated_text = ""
        self._chunks = []
        self._final_text = None

        msg = create_text_message_object(content=content)
        request = SendStreamingMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(message=msg),
        )

        chunk_index = 0

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as hc:
            client = A2AClient(httpx_client=hc, url=url)

            async for event in client.send_message_streaming(request):
                elapsed = (time.time() - self._start_time) * 1000
                text = self._extract_text_from_event(event)

                if text:
                    self._accumulated_text += text
                    chunk = StreamChunk(
                        text=text,
                        accumulated=self._accumulated_text,
                        state=self._extract_state(event),
                        index=chunk_index,
                        elapsed_ms=elapsed,
                    )
                    self._chunks.append(chunk)
                    chunk_index += 1

                    # 콜백 호출
                    if self._chunk_callback:
                        await self._chunk_callback(chunk)

                    yield chunk

                # 아티팩트에서 최종 텍스트 추출
                final = self._extract_final_text(event)
                if final:
                    self._final_text = final

    @staticmethod
    def _extract_text_from_event(
        event: SendStreamingMessageResponse,
    ) -> str:
        """스트리밍 이벤트에서 텍스트를 추출한다."""
        try:
            root = event.root
            if not hasattr(root, "result"):
                return ""
            result = root.result

            # TaskStatusUpdateEvent — 상태 메시지에서 텍스트 추출
            if hasattr(result, "status") and hasattr(result, "id"):
                status = result.status
                if hasattr(status, "message") and status.message:
                    msg = status.message
                    if hasattr(msg, "parts") and msg.parts:
                        parts_text = []
                        for part in msg.parts:
                            p_root = getattr(part, "root", part)
                            if hasattr(p_root, "text") and p_root.text:
                                parts_text.append(p_root.text)
                        return "".join(parts_text)

        except Exception as e:
            logger.debug(f"스트리밍 텍스트 추출 오류: {e}")
        return ""

    @staticmethod
    def _extract_state(event: SendStreamingMessageResponse) -> str:
        """스트리밍 이벤트에서 태스크 상태를 추출한다."""
        try:
            root = event.root
            if hasattr(root, "result"):
                result = root.result
                if hasattr(result, "status"):
                    status = result.status
                    if hasattr(status, "state"):
                        return (
                            str(status.state.value)
                            if hasattr(status.state, "value")
                            else str(status.state)
                        )
        except Exception:
            pass
        return "working"

    @staticmethod
    def _extract_final_text(event: SendStreamingMessageResponse) -> str | None:
        """아티팩트 이벤트에서 최종 텍스트를 추출한다."""
        try:
            root = event.root
            if not hasattr(root, "result"):
                return None
            result = root.result

            # TaskArtifactUpdateEvent — 아티팩트에서 최종 텍스트 추출
            if hasattr(result, "artifact"):
                artifact = result.artifact
                if hasattr(artifact, "parts") and artifact.parts:
                    texts = []
                    for part in artifact.parts:
                        p_root = getattr(part, "root", part)
                        if hasattr(p_root, "text") and p_root.text:
                            texts.append(p_root.text)
                    if texts:
                        return "".join(texts)

        except Exception:
            pass
        return None


async def stream_agent_response(
    url: str,
    content: str,
    *,
    timeout: float = 300.0,
    on_chunk: Any | None = None,
) -> str:
    """에이전트에 스트리밍 요청을 보내고 최종 결과를 반환하는 헬퍼.

    Args:
        url: 에이전트 URL
        content: 전송할 메시지
        timeout: 타임아웃 (초)
        on_chunk: 각 청크 수신 시 호출할 콜백

    Returns:
        최종 응답 텍스트
    """
    collector = StreamingResponseCollector(
        timeout=timeout,
        chunk_callback=on_chunk,
    )

    async for _chunk in collector.collect_stream(url, content):
        pass  # 콜백이 이미 호출됨

    return collector.get_final_text()
