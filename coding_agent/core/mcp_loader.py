"""MCP 도구 로딩 및 생명주기 관리.

모듈 레벨 글로벌 캐시 대신 인스턴스 기반으로 관리하여
이벤트 루프 간 안전성을 보장한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


class MCPToolLoader:
    """MCP 서버에서 도구를 로딩하고 캐싱하는 클래스.

    사용 예:
        loader = MCPToolLoader({
            "tavily": "http://localhost:3001/mcp/",
            "arxiv": "http://localhost:3000/mcp/",
        })
        tools = await loader.load()
    """

    def __init__(
        self,
        servers: dict[str, str],
        transport: str = "streamable_http",
        health_timeout: float = 3.0,
        max_retries: int = 2,
    ):
        self._servers = servers
        self._transport = transport
        self._health_timeout = health_timeout
        self._max_retries = max_retries
        self._tools: list[Any] | None = None
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._tools is not None

    async def load(self, *, force: bool = False) -> list[Any]:
        """MCP 도구를 로딩한다. 이미 로딩되었으면 캐시를 반환한다."""
        if self._tools is not None and not force:
            return self._tools

        async with self._lock:
            if self._tools is not None and not force:
                return self._tools

            available_servers = await self._check_healthy_servers()
            if not available_servers:
                logger.warning("접근 가능한 MCP 서버가 없습니다. 도구 없이 진행합니다.")
                self._tools = []
                return self._tools

            mcp_config = {
                name: {"url": url, "transport": self._transport}
                for name, url in available_servers.items()
            }

            for attempt in range(self._max_retries):
                try:
                    client = MultiServerMCPClient(mcp_config)
                    self._tools = await client.get_tools()
                    logger.info(
                        f"MCP 도구 {len(self._tools)}개 로딩 완료: "
                        f"{[getattr(t, 'name', str(t)) for t in self._tools]}"
                    )
                    return self._tools
                except Exception as e:
                    logger.warning(
                        f"MCP 도구 로딩 시도 {attempt + 1}/{self._max_retries} 실패: {e}"
                    )
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))

            logger.error("MCP 도구 로딩 최종 실패. 도구 없이 진행합니다.")
            self._tools = []
            return self._tools

    async def _check_healthy_servers(self) -> dict[str, str]:
        """각 MCP 서버의 접근 가능 여부를 확인하고 가능한 서버만 반환한다."""
        healthy: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=self._health_timeout) as client:
            for name, url in self._servers.items():
                # 1차: /health 엔드포인트 시도
                health_url = url.rstrip("/").rsplit("/mcp", 1)[0] + "/health"
                try:
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        healthy[name] = url
                        logger.debug(f"MCP 서버 '{name}' 정상 (health): {health_url}")
                        continue
                except Exception:
                    pass

                # 2차: MCP 엔드포인트 자체에 접속 가능한지 확인
                try:
                    resp = await client.get(url.rstrip("/"))
                    # 어떤 응답이든 오면 서버가 살아있는 것
                    healthy[name] = url
                    logger.debug(f"MCP 서버 '{name}' 정상 (직접 접속): {url}")
                except Exception as e:
                    logger.warning(f"MCP 서버 '{name}' 접근 불가: {e}")
        return healthy

    def get_tool_descriptions(self) -> str:
        """로딩된 도구의 설명을 포맷팅하여 반환한다."""
        if not self._tools:
            return "사용 가능한 도구가 없습니다."
        lines = []
        for tool in self._tools:
            name = getattr(tool, "name", "unknown")
            desc = getattr(tool, "description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def reset(self) -> None:
        """캐시를 초기화한다."""
        self._tools = None
