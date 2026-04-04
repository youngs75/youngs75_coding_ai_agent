"""Tavily 검색 MCP 서버 (경량 로컬 버전).

Docker 없이 로컬에서 바로 실행 가능한 Tavily 웹 검색 MCP 서버.

실행: cd Day-04 && python -m youngs75_a2a.tests.run_tavily_mcp
포트: 3001 (docker-compose와 동일)
"""

import os
import sys

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="TavilySearchServer",
    host="0.0.0.0",
    port=3001,
)


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """웹에서 정보를 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5)
    """
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY가 설정되지 않았습니다."

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=max_results)

    output = []
    for r in results.get("results", []):
        output.append(f"### {r.get('title', 'N/A')}")
        output.append(f"URL: {r.get('url', '')}")
        output.append(f"{r.get('content', '')}")
        output.append("")

    return "\n".join(output) if output else "검색 결과가 없습니다."


@mcp.tool()
def search_news(query: str, max_results: int = 5) -> str:
    """최신 뉴스를 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5)
    """
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY가 설정되지 않았습니다."

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=max_results, topic="news")

    output = []
    for r in results.get("results", []):
        output.append(f"### {r.get('title', 'N/A')}")
        output.append(f"URL: {r.get('url', '')}")
        output.append(f"{r.get('content', '')}")
        output.append("")

    return "\n".join(output) if output else "뉴스 결과가 없습니다."


if __name__ == "__main__":
    print("Tavily MCP 서버 시작: http://0.0.0.0:3001/mcp")
    mcp.run(transport="streamable-http")
