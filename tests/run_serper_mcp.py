"""Serper Google 검색 MCP 서버.

Serper.dev API를 사용하여 Google 검색을 수행한다.
SERPER_API_KEY가 없으면 Tavily로 폴백한다.

실행: python -m youngs75_a2a.tests.run_serper_mcp
포트: 3002
"""

import os
import sys

sys.path.insert(0, ".")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="SerperSearchServer",
    host="0.0.0.0",
    port=3002,
)


def _search_with_serper(query: str, max_results: int = 5, search_type: str = "search") -> str:
    """Serper API로 검색."""
    import httpx

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return ""  # 빈 문자열 → 폴백 트리거

    resp = httpx.post(
        f"https://google.serper.dev/{search_type}",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=10.0,
    )
    resp.raise_for_status()
    data = resp.json()

    output = []
    for item in data.get("organic", [])[:max_results]:
        output.append(f"### {item.get('title', 'N/A')}")
        output.append(f"URL: {item.get('link', '')}")
        output.append(f"{item.get('snippet', '')}")
        output.append("")

    return "\n".join(output) if output else "검색 결과가 없습니다."


def _search_with_tavily_fallback(query: str, max_results: int = 5) -> str:
    """Tavily API로 폴백 검색."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY와 TAVILY_API_KEY 모두 설정되지 않았습니다."

    from tavily import TavilyClient

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
def google_search(query: str, max_results: int = 5) -> str:
    """Google에서 웹 검색을 수행합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5)
    """
    result = _search_with_serper(query, max_results)
    if not result:
        result = _search_with_tavily_fallback(query, max_results)
    return result


@mcp.tool()
def google_news(query: str, max_results: int = 5) -> str:
    """Google에서 최신 뉴스를 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5)
    """
    result = _search_with_serper(query, max_results, search_type="news")
    if not result:
        # 뉴스도 Tavily로 폴백
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            results = client.search(query=query, max_results=max_results, topic="news")
            output = []
            for r in results.get("results", []):
                output.append(f"### {r.get('title', 'N/A')}")
                output.append(f"URL: {r.get('url', '')}")
                output.append(f"{r.get('content', '')}")
                output.append("")
            result = "\n".join(output) if output else "뉴스 결과가 없습니다."
        else:
            result = "Error: SERPER_API_KEY와 TAVILY_API_KEY 모두 설정되지 않았습니다."
    return result


if __name__ == "__main__":
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    print(f"Serper MCP 서버 시작: http://0.0.0.0:3002/mcp")
    print(f"  Serper API: {'활성' if has_serper else '없음 → Tavily 폴백'}")
    mcp.run(transport="streamable-http")
