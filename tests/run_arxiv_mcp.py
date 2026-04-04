"""arXiv 논문 검색 MCP 서버.

arXiv API를 사용하여 학술 논문을 검색한다. API 키 불필요.

실행: python -m youngs75_a2a.tests.run_arxiv_mcp
포트: 3000
"""

import sys

sys.path.insert(0, ".")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="ArxivSearchServer",
    host="0.0.0.0",
    port=3000,
)


@mcp.tool()
def search_papers(query: str, max_results: int = 5) -> str:
    """arXiv에서 학술 논문을 검색합니다.

    Args:
        query: 검색 쿼리 (영어 권장)
        max_results: 최대 결과 수 (기본 5)
    """
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    output = []
    for result in client.results(search):
        output.append(f"### {result.title}")
        output.append(f"Authors: {', '.join(a.name for a in result.authors[:3])}")
        output.append(f"Published: {result.published.strftime('%Y-%m-%d')}")
        output.append(f"URL: {result.entry_id}")
        summary = result.summary.replace("\n", " ")[:300]
        output.append(f"Abstract: {summary}...")
        output.append("")

    return "\n".join(output) if output else "검색 결과가 없습니다."


@mcp.tool()
def search_recent_papers(query: str, max_results: int = 5) -> str:
    """arXiv에서 최신 논문을 날짜순으로 검색합니다.

    Args:
        query: 검색 쿼리 (영어 권장)
        max_results: 최대 결과 수 (기본 5)
    """
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    output = []
    for result in client.results(search):
        output.append(f"### {result.title}")
        output.append(f"Authors: {', '.join(a.name for a in result.authors[:3])}")
        output.append(f"Published: {result.published.strftime('%Y-%m-%d')}")
        output.append(f"URL: {result.entry_id}")
        categories = ", ".join(result.categories[:3])
        output.append(f"Categories: {categories}")
        summary = result.summary.replace("\n", " ")[:300]
        output.append(f"Abstract: {summary}...")
        output.append("")

    return "\n".join(output) if output else "검색 결과가 없습니다."


if __name__ == "__main__":
    print("arXiv MCP 서버 시작: http://0.0.0.0:3000/mcp")
    mcp.run(transport="streamable-http")
