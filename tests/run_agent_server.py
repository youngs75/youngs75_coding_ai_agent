"""
youngs75_a2a A2A Agent 서버 (프로덕션 배포용)

A2A 프로토콜로 DeepResearch 에이전트를 서비스한다.
Docker 컨테이너 또는 로컬에서 실행 가능.

실행:
  로컬:   python -m youngs75_a2a.tests.run_agent_server
  Docker: docker compose up agent

환경변수:
  OPENAI_API_KEY   — 필수
  AGENT_PORT       — 서버 포트 (기본: 18080)
  AGENT_MODEL      — LLM 모델 (기본: gpt-5.4-mini)
  TAVILY_MCP_URL   — Tavily MCP 서버 URL (기본: http://localhost:3001/mcp/)
"""

import os
import sys

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv
    # 로컬 실행 시 .env 로드 (Docker에서는 env_file로 주입)
    load_dotenv(".env")
except ImportError:
    pass

import uvicorn
from starlette.routing import Route
from starlette.responses import JSONResponse

from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
from youngs75_a2a.agents.deep_research import DeepResearchAgent, ResearchConfig


def main():
    port = int(os.getenv("AGENT_PORT", "18082"))
    model = os.getenv("AGENT_MODEL", "gpt-5.4-mini")
    tavily_url = os.getenv("TAVILY_MCP_URL", "http://localhost:3001/mcp/")
    arxiv_url = os.getenv("ARXIV_MCP_URL", "http://localhost:3000/mcp/")
    serper_url = os.getenv("SERPER_MCP_URL", "http://localhost:3002/mcp/")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # 에이전트 생성
    rc = ResearchConfig(
        allow_clarification=False,
        default_model=model,
        research_model=model,
        compression_model=model,
        final_report_model=model,
        max_researcher_iterations=2,
        max_concurrent_research_units=2,
        mcp_servers={
            "tavily": tavily_url,
            "arxiv": arxiv_url,
            "serper": serper_url,
        },
    )
    agent = DeepResearchAgent(config=rc)

    # A2A 서버 조립 — astream 청크는 {노드명: {필드: 값}} 형태
    def extract_result(result: dict) -> str:
        if not isinstance(result, dict):
            return ""
        # 1단계: 최상위에서 직접 탐색
        for key in ("final_report", "research_brief"):
            if key in result and result[key]:
                return str(result[key])
        # 2단계: {노드명: {필드: 값}} 중첩 구조 탐색
        for node_output in result.values():
            if isinstance(node_output, dict):
                for key in ("final_report", "research_brief"):
                    if key in node_output and node_output[key]:
                        return str(node_output[key])
        return ""

    executor = LGAgentExecutor(graph=agent.graph, result_extractor=extract_result)
    card = create_agent_card(
        name="youngs75-research-agent",
        description="Deep Research Agent — youngs75_a2a production",
        url=f"http://0.0.0.0:{port}",
    )
    server_app = build_app(executor, card)
    app = server_app.build()

    # 헬스 체크
    async def health(request):
        return JSONResponse({
            "status": "healthy",
            "agent": "youngs75-research-agent",
            "model": model,
            "mcp": tavily_url,
        })

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    print(f"🚀 youngs75 A2A Agent 서버 시작")
    print(f"   포트: {port}")
    print(f"   모델: {model}")
    print(f"   MCP:  {tavily_url}")
    print(f"   헬스: http://0.0.0.0:{port}/health")
    print(f"   카드: http://0.0.0.0:{port}/.well-known/agent-card.json")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
