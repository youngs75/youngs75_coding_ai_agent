"""DeepResearchA2AAgent A2A 서버.

HITL 지원 + A2A supervisor 위임이 가능한 Deep Research 에이전트.

실행: python -m youngs75_a2a.tests.run_deep_research_a2a_server
포트: 18083 (환경변수 AGENT_PORT로 변경 가능)
"""

import os
import sys

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass
        break

import uvicorn
from starlette.routing import Route
from starlette.responses import JSONResponse

from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
from youngs75_a2a.agents.deep_research import DeepResearchA2AAgent, ResearchConfig


def main():
    port = int(os.getenv("AGENT_PORT", "18083"))
    model = os.getenv("AGENT_MODEL", "gpt-5.4-mini")
    tavily_url = os.getenv("TAVILY_MCP_URL", "http://localhost:3001/mcp/")
    arxiv_url = os.getenv("ARXIV_MCP_URL", "http://localhost:3000/mcp/")
    serper_url = os.getenv("SERPER_MCP_URL", "http://localhost:3002/mcp/")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    rc = ResearchConfig(
        allow_clarification=False,
        enable_hitl=os.getenv("ENABLE_HITL", "false").lower() == "true",
        default_model=model,
        research_model=model,
        compression_model=model,
        final_report_model=model,
        max_researcher_iterations=2,
        max_concurrent_research_units=3,
        mcp_servers={
            "tavily": tavily_url,
            "arxiv": arxiv_url,
            "serper": serper_url,
        },
    )
    agent = DeepResearchA2AAgent(config=rc, use_a2a_supervisor=False)

    def extract_result(result: dict) -> str:
        if not isinstance(result, dict):
            return ""
        for key in ("final_report", "research_brief"):
            if key in result and result[key]:
                return str(result[key])
        for node_output in result.values():
            if isinstance(node_output, dict):
                for key in ("final_report", "research_brief"):
                    if key in node_output and node_output[key]:
                        return str(node_output[key])
        return ""

    executor = LGAgentExecutor(graph=agent.graph, result_extractor=extract_result)
    card = create_agent_card(
        name="youngs75-deep-research-a2a",
        description="Deep Research A2A Agent — HITL 지원, 3종 MCP 도구 활용",
        url=f"http://0.0.0.0:{port}",
    )
    server_app = build_app(executor, card)
    app = server_app.build()

    async def health(request):
        return JSONResponse({
            "status": "healthy",
            "agent": "deep-research-a2a",
            "model": model,
            "hitl": rc.enable_hitl,
        })

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    print(f"🚀 DeepResearchA2A Agent 서버 시작")
    print(f"   포트: {port}, 모델: {model}, HITL: {rc.enable_hitl}")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
