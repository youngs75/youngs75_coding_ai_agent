"""SimpleMCPReActAgent A2A 서버.

MCP 도구를 사용하는 ReAct 에이전트를 A2A 서버로 노출한다.

실행: python -m youngs75_a2a.tests.run_simple_react_server
포트: 18081 (환경변수 AGENT_PORT로 변경 가능)
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

import asyncio
import uvicorn
from starlette.routing import Route
from starlette.responses import JSONResponse

from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
from youngs75_a2a.agents.simple_react import SimpleMCPReActAgent, SimpleReActConfig


async def main():
    port = int(os.getenv("AGENT_PORT", "18081"))
    model = os.getenv("AGENT_MODEL", "gpt-5.4-mini")
    tavily_url = os.getenv("TAVILY_MCP_URL", "http://localhost:3001/mcp/")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # 비동기 초기화 (MCP 도구 로딩)
    config = SimpleReActConfig(
        default_model=model,
        mcp_servers={"tavily": tavily_url},
    )
    agent = await SimpleMCPReActAgent.create(config=config)
    tool_count = len(agent._tools)

    # A2A 서버 조립
    executor = LGAgentExecutor(graph=agent.graph)
    card = create_agent_card(
        name="youngs75-simple-react",
        description="Simple MCP ReAct Agent — Tavily 검색 도구 활용",
        url=f"http://0.0.0.0:{port}",
    )
    server_app = build_app(executor, card)
    app = server_app.build()

    async def health(request):
        return JSONResponse({
            "status": "healthy",
            "agent": "simple-react",
            "model": model,
            "tools_loaded": tool_count,
        })

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    print(f"🚀 SimpleMCPReAct A2A 서버 시작")
    print(f"   포트: {port}, 모델: {model}, 도구: {tool_count}개")

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
