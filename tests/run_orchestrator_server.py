"""Orchestrator A2A 서버.

사용자 요청을 분석하여 적합한 하위 에이전트로 라우팅한다.

실행: python -m youngs75_a2a.tests.run_orchestrator_server
포트: 18080 (환경변수 ORCH_PORT로 변경 가능)

의존: 하위 에이전트 서버가 먼저 실행되어 있어야 한다.
"""

import os
import sys

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass

import asyncio
import uvicorn
from starlette.routing import Route
from starlette.responses import JSONResponse

from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
from youngs75_a2a.agents.orchestrator import OrchestratorAgent, OrchestratorConfig
from youngs75_a2a.agents.orchestrator.config import AgentEndpoint


async def main():
    port = int(os.getenv("ORCH_PORT", "18080"))
    model = os.getenv("AGENT_MODEL", "gpt-5.4-mini")

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # 하위 에이전트 엔드포인트 구성
    endpoints = [
        AgentEndpoint(
            name="simple-react",
            url=os.getenv("SIMPLE_REACT_URL", "http://localhost:18081"),
            description="간단한 검색, 뉴스 조회, 사실 확인 등 단일 도구로 해결 가능한 작업",
        ),
        AgentEndpoint(
            name="deep-research",
            url=os.getenv("DEEP_RESEARCH_URL", "http://localhost:18082"),
            description="심층 조사, 논문 분석, 기술 동향 파악 등 여러 소스를 종합하는 연구 작업",
        ),
        AgentEndpoint(
            name="deep-research-a2a",
            url=os.getenv("DEEP_RESEARCH_A2A_URL", "http://localhost:18083"),
            description="멀티에이전트 협업이 필요한 대규모 조사, 보고서 작성 및 사람 승인이 필요한 작업",
        ),
        AgentEndpoint(
            name="coding-assistant",
            url=os.getenv("CODING_ASSISTANT_URL", "http://localhost:18084"),
            description="코드 생성, 버그 수정, 리팩토링, 테스트 작성, 보안 검토 등 프로그래밍 관련 작업",
        ),
    ]

    config = OrchestratorConfig(
        default_model=model,
        agent_endpoints=endpoints,
    )
    agent = OrchestratorAgent(config=config)

    # 결과 추출기
    def extract_result(result: dict) -> str:
        if "agent_response" in result and result["agent_response"]:
            return str(result["agent_response"])
        # 노드 출력에서 탐색
        for node_output in result.values():
            if isinstance(node_output, dict) and "agent_response" in node_output:
                return str(node_output["agent_response"])
        return ""

    executor = LGAgentExecutor(graph=agent.graph, result_extractor=extract_result)
    card = create_agent_card(
        name="youngs75-orchestrator",
        description="사용자 요청을 분석하여 적합한 에이전트로 라우팅하는 오케스트레이터",
        url=f"http://0.0.0.0:{port}",
    )
    server_app = build_app(executor, card)
    app = server_app.build()

    async def health(request):
        return JSONResponse(
            {
                "status": "healthy",
                "agent": "orchestrator",
                "model": model,
                "registered_agents": [ep.name for ep in endpoints],
            }
        )

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    agent_list = ", ".join(ep.name for ep in endpoints)
    print("🚀 Orchestrator A2A 서버 시작")
    print(f"   포트: {port}, 모델: {model}")
    print(f"   등록 에이전트: {agent_list}")

    uvi_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(uvi_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
