"""
AgentExecutor를 A2A 서버로 조립하고 실행하는 모듈

조립 순서:
  에이전트 → AgentExecutor → DefaultRequestHandler → A2AStarletteApplication → Uvicorn

사용 예시:
    # 1) 일반 에이전트
    async def my_agent(query, ctx):
        return f"응답: {query}"

    run_server(BaseAgentExecutor(my_agent), name="echo-agent", port=8080)

    # 2) LangGraph 에이전트
    run_server(LGAgentExecutor(my_graph), name="lg-agent", port=8081)
"""

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.routing import Route
from starlette.responses import JSONResponse


def create_agent_card(
    name: str,
    description: str = "",
    url: str = "http://localhost:8080",
    skills: list[AgentSkill] | None = None,
    streaming: bool = True,
) -> AgentCard:
    """A2A AgentCard 생성."""
    return AgentCard(
        name=name,
        description=description,
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text/plain", "application/json"],
        capabilities=AgentCapabilities(streaming=streaming, push_notifications=False),
        skills=skills or [],
    )


def build_app(
    executor: AgentExecutor,
    agent_card: AgentCard,
) -> A2AStarletteApplication:
    """AgentExecutor → DefaultRequestHandler → A2AStarletteApplication 조립."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )


def run_server(
    executor: AgentExecutor,
    name: str = "my-agent",
    description: str = "",
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """AgentExecutor를 A2A 서버로 띄운다.

    이 함수 하나로 에이전트가 A2A 프로토콜로 노출된다.
    """
    agent_card = create_agent_card(
        name=name,
        description=description,
        url=f"http://{host}:{port}",
    )
    server_app = build_app(executor, agent_card)
    app = server_app.build()

    # 헬스체크 엔드포인트 추가
    async def health(request):
        return JSONResponse({"status": "healthy", "agent": name, "port": port})

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    print(f"A2A 서버 시작: http://{host}:{port} (에이전트: {name})")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


# --- 직접 실행 예시 ---
if __name__ == "__main__":
    from youngs75_a2a.a2a.executor import BaseAgentExecutor

    async def echo_agent(query: str, ctx: dict) -> str:
        return f"에코: {query}"

    run_server(
        executor=BaseAgentExecutor(agent_fn=echo_agent),
        name="echo-agent",
        description="입력을 그대로 돌려주는 에코 에이전트",
        port=8080,
    )
