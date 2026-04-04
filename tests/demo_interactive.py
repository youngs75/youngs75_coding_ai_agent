"""
youngs75_a2a 대화형 데모

A2A 서버를 기동하고, A2A SDK 클라이언트로 실시간 대화할 수 있는 인터랙티브 데모.

실행 방법:
  터미널 1 (MCP 서버):
    export $(grep -v '^#' .env | xargs)
    python -m youngs75_a2a.tests.run_tavily_mcp

  터미널 2 (대화형 데모):
    export $(grep -v '^#' .env | xargs)
    python -m youngs75_a2a.tests.demo_interactive

  또는 MCP 없이 (LLM만으로 테스트):
    export $(grep -v '^#' .env | xargs)
    python -m youngs75_a2a.tests.demo_interactive --no-mcp
"""

import argparse
import asyncio
import sys
import uuid

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass

import httpx
import uvicorn
from starlette.routing import Route
from starlette.responses import JSONResponse

from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import MessageSendParams, SendMessageRequest

from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
from youngs75_a2a.agents.deep_research import DeepResearchAgent, ResearchConfig


# ─── 서버 기동 ───────────────────────────────────────────────


async def start_a2a_server(agent_graph, port: int = 18080):
    """A2A 에이전트 서버를 백그라운드로 기동한다."""
    executor = LGAgentExecutor(graph=agent_graph)
    card = create_agent_card(
        name="youngs75-research-agent",
        description="Deep Research Agent (youngs75_a2a)",
        url=f"http://localhost:{port}",
    )
    server_app = build_app(executor, card)
    app = server_app.build()

    async def health(request):
        return JSONResponse({"status": "healthy", "agent": "youngs75-research-agent"})

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    return server


async def wait_for_server(port: int, timeout: int = 15):
    """서버가 준비될 때까지 대기."""
    async with httpx.AsyncClient() as client:
        for i in range(timeout * 2):
            try:
                resp = await client.get(f"http://localhost:{port}/health", timeout=1.0)
                if resp.status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(0.5)
    return False


# ─── A2A 클라이언트 대화 ──────────────────────────────────────


async def send_query(client: A2AClient, query: str) -> str:
    """A2A 클라이언트로 질의를 보내고 응답을 받는다."""
    message = create_text_message_object(content=query)
    request = SendMessageRequest(
        id=str(uuid.uuid4()),
        params=MessageSendParams(message=message),
    )

    print("\n⏳ 에이전트 처리 중...")

    response = await client.send_message(request)

    # 응답에서 텍스트 추출
    result = response.root
    if hasattr(result, "result"):
        task_or_msg = result.result
        # Task인 경우
        if hasattr(task_or_msg, "artifacts"):
            artifacts = task_or_msg.artifacts or []
            texts = []
            for artifact in artifacts:
                for part in artifact.parts or []:
                    root = getattr(part, "root", part)
                    if hasattr(root, "text"):
                        texts.append(root.text)
                    elif hasattr(root, "data"):
                        import json

                        texts.append(
                            json.dumps(root.data, ensure_ascii=False, indent=2)
                        )
            return "\n".join(texts) if texts else str(task_or_msg)

        # Message인 경우
        if hasattr(task_or_msg, "parts"):
            texts = []
            for part in task_or_msg.parts or []:
                root = getattr(part, "root", part)
                if hasattr(root, "text"):
                    texts.append(root.text)
            return "\n".join(texts) if texts else str(task_or_msg)

        return str(task_or_msg)

    elif hasattr(result, "error"):
        return f"❌ 에러: {result.error}"

    return str(result)


async def interactive_loop(client: A2AClient, port: int = 18080):
    """대화형 입력 루프."""
    print("\n" + "=" * 60)
    print("  youngs75_a2a 대화형 데모")
    print("  A2A 프로토콜로 에이전트와 대화합니다")
    print("=" * 60)
    print()
    print("  명령어:")
    print("    /quit    — 종료")
    print("    /card    — AgentCard 조회")
    print("    /health  — 헬스 체크")
    print()

    async with httpx.AsyncClient() as http:
        while True:
            try:
                query = input("🧑 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break

            if not query:
                continue

            if query == "/quit":
                print("종료합니다.")
                break

            if query == "/card":
                try:
                    resp = await http.get(
                        f"http://localhost:{port}/.well-known/agent-card.json"
                    )
                    card_data = resp.json()
                    print("\n📋 AgentCard:")
                    print(f"  Name: {card_data.get('name')}")
                    print(f"  Description: {card_data.get('description')}")
                    print(f"  URL: {card_data.get('url')}")
                    print(
                        f"  Streaming: {card_data.get('capabilities', {}).get('streaming')}"
                    )
                except Exception as e:
                    print(f"  ❌ {e}")
                continue

            if query == "/health":
                try:
                    resp = await http.get(
                        f"{client._url or 'http://localhost:18080'}/health"
                    )
                    print(f"\n💚 {resp.json()}")
                except Exception as e:
                    print(f"  ❌ {e}")
                continue

            # A2A 프로토콜로 질의
            try:
                response = await send_query(client, query)
                print(f"\n🤖 Agent:\n{response}\n")
            except Exception as e:
                print(f"\n❌ 에러: {e}\n")


# ─── 메인 ──────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="youngs75_a2a 대화형 데모")
    parser.add_argument("--port", type=int, default=18080, help="A2A 서버 포트")
    parser.add_argument("--no-mcp", action="store_true", help="MCP 서버 없이 실행")
    parser.add_argument("--model", default="deepseek/deepseek-v3.2", help="LLM 모델")
    args = parser.parse_args()

    # 1. 에이전트 생성
    print("📦 에이전트 생성 중...")
    rc = ResearchConfig(
        allow_clarification=False,
        default_model=args.model,
        research_model=args.model,
        compression_model=args.model,
        final_report_model=args.model,
        max_researcher_iterations=2,
        max_concurrent_research_units=2,
    )
    agent = DeepResearchAgent(config=rc)
    print(f"  ✓ DeepResearchAgent (model={args.model})")

    # 2. A2A 서버 기동
    print(f"\n🚀 A2A 서버 기동 중 (port={args.port})...")
    server = await start_a2a_server(agent.graph, port=args.port)
    server_task = asyncio.create_task(server.serve())

    try:
        ready = await wait_for_server(args.port)
        if not ready:
            print("  ❌ 서버 시작 실패")
            return
        print(f"  ✓ A2A 서버 기동 완료: http://localhost:{args.port}")
        print(f"  ✓ AgentCard: http://localhost:{args.port}/.well-known/agent.json")

        if args.no_mcp:
            print("\n  ⚠ --no-mcp: MCP 도구 없이 LLM만 사용합니다")

        # 3. A2A 클라이언트 연결
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http_client:
            client = A2AClient(
                httpx_client=http_client,
                url=f"http://localhost:{args.port}",
            )
            print("  ✓ A2A 클라이언트 연결")

            # 4. 대화형 루프
            await interactive_loop(client, port=args.port)

    finally:
        print("\n🔻 서버 종료 중...")
        server.should_exit = True
        await asyncio.sleep(0.5)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
        print("  ✓ 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())
