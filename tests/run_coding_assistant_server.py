"""CodingAssistantAgent A2A 서버.

코드 생성/검증 에이전트를 A2A 서버로 노출한다.
Langfuse 트레이싱이 활성화되면 모든 LLM 호출이 자동으로 기록된다.

실행: python -m youngs75_a2a.tests.run_coding_assistant_server
포트: 18084 (환경변수 AGENT_PORT로 변경 가능)
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
from typing import Any

import uvicorn
from starlette.routing import Route
from starlette.responses import JSONResponse

from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
from youngs75_a2a.agents.coding_assistant import CodingAssistantAgent, CodingConfig
from youngs75_a2a.eval_pipeline.observability.langfuse import enabled, enrich_trace


def extract_coding_result(result: dict[str, Any]) -> str:
    """CodingState에서 generated_code 또는 마지막 AI 메시지를 추출."""
    if result.get("generated_code"):
        return result["generated_code"]
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content
    return ""


class LangfuseLGAgentExecutor(LGAgentExecutor):
    """Langfuse 트레이싱을 자동 주입하는 LGAgentExecutor."""

    async def execute(self, context, event_queue):
        """요청마다 Langfuse 트레이스로 감싸서 실행한다."""
        task_id = getattr(context, "task_id", "unknown")
        with enrich_trace(
            user_id=f"a2a-client",
            session_id=f"coding-{task_id}",
            tags=["coding-assistant", "a2a"],
        ):
            return await super().execute(context, event_queue)


async def main():
    port = int(os.getenv("AGENT_PORT", "18084"))
    model = os.getenv("AGENT_MODEL", "gpt-5.4-mini")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 또는 OPENROUTER_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    config = CodingConfig(
        default_model=model,
        generation_model=model,
        verification_model=model,
    )
    agent = CodingAssistantAgent(config=config)

    langfuse_active = enabled()
    ExecutorClass = LangfuseLGAgentExecutor if langfuse_active else LGAgentExecutor
    executor = ExecutorClass(
        graph=agent.graph,
        result_extractor=extract_coding_result,
    )

    card = create_agent_card(
        name="youngs75-coding-assistant",
        description="코드 생성/검증 에이전트 — parse → execute → verify 3노드 구조",
        url=f"http://0.0.0.0:{port}",
    )
    server_app = build_app(executor, card)
    app = server_app.build()

    async def health(request):
        return JSONResponse({
            "status": "healthy",
            "agent": "coding-assistant",
            "model": model,
            "langfuse": langfuse_active,
            "nodes": ["parse_request", "execute_code", "verify_result"],
        })

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    print(f"🚀 CodingAssistant A2A 서버 시작")
    print(f"   포트: {port}, 모델: {model}, Langfuse: {'ON' if langfuse_active else 'OFF'}")

    uv_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(uv_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
