"""평가 대상 에이전트 — Coding Assistant Harness.

youngs75_a2a의 CodingAssistantAgent를 래핑하여
DeepEval 평가 파이프라인에서 사용할 수 있는 형태로 제공한다.

동작 모드:
    - CODING_AGENT_URL 환경변수 설정 시: A2A 서비스에 HTTP 요청 (Docker 환경)
    - 미설정 시: in-process로 직접 실행 (로컬 개발 환경)
      - CODE_TOOLS_MCP_URL 설정 시: MCP 도구 연동 (Harness 모드)
      - 미설정 시: MCP 없이 LLM만으로 실행 (Fallback 모드)
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from youngs75_a2a.eval_pipeline.observability.langfuse import build_langchain_config


def _run_via_a2a_service(query: str, agent_url: str) -> str:
    """A2A 서비스에 HTTP 요청으로 Coding Agent를 실행한다."""
    import httpx

    response = httpx.post(
        f"{agent_url}/",
        json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": "eval-1",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": query}],
                    "messageId": "eval-msg-1",
                },
            },
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    result = data.get("result", {})
    artifacts = result.get("artifacts", [])
    for artifact in artifacts:
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                return part["text"]

    return ""


async def _run_in_process_async(query: str) -> str:
    """CodingAssistantAgent를 in-process로 실행한다 (MCP 연동 포함)."""
    from youngs75_a2a.agents.coding_assistant import CodingAssistantAgent, CodingConfig

    config = CodingConfig()
    agent = await CodingAssistantAgent.create(config=config)

    invoke_input = {
        "messages": [HumanMessage(content=query)],
        "iteration": 0,
        "max_iterations": 2,
    }

    result = await agent.graph.ainvoke(invoke_input)
    return result.get("generated_code", "")


def _run_in_process(query: str) -> str:
    """동기 래퍼 — async 에이전트를 동기 컨텍스트에서 실행한다."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()

    return asyncio.run(_run_in_process_async(query))


def run_coding_agent(query: str) -> str:
    """Coding Agent를 실행하고 결과를 반환한다.

    DeepEval 평가에서 actual_output을 생성하기 위해 사용.
    CODING_AGENT_URL이 설정되면 A2A 서비스에 요청, 아니면 in-process 실행.
    """
    agent_url = os.getenv("CODING_AGENT_URL")
    if agent_url:
        return _run_via_a2a_service(query, agent_url)
    return _run_in_process(query)


if __name__ == "__main__":
    langfuse_handler = CallbackHandler()
    config = build_langchain_config(
        user_id="demo-user",
        session_id="coding-eval",
        callbacks=[langfuse_handler],
        tags=["coding-agent"],
    )

    response = run_coding_agent("파이썬으로 피보나치 함수를 작성해줘")
    print(response)

    get_client().flush()
