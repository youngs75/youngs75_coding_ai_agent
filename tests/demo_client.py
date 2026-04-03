"""
A2A 대화형 클라이언트 — Docker로 떠있는 Agent에 연결

이미 docker compose up 으로 서버가 실행 중인 상태에서 사용.

실행:
  export $(grep -v '^#' .env | xargs)

  # SimpleReAct (빠른 검색 답변)
  python -m youngs75_a2a.tests.demo_client --agent simple

  # DeepResearch (심층 연구 보고서)
  python -m youngs75_a2a.tests.demo_client --agent deep

  # DeepResearchA2A (A2A + HITL)
  python -m youngs75_a2a.tests.demo_client --agent a2a
"""

import argparse
import asyncio
import sys
import uuid

sys.path.insert(0, ".")

import httpx
from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import MessageSendParams, SendMessageRequest


AGENTS = {
    "simple": {"url": "http://localhost:18081", "name": "SimpleReAct"},
    "deep":   {"url": "http://localhost:18082", "name": "DeepResearch"},
    "a2a":    {"url": "http://localhost:18083", "name": "DeepResearchA2A"},
}


async def send_query(client: A2AClient, query: str) -> str:
    msg = create_text_message_object(content=query)
    request = SendMessageRequest(
        id=str(uuid.uuid4()),
        params=MessageSendParams(message=msg),
    )
    response = await client.send_message(request)

    result = response.root
    if hasattr(result, "result"):
        obj = result.result
        if hasattr(obj, "artifacts") and obj.artifacts:
            texts = []
            for artifact in obj.artifacts:
                for part in (artifact.parts or []):
                    root = getattr(part, "root", part)
                    if hasattr(root, "text") and root.text and len(root.text) > 10:
                        texts.append(root.text)
            if texts:
                return "\n\n".join(texts)
        if hasattr(obj, "status"):
            return f"[Task 상태: {obj.status}]"
    return "[응답 없음]"


async def main():
    parser = argparse.ArgumentParser(description="A2A 대화형 클라이언트")
    parser.add_argument(
        "--agent", choices=["simple", "deep", "a2a"], default="simple",
        help="연결할 에이전트: simple(18081), deep(18082), a2a(18083)",
    )
    args = parser.parse_args()

    agent_info = AGENTS[args.agent]
    url = agent_info["url"]
    name = agent_info["name"]

    # 헬스 체크
    async with httpx.AsyncClient(timeout=5.0) as hc:
        try:
            resp = await hc.get(f"{url}/health")
            if resp.status_code != 200:
                raise Exception(f"status={resp.status_code}")
            health = resp.json()
            print(f"✓ {name} 연결 성공: {health}")
        except Exception as e:
            print(f"❌ {name}({url})에 연결할 수 없습니다: {e}")
            print(f"   docker compose up -d 를 먼저 실행하세요.")
            sys.exit(1)

    print()
    print(f"{'=' * 60}")
    print(f"  A2A 대화형 클라이언트 — {name}")
    print(f"  서버: {url}")
    print(f"{'=' * 60}")
    print()
    print("  명령어:")
    print("    /quit      — 종료")
    print("    /switch N  — 에이전트 전환 (simple, deep, a2a)")
    print("    /health    — 헬스 체크")
    print("    /card      — AgentCard 조회")
    print()

    current_url = url
    current_name = name

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as hc:
        client = A2AClient(httpx_client=hc, url=current_url)

        while True:
            try:
                query = input(f"🧑 [{current_name}] You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break

            if not query:
                continue

            if query == "/quit":
                print("종료합니다.")
                break

            if query.startswith("/switch"):
                parts = query.split()
                if len(parts) >= 2 and parts[1] in AGENTS:
                    agent_info = AGENTS[parts[1]]
                    current_url = agent_info["url"]
                    current_name = agent_info["name"]
                    client = A2AClient(httpx_client=hc, url=current_url)
                    print(f"  → {current_name}({current_url})로 전환\n")
                else:
                    print(f"  사용법: /switch simple|deep|a2a\n")
                continue

            if query == "/health":
                try:
                    resp = await hc.get(f"{current_url}/health")
                    print(f"  💚 {resp.json()}\n")
                except Exception as e:
                    print(f"  ❌ {e}\n")
                continue

            if query == "/card":
                try:
                    resp = await hc.get(f"{current_url}/.well-known/agent-card.json")
                    card = resp.json()
                    print(f"  📋 {card.get('name')} — {card.get('description')}")
                    print(f"     streaming={card.get('capabilities', {}).get('streaming')}\n")
                except Exception as e:
                    print(f"  ❌ {e}\n")
                continue

            # A2A 질의
            print(f"  ⏳ 처리 중...")
            try:
                response = await send_query(client, query)
                print(f"\n🤖 [{current_name}] Agent:\n{response}\n")
            except Exception as e:
                print(f"\n  ❌ 에러: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
