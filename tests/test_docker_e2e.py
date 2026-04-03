"""
Docker 배포 후 E2E 테스트 — 6개 컨테이너 전체 검증

MCP 3개 + Agent 3개가 Docker로 떠있는 상태에서 실행.

실행:
  cd youngs75_a2a/docker && docker compose up -d
  export $(grep -v '^#' .env | xargs)
  python -m youngs75_a2a.tests.test_docker_e2e
"""

import asyncio
import sys
import uuid

sys.path.insert(0, ".")

import httpx
from a2a.client import A2AClient
from a2a.client.helpers import create_text_message_object
from a2a.types import MessageSendParams, SendMessageRequest


# ─── 서비스 목록 ──────────────────────────────────────

MCP_SERVICES = {
    "Tavily":  "http://localhost:3001",
    "arXiv":   "http://localhost:3000",
    "Serper":  "http://localhost:3002",
}

AGENT_SERVICES = {
    "SimpleReAct":       "http://localhost:18081",
    "DeepResearch":      "http://localhost:18082",
    "DeepResearchA2A":   "http://localhost:18083",
}


# ─── 유틸리티 ─────────────────────────────────────────

async def check_health(name: str, url: str) -> bool:
    async with httpx.AsyncClient(timeout=5.0) as client:
        # /health 시도
        try:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                info = resp.json()
                print(f"  ✓ {name:20s} — {info}")
                return True
        except Exception:
            pass
        # MCP 서버: /mcp 직접 접속
        try:
            resp = await client.get(f"{url}/mcp")
            print(f"  ✓ {name:20s} — 접속 가능 (status={resp.status_code})")
            return True
        except Exception as e:
            print(f"  ❌ {name:20s} — 접근 불가: {e}")
            return False


async def query_agent(url: str, query: str) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as hc:
        client = A2AClient(httpx_client=hc, url=url)
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
                for artifact in obj.artifacts:
                    for part in (artifact.parts or []):
                        root = getattr(part, "root", part)
                        if hasattr(root, "text") and root.text and len(root.text) > 20:
                            return root.text
            if hasattr(obj, "status"):
                return f"[Task 상태: {obj.status}]"
        return "[응답 파싱 실패]"


async def test_agent_card(url: str, name: str) -> bool:
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{url}/.well-known/agent-card.json")
        if resp.status_code == 200:
            card = resp.json()
            print(f"  ✓ {name:20s} — name={card.get('name')}, streaming={card.get('capabilities', {}).get('streaming')}")
            return True
        print(f"  ❌ {name:20s} — AgentCard 조회 실패")
        return False


# ─── 메인 ─────────────────────────────────────────────

async def main():
    print("=" * 65)
    print("  youngs75_a2a Docker E2E 테스트 — 6개 컨테이너")
    print("=" * 65)

    # 1. MCP 서비스 헬스 체크
    print("\n1️⃣  MCP 서버 헬스 체크")
    mcp_ok = {}
    for name, url in MCP_SERVICES.items():
        mcp_ok[name] = await check_health(name, url)

    # 2. Agent 서비스 헬스 체크
    print("\n2️⃣  Agent 서버 헬스 체크")
    agent_ok = {}
    for name, url in AGENT_SERVICES.items():
        agent_ok[name] = await check_health(name, url)

    if not any(agent_ok.values()):
        print("\n❌ 실행 중인 Agent 서버가 없습니다.")
        print("  cd youngs75_a2a/docker && docker compose up -d")
        sys.exit(1)

    # 3. AgentCard 조회
    print("\n3️⃣  AgentCard 조회")
    for name, url in AGENT_SERVICES.items():
        if agent_ok.get(name):
            await test_agent_card(url, name)

    # 4. 질의 테스트
    print("\n4️⃣  A2A 프로토콜 질의 테스트")

    test_cases = [
        ("SimpleReAct", "http://localhost:18081", "오늘 AI 관련 뉴스 1개만 알려줘"),
        ("DeepResearch", "http://localhost:18082", "Python의 GIL이란 무엇인지 간단히 설명해줘"),
        ("DeepResearchA2A", "http://localhost:18083", "2026년 LLM 에이전트 동향을 간략히 조사해줘"),
    ]

    for name, url, query in test_cases:
        if not agent_ok.get(name):
            print(f"\n  ⏭️  {name} — 서버 미실행, 건너뜀")
            continue

        print(f"\n  🧑 [{name}] {query}")
        print(f"  ⏳ 처리 중...")
        try:
            response = await query_agent(url, query)
            preview = response[:200].replace("\n", " ")
            print(f"  🤖 응답 ({len(response)}자): {preview}...")
        except Exception as e:
            print(f"  ❌ 에러: {e}")

    # 5. 요약
    print("\n" + "=" * 65)
    total = len(MCP_SERVICES) + len(AGENT_SERVICES)
    ok_count = sum(mcp_ok.values()) + sum(agent_ok.values())
    print(f"  결과: {ok_count}/{total} 서비스 정상")
    for name, ok in {**mcp_ok, **agent_ok}.items():
        status = "✓" if ok else "❌"
        print(f"    {status} {name}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
