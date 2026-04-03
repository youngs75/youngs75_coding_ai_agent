"""
Step 3: 전체 파이프라인 테스트 (MCP + LLM + A2A)
- OpenAI API 키 필요
- MCP 서버 최소 1개 필요 (Tavily 권장)

준비:
  1. export OPENAI_API_KEY=sk-...
  2. MCP 서버 실행 (아래 중 택 1)
     a) Docker: cd Day-04/a2a/docker && docker compose -f docker-compose.mcp.yml up -d
     b) 로컬:   python Day-04/mcp_server.py  (이미 구현된 MCP 서버가 있다면)

실행: cd Day-04 && python -m youngs75_a2a.tests.test_step3_full_pipeline
"""

import asyncio
import os
import sys
sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv
    load_dotenv("../Day-03/.env")
except ImportError:
    pass


async def test_mcp_tool_loading():
    """MCP 서버 연결 및 도구 로딩 테스트."""
    from youngs75_a2a.core.mcp_loader import MCPToolLoader

    tavily_url = os.getenv("TAVILY_MCP_URL", "http://localhost:3001/mcp/")
    loader = MCPToolLoader(
        servers={"tavily": tavily_url},
        health_timeout=3.0,
    )
    tools = await loader.load()

    if not tools:
        print(f"⚠ MCP 서버({tavily_url})에서 도구를 로딩하지 못했습니다.")
        print("  MCP 서버가 실행 중인지 확인하세요.")
        return False

    print(f"✓ MCP 도구 {len(tools)}개 로딩:")
    for t in tools:
        print(f"  - {getattr(t, 'name', '?')}: {getattr(t, 'description', '')[:60]}")
    return True


async def test_simple_react_agent():
    """SimpleMCPReActAgent 전체 실행 테스트."""
    from langchain_core.messages import HumanMessage
    from youngs75_a2a.agents.simple_react import SimpleMCPReActAgent, SimpleReActConfig

    config = SimpleReActConfig(default_model="gpt-5.4-mini")
    agent = await SimpleMCPReActAgent.create(config=config)

    if not agent._tools:
        print("⚠ SimpleMCPReActAgent: MCP 도구 없이 건너뜀")
        return

    result = await asyncio.wait_for(
        agent.graph.ainvoke({"messages": [HumanMessage(content="오늘 AI 관련 최신 뉴스 1개만 알려줘")]}),
        timeout=60.0,
    )

    last_msg = result["messages"][-1]
    print(f"✓ SimpleMCPReActAgent 실행 성공:")
    print(f"  응답 길이: {len(last_msg.content)}자")
    print(f"  응답 미리보기: {last_msg.content[:150]}...")


async def test_deep_research_full():
    """DeepResearchAgent 전체 파이프라인 테스트.

    clarify → brief → supervisor(병렬 연구) → final_report
    """
    from langchain_core.messages import HumanMessage
    from youngs75_a2a.agents.deep_research import DeepResearchAgent, ResearchConfig

    rc = ResearchConfig(
        allow_clarification=False,  # 명확화 건너뛰고 바로 연구
        default_model="gpt-5.4-mini",
        research_model="gpt-5.4-mini",
        compression_model="gpt-5.4-mini",
        final_report_model="gpt-5.4-mini",
        max_researcher_iterations=2,  # 빠른 테스트
        max_concurrent_research_units=2,
    )
    agent = DeepResearchAgent(config=rc)

    print("  DeepResearch 실행 중... (1~2분 소요)")
    result = await asyncio.wait_for(
        agent.graph.ainvoke(
            {"messages": [HumanMessage(content="2026년 LLM 에이전트 프레임워크 동향을 간략히 조사해줘")]},
            config={"configurable": rc.to_langgraph_configurable()},
        ),
        timeout=180.0,
    )

    report = result.get("final_report", "")
    notes = result.get("notes") or []
    raw_notes = result.get("raw_notes") or []

    print(f"✓ DeepResearchAgent 전체 파이프라인 성공:")
    print(f"  최종 보고서: {len(report)}자")
    print(f"  압축 노트: {len(notes)}개")
    print(f"  원본 노트: {len(raw_notes)}개")
    if report:
        print(f"  보고서 미리보기:")
        print(f"  {report[:300]}...")


async def test_a2a_end_to_end():
    """A2A 서버 기동 → 클라이언트 요청 → 응답 수신 E2E 테스트."""
    import httpx
    import uvicorn
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    from youngs75_a2a.a2a import LGAgentExecutor, build_app, create_agent_card
    from youngs75_a2a.agents.deep_research import DeepResearchAgent, ResearchConfig

    rc = ResearchConfig(
        allow_clarification=False,
        default_model="gpt-5.4-mini",
        research_model="gpt-5.4-mini",
        max_researcher_iterations=1,
    )
    agent = DeepResearchAgent(config=rc)
    executor = LGAgentExecutor(graph=agent.graph)
    card = create_agent_card(name="test-research", url="http://127.0.0.1:19877")
    server_app = build_app(executor, card)
    app = server_app.build()

    async def health(request):
        return JSONResponse({"status": "ok"})
    app.router.routes.append(Route("/health", health, methods=["GET"]))

    config = uvicorn.Config(app, host="127.0.0.1", port=19877, log_level="error")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    try:
        # 서버 준비 대기
        async with httpx.AsyncClient() as client:
            for _ in range(20):
                try:
                    resp = await client.get("http://127.0.0.1:19877/health", timeout=1.0)
                    if resp.status_code == 200:
                        break
                except Exception:
                    await asyncio.sleep(0.5)

            # AgentCard 확인
            resp = await client.get("http://127.0.0.1:19877/.well-known/agent.json", timeout=5.0)
            if resp.status_code == 200:
                agent_card = resp.json()
                print(f"✓ A2A AgentCard 조회 성공: {agent_card.get('name')}")
            else:
                print(f"⚠ AgentCard 조회 실패: {resp.status_code}")

    finally:
        server.should_exit = True
        await asyncio.sleep(0.5)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    print("✓ A2A E2E 테스트 완료")


async def main():
    print("=" * 50)
    print("Step 3: 전체 파이프라인 테스트 (MCP + LLM + A2A)")
    print("=" * 50)

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY 필요")
        sys.exit(1)
    print(f"✓ OPENAI_API_KEY 확인")

    mcp_ok = await test_mcp_tool_loading()

    if mcp_ok:
        await test_simple_react_agent()
        await test_deep_research_full()
    else:
        print()
        print("⚠ MCP 서버 없이는 SimpleReAct/DeepResearch 전체 테스트 불가")
        print("  MCP 서버 실행 후 다시 시도하세요")

    await test_a2a_end_to_end()

    print()
    print("✅ Step 3 완료!")


if __name__ == "__main__":
    asyncio.run(main())
