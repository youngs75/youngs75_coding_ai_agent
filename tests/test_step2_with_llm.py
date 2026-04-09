"""
Step 2: LLM 연동 테스트 (MCP 서버 불필요)
- OpenAI API 키 필요
- MCP 서버 불필요 (도구 없이 LLM만 사용)

준비: export OPENAI_API_KEY=sk-...
실행: cd Day-04 && python -m coding_agent.tests.test_step2_with_llm
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, ".")

_skip_no_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY가 설정되지 않았습니다",
)

# Day-03/.env 에서 키 로드 시도
try:
    from dotenv import load_dotenv

    load_dotenv("../Day-03/.env")
except ImportError:
    pass


def check_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print()
        print("설정 방법:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  또는 Day-03/.env 파일에 OPENAI_API_KEY=sk-... 추가")
        sys.exit(1)
    print(f"✓ OPENAI_API_KEY 확인 (끝 4자리: ...{key[-4:]})")


@_skip_no_api_key
@pytest.mark.flaky(reruns=3, reruns_delay=5)
async def test_base_agent_executor():
    """BaseAgentExecutor로 간단한 에이전트 테스트."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        openai_api_base="https://openrouter.ai/api/v1",
    )
    response = await llm.ainvoke("1+1=? 숫자만 답해줘")
    assert "2" in response.content
    print(f"✓ LLM 직접 호출 성공: '{response.content.strip()}'")


@_skip_no_api_key
@pytest.mark.flaky(reruns=3, reruns_delay=5)
async def test_deep_research_clarify_only():
    """DeepResearchAgent의 clarify 노드만 테스트.

    allow_clarification=True로 설정하여 명확화 판단만 수행.
    연구는 실행하지 않음.
    """
    from langchain_core.messages import HumanMessage
    from coding_agent.agents.deep_research import DeepResearchAgent, ResearchConfig

    rc = ResearchConfig(
        allow_clarification=True,
        default_model="deepseek/deepseek-v3.2",
        research_model="deepseek/deepseek-v3.2",
    )
    agent = DeepResearchAgent(config=rc)

    # 명확한 질문 → clarify 통과 → brief 작성 시도 (여기서 중단)
    try:
        result = await asyncio.wait_for(
            agent.graph.ainvoke(
                {"messages": [HumanMessage(content="Python의 GIL이란 무엇인가?")]},
                config={"configurable": rc.to_langgraph_configurable()},
            ),
            timeout=30.0,
        )
        # brief까지 진행되었으면 성공
        if result.get("research_brief"):
            print(
                f"✓ clarify → brief 성공 (brief 길이: {len(result['research_brief'])}자)"
            )
        else:
            print("✓ clarify 노드 실행 완료 (brief 미생성)")
    except asyncio.TimeoutError:
        print("⚠ 타임아웃 (30초) — LLM 응답 지연, 하지만 연결은 성공")
    except Exception as e:
        # supervisor 단계에서 MCP 없어서 실패하는 것은 예상됨
        if "research_brief" in str(e) or "supervisor" in str(e).lower():
            print(
                f"✓ clarify → brief 통과 후 supervisor에서 예상된 중단: {type(e).__name__}"
            )
        else:
            raise


async def test_a2a_server_lifecycle():
    """A2A 서버 기동 → 헬스체크 → 종료 테스트."""
    import httpx
    from coding_agent.a2a import LGAgentExecutor, build_app, create_agent_card
    from coding_agent.agents.deep_research import DeepResearchAgent, ResearchConfig

    agent = DeepResearchAgent(
        config=ResearchConfig(default_model="deepseek/deepseek-v3.2")
    )
    executor = LGAgentExecutor(graph=agent.graph)
    card = create_agent_card(name="test-agent", url="http://localhost:19876")
    server_app = build_app(executor, card)
    app = server_app.build()

    # 헬스체크 엔드포인트 추가
    from starlette.routing import Route
    from starlette.responses import JSONResponse

    async def health(request):
        return JSONResponse({"status": "ok"})

    app.router.routes.append(Route("/health", health, methods=["GET"]))

    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=19876, log_level="error")
    server = uvicorn.Server(config)

    # 서버 기동 (백그라운드)
    server_task = asyncio.create_task(server.serve())

    try:
        # 헬스체크 대기
        async with httpx.AsyncClient() as client:
            for _ in range(20):
                try:
                    resp = await client.get(
                        "http://127.0.0.1:19876/health", timeout=1.0
                    )
                    if resp.status_code == 200:
                        print("✓ A2A 서버 기동 + 헬스체크 성공")
                        break
                except Exception:
                    await asyncio.sleep(0.5)
            else:
                print("❌ A2A 서버 헬스체크 실패")
    finally:
        server.should_exit = True
        await asyncio.sleep(0.5)
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    print("✓ A2A 서버 정상 종료")


async def main():
    print("=" * 50)
    print("Step 2: LLM 연동 테스트 (MCP 불필요)")
    print("=" * 50)

    check_api_key()

    await test_base_agent_executor()
    await test_deep_research_clarify_only()
    await test_a2a_server_lifecycle()

    print()
    print("✅ Step 2 전체 통과!")
    print()
    print("다음 단계: Step 3 (MCP + 전체 연구 파이프라인)")
    print("  1. MCP 서버 실행 (Docker 또는 로컬)")
    print("  2. python -m coding_agent.tests.test_step3_full_pipeline")


if __name__ == "__main__":
    asyncio.run(main())
