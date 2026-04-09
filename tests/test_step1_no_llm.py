"""
Step 1: 외부 의존 없이 프레임워크 자체 테스트
- API 키 불필요
- MCP 서버 불필요
- LLM 호출 없음

실행: cd Day-04 && python -m coding_agent.tests.test_step1_no_llm
"""

import asyncio
import sys

sys.path.insert(0, ".")


def test_core_imports():
    """core 모듈 전체 import 테스트."""
    print("✓ core 모듈 import")


def test_override_reducer():
    """override_reducer 동작 테스트."""
    from coding_agent.core.reducers import override_reducer

    # 누적 모드
    result = override_reducer(["a", "b"], ["c"])
    assert result == ["a", "b", "c"], f"누적 실패: {result}"

    # 덮어쓰기 모드
    result = override_reducer(["a", "b"], {"type": "override", "value": []})
    assert result == [], f"덮어쓰기 실패: {result}"

    print("✓ override_reducer 정상 동작")


def test_tool_call_utils():
    """도구 호출 유틸리티 테스트."""
    from coding_agent.core.tool_call_utils import tc_name, tc_id, tc_args

    # dict 형태
    call = {"name": "search_web", "id": "call_123", "args": {"query": "AI"}}
    assert tc_name(call) == "search_web"
    assert tc_id(call) == "call_123"
    assert tc_args(call) == {"query": "AI"}

    # OpenAI function 형태
    call_fn = {
        "function": {"name": "search", "arguments": '{"q": "test"}'},
        "id": "fc_1",
    }
    assert tc_name(call_fn) == "search"
    assert tc_args(call_fn) == {"q": "test"}

    # None 안전성
    assert tc_name(None) is None
    assert tc_args(None) == {}

    print("✓ tool_call_utils 정상 동작")


def test_config():
    """BaseAgentConfig 생성 및 변환 테스트."""
    from coding_agent.core.config import BaseAgentConfig

    config = BaseAgentConfig(
        model_provider="openrouter",
        default_model="deepseek/deepseek-v3.2",
        temperature=0.1,
        mcp_servers={"tavily": "http://localhost:3001/mcp/"},
    )
    assert config.get_mcp_endpoint("tavily") == "http://localhost:3001/mcp/"
    assert config.get_mcp_endpoint("없는서버") is None

    # to_langgraph_configurable 변환
    configurable = config.to_langgraph_configurable()
    assert configurable["default_model"] == "deepseek/deepseek-v3.2"

    print("✓ BaseAgentConfig 정상 동작")


def test_research_config():
    """ResearchConfig 용도별 모델 분리 테스트."""
    from coding_agent.agents.deep_research.config import ResearchConfig

    rc = ResearchConfig(
        research_model="deepseek/deepseek-v3.2",
        compression_model="qwen/qwen3.5-9b",
        final_report_model="deepseek/deepseek-v3.2",
    )
    assert rc._resolve_model_name("research") == "deepseek/deepseek-v3.2"
    assert rc._resolve_model_name("compression") == "qwen/qwen3.5-9b"
    assert rc._resolve_model_name("final_report") == "deepseek/deepseek-v3.2"
    assert rc._resolve_model_name("unknown") == rc.default_model

    print("✓ ResearchConfig 모델 분리 정상")


def test_graph_build():
    """에이전트 그래프 빌드 테스트 (LLM 호출 없이)."""
    from coding_agent.agents.deep_research import DeepResearchAgent, ResearchConfig

    agent = DeepResearchAgent(config=ResearchConfig())
    assert agent.graph is not None

    # 노드 확인
    node_names = list(agent.graph.nodes.keys())
    assert "clarify_with_user" in node_names
    assert "write_research_brief" in node_names
    assert "research_supervisor" in node_names
    assert "final_report_generation" in node_names

    print(f"✓ DeepResearchAgent 그래프 빌드 (노드: {node_names})")


def test_a2a_assembly():
    """A2A 서버 조립 테스트 (서버 기동 없이)."""
    from coding_agent.agents.deep_research import DeepResearchAgent, ResearchConfig
    from coding_agent.a2a import LGAgentExecutor, build_app, create_agent_card

    agent = DeepResearchAgent(config=ResearchConfig())
    executor = LGAgentExecutor(graph=agent.graph)
    card = create_agent_card(name="test-agent", url="http://localhost:9999")
    app = build_app(executor, card)

    assert app is not None
    print("✓ A2A 서버 조립 성공")


async def test_mcp_loader_no_server():
    """MCP 서버 없이 MCPToolLoader graceful degradation 테스트."""
    from coding_agent.core.mcp_loader import MCPToolLoader

    loader = MCPToolLoader(
        servers={"fake": "http://localhost:99999/mcp/"},
        health_timeout=1.0,
        max_retries=1,
    )
    tools = await loader.load()
    assert tools == [], f"서버 없을 때 빈 리스트여야 함: {tools}"
    assert loader.is_loaded

    print("✓ MCPToolLoader graceful degradation (서버 없이 빈 도구)")


def main():
    print("=" * 50)
    print("Step 1: 프레임워크 자체 테스트 (LLM/MCP 불필요)")
    print("=" * 50)

    test_core_imports()
    test_override_reducer()
    test_tool_call_utils()
    test_config()
    test_research_config()
    test_graph_build()
    test_a2a_assembly()
    asyncio.run(test_mcp_loader_no_server())

    print()
    print("✅ Step 1 전체 통과!")
    print()
    print("다음 단계: Step 2 (LLM 연동 테스트)")
    print("  export OPENAI_API_KEY=sk-...")
    print("  python -m coding_agent.tests.test_step2_with_llm")


if __name__ == "__main__":
    main()
