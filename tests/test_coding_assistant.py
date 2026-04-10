"""Coding Assistant 에이전트 테스트.

Step 1: 프레임워크 자체 검증 (LLM 호출 없이 그래프 구조 확인)
Step 2: LLM 연동 테스트 (API 키 필요)
"""

import asyncio
import json
import os
import sys

import pytest
from langchain_core.messages import HumanMessage

_skip_no_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY가 설정되지 않았습니다",
)


def test_step1_graph_structure():
    """그래프 노드/엣지 구조가 올바른지 확인한다."""
    from coding_agent.agents.coding_assistant import CodingAssistantAgent, CodingConfig

    config = CodingConfig()
    agent = CodingAssistantAgent(config=config)
    # CodingAssistantAgent는 MCP 비동기 로딩을 위해 auto_build=False.
    # 그래프 구조 검증만 하므로 수동으로 빌드한다.
    agent.build_graph()

    assert agent.graph is not None, "그래프가 빌드되지 않음"

    # 노드 확인 (단순화된 4-노드 그래프 v2)
    node_names = set(agent.graph.get_graph().nodes.keys())
    expected = {
        "retrieve_memory",
        "generate_code",
        "run_tests",
        "__start__",
        "__end__",
    }
    assert expected.issubset(node_names), f"누락된 노드: {expected - node_names}"

    print("[PASS] 그래프 구조 검증 통과")
    print(f"  노드: {node_names}")


def test_step1_safety_envelope():
    """ActionValidator가 위험 코드를 차단하는지 확인한다."""
    from coding_agent.core.action_validator import ActionValidator

    validator = ActionValidator()

    # 안전한 코드
    safe_code = 'def hello():\n    return "hello world"'
    report = validator.validate(safe_code)
    assert report.is_safe, f"안전한 코드가 차단됨: {report.summary()}"

    # 시크릿 노출
    secret_code = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"'
    report = validator.validate(secret_code)
    assert not report.is_safe, "시크릿 코드가 통과됨"
    assert "secret_exposure" in report.blocked_rules

    # 위험 명령
    danger_code = 'os.system("rm -rf /")'
    report = validator.validate(danger_code)
    assert not report.is_safe, "위험 명령이 통과됨"
    assert "dangerous_command" in report.blocked_rules

    # 파일 확장자 검증
    report = validator.validate(safe_code, target_files=["main.py"])
    assert report.is_safe

    report = validator.validate(safe_code, target_files=["virus.exe"])
    assert not report.is_safe
    assert "file_extension" in report.blocked_rules

    print("[PASS] Safety Envelope 검증 통과")


@_skip_no_api_key
async def test_step2_llm_integration():
    """LLM 연동 전체 파이프라인 테스트."""
    from coding_agent.agents.coding_assistant import CodingAssistantAgent, CodingConfig

    config = CodingConfig()
    agent = await CodingAssistantAgent.create(config=config)

    result = await agent.graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="파이썬으로 피보나치 수열의 n번째 값을 반환하는 함수를 작성해줘"
                )
            ],
            "iteration": 0,
            "max_iterations": 2,
        }
    )

    print("\n[결과]")
    print(
        f"  parse_result: {json.dumps(result.get('parse_result', {}), ensure_ascii=False, indent=2)}"
    )
    print(
        f"  verify_result: {json.dumps(result.get('verify_result', {}), ensure_ascii=False, indent=2)}"
    )
    print(f"  iteration: {result.get('iteration')}")
    print(f"  execution_log: {result.get('execution_log')}")
    print(f"\n  generated_code (앞 500자):\n{result.get('generated_code', '')[:500]}")
    print("\n[PASS] LLM 연동 테스트 통과")


##############################################################################
# Phase 12: ReAct 루프 강화 테스트
##############################################################################


class TestMaxToolCallsEnforcement:
    """max_tool_calls 한도 체크 테스트."""

    def _make_agent(self, max_tool_calls: int = 3):
        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        config = CodingConfig(max_tool_calls=max_tool_calls)
        agent = CodingAssistantAgent(config=config)
        agent.build_graph()
        return agent

    def test_tool_call_count_initialized_in_schema(self):
        """CodingState에 tool_call_count 필드가 존재한다."""
        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        config = CodingConfig()
        CodingAssistantAgent(config=config)
        from coding_agent.agents.coding_assistant.schemas import CodingState

        # CodingState에 tool_call_count 필드가 존재하는지 확인
        annotations = CodingState.__annotations__
        assert "tool_call_count" in annotations, "tool_call_count 필드 누락"

    @pytest.mark.asyncio
    async def test_max_tool_calls_skip_when_exceeded(self):
        """한도 초과 시 도구 실행이 스킵되고 알림 메시지가 반환된다."""
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage, ToolMessage

        agent = self._make_agent(max_tool_calls=2)

        # max_tool_calls=2이고 현재 카운트=2이면 초과
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "read_file", "args": {"path": "/tmp/test.py"}, "id": "call_1"},
        ]

        state = {
            "messages": [ai_msg],
            "tool_call_count": 2,  # 이미 한도 도달
            "project_context": [],
        }

        result = await agent._execute_tools(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert isinstance(msg, ToolMessage)
        assert "한도" in msg.content
        # tool_call_count는 증가하지 않아야 함 (skip이므로 반환에 없음)
        assert "tool_call_count" not in result

    @pytest.mark.asyncio
    async def test_tool_call_count_increments(self):
        """도구 실행 후 tool_call_count가 증가한다."""
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import AIMessage

        agent = self._make_agent(max_tool_calls=10)
        # 가짜 도구 설정
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.ainvoke = AsyncMock(return_value="result")
        agent._tools = [mock_tool]

        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "test_tool", "args": {}, "id": "call_1"},
            {"name": "test_tool", "args": {}, "id": "call_2"},
        ]

        state = {
            "messages": [ai_msg],
            "tool_call_count": 3,
            "project_context": [],
        }

        result = await agent._execute_tools(state)

        assert result["tool_call_count"] == 5  # 3 + 2

    def test_tool_call_count_field_exists(self):
        """CodingState 스키마에 tool_call_count 필드가 존재한다."""
        # verify_result 반환 구조 확인 (LLM 없이 반환값 구조만 테스트)
        from coding_agent.agents.coding_assistant.schemas import CodingState

        # verify_result가 반환하는 dict에 tool_call_count: 0이 포함되는지
        # agent.py 소스 코드에서 확인 가능하지만, 여기서는 스키마 필드 존재만 확인
        annotations = CodingState.__annotations__
        assert "tool_call_count" in annotations


class TestProjectContextDeduplication:
    """project_context 파일 경로 기준 중복 제거 테스트."""

    @pytest.mark.asyncio
    async def test_dedup_same_path(self):
        """같은 경로의 read_file 결과는 갱신(덮어쓰기)된다."""
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import AIMessage

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        config = CodingConfig(max_tool_calls=20)
        agent = CodingAssistantAgent(config=config)
        agent.build_graph()

        # read_file 도구 모킹
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.ainvoke = AsyncMock(return_value="new content")
        agent._tools = [mock_tool]

        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "read_file", "args": {"path": "/app/main.py"}, "id": "call_1"},
        ]

        state = {
            "messages": [ai_msg],
            "tool_call_count": 0,
            "project_context": ["[/app/main.py]\nold content"],
        }

        result = await agent._execute_tools(state)

        ctx = result["project_context"]
        # 같은 경로는 1개만 존재해야 함
        matching = [e for e in ctx if e.startswith("[/app/main.py]")]
        assert len(matching) == 1, f"중복 항목 존재: {len(matching)}개"
        assert "new content" in matching[0], "최신 내용으로 갱신되어야 함"

    @pytest.mark.asyncio
    async def test_different_paths_both_kept(self):
        """다른 경로의 read_file 결과는 모두 유지된다."""
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import AIMessage

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        config = CodingConfig(max_tool_calls=20)
        agent = CodingAssistantAgent(config=config)
        agent.build_graph()

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.ainvoke = AsyncMock(return_value="file_b content")
        agent._tools = [mock_tool]

        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "read_file", "args": {"path": "/app/utils.py"}, "id": "call_1"},
        ]

        state = {
            "messages": [ai_msg],
            "tool_call_count": 0,
            "project_context": ["[/app/main.py]\nmain content"],
        }

        result = await agent._execute_tools(state)

        ctx = result["project_context"]
        assert len(ctx) == 2, f"서로 다른 경로는 모두 유지되어야 함: {len(ctx)}개"


##############################################################################
# Memory Integration 테스트
##############################################################################


class TestMemoryRetrieval:
    """_retrieve_memory 노드 — Procedural/Episodic Memory 자동 검색."""

    def _make_agent_with_memory(self):
        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )
        from coding_agent.core.memory.store import MemoryStore

        store = MemoryStore()
        config = CodingConfig()
        agent = CodingAssistantAgent(config=config, memory_store=store)
        agent.build_graph()
        return agent, store

    @pytest.mark.asyncio
    async def test_retrieve_memory_returns_empty_without_store(self):
        """memory_store가 없으면 빈 딕셔너리를 반환한다."""
        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        agent = CodingAssistantAgent(config=CodingConfig())
        agent.build_graph()

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "피보나치 함수",
            },
        }
        result = await agent._retrieve_memory(state)
        # 단순화된 그래프(v2): 안전장치 초기화 값이 포함됨
        assert "iteration" in result
        assert "max_iterations" in result
        # 메모리 관련 키는 없어야 함
        assert "procedural_skills" not in result
        assert "episodic_log" not in result

    @pytest.mark.asyncio
    async def test_retrieve_procedural_skills(self):
        """저장된 Procedural 스킬이 검색되어 상태에 주입된다."""
        agent, store = self._make_agent_with_memory()

        # 스킬 사전 저장
        store.accumulate_skill(
            code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            description="피보나치 재귀 함수 생성",
            tags=["generate", "python"],
        )

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "피보나치 함수를 작성해줘",
            },
        }
        result = await agent._retrieve_memory(state)

        assert "procedural_skills" in result
        assert len(result["procedural_skills"]) >= 1
        assert any("피보나치" in s for s in result["procedural_skills"])

    @pytest.mark.asyncio
    async def test_retrieve_episodic_log(self):
        """저장된 Episodic 이력이 검색되어 상태에 주입된다."""
        from coding_agent.core.memory.schemas import MemoryItem, MemoryType

        agent, store = self._make_agent_with_memory()

        # 에피소딕 이력 사전 저장
        store.put(
            MemoryItem(
                type=MemoryType.EPISODIC,
                content="[성공] generate/python: 피보나치 함수 작성",
                tags=["generate", "python", "성공"],
            )
        )

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "피보나치 함수를 작성해줘",
            },
        }
        result = await agent._retrieve_memory(state)

        assert "episodic_log" in result
        assert len(result["episodic_log"]) >= 1
        assert any("피보나치" in e for e in result["episodic_log"])

    @pytest.mark.asyncio
    async def test_retrieve_memory_empty_store_returns_empty(self):
        """메모리가 비어 있으면 관련 키가 반환되지 않는다."""
        agent, store = self._make_agent_with_memory()

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "리스트 정렬",
            },
        }
        result = await agent._retrieve_memory(state)

        # 빈 스토어에서는 결과 없음 → 키 자체가 없어야 함
        assert "procedural_skills" not in result
        assert "episodic_log" not in result

    @pytest.mark.asyncio
    async def test_retrieve_memory_node_in_graph(self):
        """retrieve_memory 노드가 그래프에 등록되어 있다."""
        agent, _ = self._make_agent_with_memory()
        node_names = set(agent.graph.get_graph().nodes.keys())
        assert "retrieve_memory" in node_names


class TestEpisodicMemoryAccumulation:
    """_record_episodic_memory — 실행 결과를 Episodic Memory에 기록."""

    def _make_agent_with_memory(self):
        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )
        from coding_agent.core.memory.store import MemoryStore

        store = MemoryStore()
        config = CodingConfig()
        agent = CodingAssistantAgent(config=config, memory_store=store)
        agent.build_graph()
        return agent, store

    def test_record_episodic_on_pass(self):
        """검증 통과 시 에피소딕 메모리가 기록된다."""
        from coding_agent.core.memory.schemas import MemoryType

        agent, store = self._make_agent_with_memory()

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "피보나치 함수 작성",
            },
            "generated_code": "def fib(n): ...",
        }
        verify_result = {"passed": True, "issues": [], "suggestions": []}

        agent._record_episodic_memory(state, verify_result)

        items = store.list_by_type(MemoryType.EPISODIC)
        assert len(items) == 1
        assert "성공" in items[0].content
        assert "피보나치" in items[0].content
        assert items[0].metadata["passed"] is True

    def test_record_episodic_on_fail(self):
        """검증 실패 시 에피소딕 메모리에 실패 이력이 기록된다."""
        from coding_agent.core.memory.schemas import MemoryType

        agent, store = self._make_agent_with_memory()

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "fix",
                "language": "javascript",
                "description": "버그 수정",
            },
            "generated_code": "console.log('fix')",
        }
        verify_result = {
            "passed": False,
            "issues": ["타입 오류 발견"],
            "suggestions": [],
        }

        agent._record_episodic_memory(state, verify_result)

        items = store.list_by_type(MemoryType.EPISODIC)
        assert len(items) == 1
        assert "실패" in items[0].content
        assert "버그 수정" in items[0].content
        assert "타입 오류" in items[0].content
        assert items[0].metadata["passed"] is False

    def test_no_record_without_memory_store(self):
        """memory_store가 없으면 에러 없이 무시된다."""
        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        agent = CodingAssistantAgent(config=CodingConfig())
        agent.build_graph()

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "test",
            },
        }
        verify_result = {"passed": True, "issues": []}

        # memory_store가 None이므로 _record_episodic_memory는 호출되지 않지만,
        # 직접 호출해도 에러가 나지 않아야 한다 (try/except 보호)
        agent._record_episodic_memory(state, verify_result)
        # 에러 없이 통과하면 성공

    def test_episodic_tags_include_status(self):
        """에피소딕 메모리 태그에 성공/실패 상태가 포함된다."""
        from coding_agent.core.memory.schemas import MemoryType

        agent, store = self._make_agent_with_memory()

        state = {
            "messages": [],
            "parse_result": {
                "task_type": "generate",
                "language": "python",
                "description": "테스트",
            },
        }

        # 성공 케이스
        agent._record_episodic_memory(state, {"passed": True, "issues": []})
        items = store.list_by_type(MemoryType.EPISODIC)
        assert "성공" in items[0].tags

        # 실패 케이스
        agent._record_episodic_memory(state, {"passed": False, "issues": ["에러"]})
        items = store.list_by_type(MemoryType.EPISODIC)
        fail_items = [i for i in items if "실패" in i.tags]
        assert len(fail_items) == 1


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "1"

    if step == "1":
        print("=== Step 1: 프레임워크 검증 (LLM 불필요) ===\n")
        test_step1_graph_structure()
        test_step1_safety_envelope()
    elif step == "2":
        print("=== Step 2: LLM 연동 테스트 ===\n")
        asyncio.run(test_step2_llm_integration())
    else:
        print(f"알 수 없는 step: {step}")
