"""수동 시나리오 E2E 테스트 — 실제 LLM + MCP로 에이전트 전체 동작 검증.

시나리오:
1. 단순 코드 생성: 도구 호출 최소화 확인
2. MCP 파일 읽기: read_file 도구 정상 동작
3. 버그 수정: fix 요청 처리
5. 검증기 품질: 문법 오류 코드 탐지

실행: python -m pytest tests/test_scenarios_e2e.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _has_mcp_server(port: int = 3003) -> bool:
    import socket

    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


_skip_no_api = pytest.mark.skipif(
    not _has_openrouter_key(), reason="OPENROUTER_API_KEY 필요"
)
_skip_no_mcp = pytest.mark.skipif(
    not _has_mcp_server(), reason="MCP code_tools 서버 필요"
)


# ── 시나리오 1: 단순 코드 생성 ──────────────────────────────


class TestScenario1SimpleCodeGeneration:
    """도구 호출 없이 바로 코드를 생성하는지 검증."""

    @_skip_no_api
    @_skip_no_mcp
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_simple_function_minimal_tool_calls(self):
        """이진 탐색 함수 요청 시 도구 호출 횟수가 적어야 한다."""
        from langchain_core.messages import HumanMessage

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        config = CodingConfig()
        agent = await CodingAssistantAgent.create(config=config)

        result = await asyncio.wait_for(
            agent.graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="파이썬으로 이진 탐색 함수를 하나만 작성해줘. 간단하게."
                        )
                    ],
                    "iteration": 0,
                    "max_iterations": 2,
                }
            ),
            timeout=120.0,
        )

        generated = result.get("generated_code", "")
        tool_count = result.get("tool_call_count", 0)
        log = result.get("execution_log", [])

        print(f"  생성 코드 길이: {len(generated)}자")
        print(f"  도구 호출 횟수: {tool_count}")
        print(f"  실행 로그: {log}")

        assert generated, "코드가 생성되지 않았습니다"
        assert "def " in generated or "binary" in generated.lower(), (
            "이진 탐색 함수가 포함되어 있지 않습니다"
        )
        # 단순 코드 생성이므로 도구 호출이 3회 이하여야 함
        assert tool_count <= 3, f"도구 호출이 너무 많습니다: {tool_count}회 (기대: ≤3)"


# ── 시나리오 2: MCP 도구 활용 ───────────────────────────────


class TestScenario2MCPToolUsage:
    """MCP 도구를 사용해 파일을 읽고 분석하는 작업 검증."""

    @_skip_no_api
    @_skip_no_mcp
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_read_and_analyze_file(self):
        """파일 분석 요청 시 응답이 올바른 정보를 포함한다."""
        from langchain_core.messages import HumanMessage

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        config = CodingConfig()
        agent = await CodingAssistantAgent.create(config=config)

        result = await asyncio.wait_for(
            agent.graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Makefile을 읽고 사용 가능한 make 타겟 목록을 알려줘"
                        )
                    ],
                    "iteration": 0,
                    "max_iterations": 2,
                }
            ),
            timeout=360.0,
        )

        generated = result.get("generated_code", "")
        tool_count = result.get("tool_call_count", 0)
        parse_result = result.get("parse_result", {})
        log = result.get("execution_log", [])

        print(f"  parse task_type: {parse_result.get('task_type')}")
        print(f"  응답 길이: {len(generated)}자")
        print(f"  도구 호출: {tool_count}회")
        print(f"  로그: {log}")

        assert generated, "응답이 비어있습니다"
        # Makefile 내용은 semantic memory에 없으므로 도구 호출 필요
        assert tool_count >= 1, (
            f"read_file 도구가 호출되지 않았습니다 (task_type={parse_result.get('task_type')})"
        )


# ── 시나리오 3: 버그 수정 ───────────────────────────────────


class TestScenario3BugFix:
    """버그가 있는 코드의 수정 요청 검증."""

    @_skip_no_api
    @_skip_no_mcp
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_fix_recursive_bug(self):
        """재귀 호출 버그를 수정하는지 확인."""
        from langchain_core.messages import HumanMessage

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )

        buggy_code = """다음 코드의 버그를 찾아서 수정해줘:

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n) + fibonacci(n-1)
"""

        config = CodingConfig()
        agent = await CodingAssistantAgent.create(config=config)

        result = await asyncio.wait_for(
            agent.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=buggy_code)],
                    "iteration": 0,
                    "max_iterations": 2,
                }
            ),
            timeout=120.0,
        )

        generated = result.get("generated_code", "")
        parse_result = result.get("parse_result", {})

        print(f"  parse task_type: {parse_result.get('task_type')}")
        print(f"  생성 코드 길이: {len(generated)}자")

        assert generated, "수정된 코드가 생성되지 않았습니다"
        # 수정된 코드에 fibonacci(n-2) 또는 올바른 재귀 호출이 있어야 함
        assert "n-2" in generated or "n - 2" in generated, (
            f"fibonacci(n-2) 수정이 반영되지 않았습니다: {generated[:200]}"
        )


# ── 시나리오 5: 검증기 품질 ─────────────────────────────────


class TestScenario5VerifierQuality:
    """검증기가 문법 오류/버그를 탐지하는지 확인."""

    @_skip_no_api
    async def test_verifier_catches_syntax_error(self):
        """문법 오류가 있는 코드를 검증기가 실패로 판정하는지."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from coding_agent.agents.coding_assistant.prompts import VERIFY_SYSTEM_PROMPT
        from coding_agent.core.model_tiers import (
            build_default_purpose_tiers,
            build_default_tiers,
            create_chat_model,
            resolve_tier_config,
        )

        tiers = build_default_tiers()
        purpose_tiers = build_default_purpose_tiers()
        tier_config = resolve_tier_config("verification", tiers, purpose_tiers)
        llm = create_chat_model(tier_config, temperature=0.1)

        buggy_code = """
def binary_search_leftmost(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if mid target:  # 문법 오류!
            left = mid + 1
        else:
            right = mid
    return left
"""

        verify_prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=50,
            allowed_extensions=".py, .js, .ts",
        )

        response = await asyncio.wait_for(
            llm.ainvoke(
                [
                    SystemMessage(content=verify_prompt),
                    HumanMessage(
                        content=f"원래 요청: 이진 탐색 함수 작성\n\n생성된 코드:\n{buggy_code}"
                    ),
                ]
            ),
            timeout=60.0,
        )

        import json

        content = response.content
        print(f"  검증 응답: {content[:300]}")

        try:
            verify_result = json.loads(content)
            passed = verify_result.get("passed", True)
            issues = verify_result.get("issues", [])
            print(f"  passed: {passed}")
            print(f"  issues: {issues}")
            # 문법 오류를 잡아야 함
            assert not passed, "문법 오류가 있는 코드를 통과시켰습니다"
        except json.JSONDecodeError:
            # JSON 파싱 실패해도 "passed": false 가 텍스트에 있으면 OK
            if '"passed": false' in content or '"passed":false' in content:
                print("  JSON 파싱 실패했지만 passed=false 확인됨")
            else:
                pytest.fail(f"검증 응답이 유효한 JSON이 아닙니다: {content[:200]}")


# ── 시나리오 4+: Semantic Memory → 에이전트 주입 ───────────


class TestScenario4SemanticMemoryInjection:
    """Semantic Memory가 에이전트 상태에 주입되는지 확인."""

    @_skip_no_api
    @_skip_no_mcp
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    async def test_semantic_context_in_agent_state(self):
        """CodingAssistant에 semantic_context가 주입된 상태로 실행된다."""
        from langchain_core.messages import HumanMessage
        from pathlib import Path

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )
        from coding_agent.core.memory.store import MemoryStore
        from coding_agent.core.memory.semantic_loader import SemanticMemoryLoader
        from coding_agent.core.memory.schemas import MemoryType

        # 메모리 준비
        store = MemoryStore()
        loader = SemanticMemoryLoader(workspace=Path.cwd(), store=store)
        count = loader.load_all()
        assert count > 0, "Semantic Memory 로딩 실패"

        # 에이전트 생성 (memory_store 주입)
        config = CodingConfig()
        agent = await CodingAssistantAgent.create(config=config, memory_store=store)

        # semantic_context를 입력 상태에 포함
        semantic_items = store.list_by_type(MemoryType.SEMANTIC)
        input_state = {
            "messages": [HumanMessage(content="간단한 hello world 함수를 작성해줘")],
            "iteration": 0,
            "max_iterations": 1,
            "semantic_context": [item.content for item in semantic_items],
        }

        result = await asyncio.wait_for(
            agent.graph.ainvoke(input_state),
            timeout=120.0,
        )

        generated = result.get("generated_code", "")
        print(f"  semantic_context 주입: {len(semantic_items)}건")
        print(f"  생성 코드 길이: {len(generated)}자")

        assert generated, "코드가 생성되지 않았습니다"
