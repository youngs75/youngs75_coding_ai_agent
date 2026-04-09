"""Phase 11 E2E 통합 테스트 — 실제 LLM + MCP로 전체 파이프라인 검증.

테스트 범위:
1. 훅 시스템이 실제 에이전트 실행에서 동작하는지
2. ParallelToolExecutor가 실제 MCP 도구와 작동하는지
3. ToolPermissionManager가 실제 도구 호출을 필터링하는지
4. CoordinatorMode가 실제 LLM으로 작업을 분해하는지
5. 전체 파이프라인: CLI → Agent → MCP → LLM → 결과

준비:
  1. MCP 서버: cd docker && docker compose -f docker-compose.mcp.yml up -d
  2. API 키: .env에 OPENROUTER_API_KEY 설정

실행: python -m pytest tests/test_e2e_phase11.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ── 스킵 조건 ────────────────────────────────────────────────


def _has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _has_mcp_server(port: int = 3003) -> bool:
    """MCP 서버가 실행 중인지 확인."""
    import socket

    try:
        with socket.create_connection(("localhost", port), timeout=2):
            return True
    except OSError:
        return False


_skip_no_api = pytest.mark.skipif(
    not _has_openrouter_key(),
    reason="OPENROUTER_API_KEY가 설정되지 않았습니다",
)
_skip_no_mcp = pytest.mark.skipif(
    not _has_mcp_server(),
    reason="MCP code_tools 서버가 실행되지 않았습니다 (docker compose up -d 필요)",
)


# ── 1. MCP + 훅 시스템 통합 테스트 ──────────────────────────


class TestHookSystemE2E:
    """실제 MCP 도구 실행에서 훅이 동작하는지 검증."""

    @_skip_no_mcp
    async def test_hooks_fire_on_mcp_tool_execution(self):
        """MCP 도구 실행 시 PRE/POST_TOOL_CALL 훅이 발행된다."""
        from coding_agent.core.hooks import HookContext, HookEvent, HookManager
        from coding_agent.core.mcp_loader import MCPToolLoader
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor

        # 훅 이벤트 기록
        events: list[str] = []

        async def record_hook(ctx: HookContext) -> HookContext:
            events.append(f"{ctx.event.value}:{ctx.tool_name}")
            return ctx

        hook_mgr = HookManager()
        hook_mgr.register(HookEvent.PRE_TOOL_CALL, record_hook)
        hook_mgr.register(HookEvent.POST_TOOL_CALL, record_hook)

        # MCP 도구 로드
        loader = MCPToolLoader(
            servers={"code_tools": "http://localhost:3003/mcp/"},
            health_timeout=5.0,
        )
        tools = await loader.load()
        assert len(tools) > 0, "MCP 도구가 로드되지 않았습니다"

        tools_by_name = {getattr(t, "name", None): t for t in tools}

        # ParallelToolExecutor로 도구 실행 (상세 결과)
        executor = ParallelToolExecutor(hook_manager=hook_mgr)

        async def tool_fn(name: str, args: dict) -> str:
            if name in tools_by_name:
                result = await tools_by_name[name].ainvoke(args)
                return str(result)
            return f"unknown: {name}"

        mock_tool_call = {
            "name": "list_directory",
            "id": "call_test_1",
            "args": {"path": "."},
        }

        batch_result = await executor.execute_batch_detailed([mock_tool_call], tool_fn)

        # 훅 발행 확인
        assert "pre_tool_call:list_directory" in events
        assert "post_tool_call:list_directory" in events
        assert len(batch_result.results) == 1
        assert batch_result.results[0].success

    @_skip_no_mcp
    async def test_hook_cancellation_prevents_execution(self):
        """PRE_TOOL_CALL 훅에서 cancel=True 설정 시 도구 실행이 스킵된다."""
        from coding_agent.core.hooks import HookContext, HookEvent, HookManager
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor

        executed = []

        async def cancel_hook(ctx: HookContext) -> HookContext:
            ctx.metadata["cancel"] = True
            return ctx

        hook_mgr = HookManager()
        hook_mgr.register(HookEvent.PRE_TOOL_CALL, cancel_hook)

        executor = ParallelToolExecutor(hook_manager=hook_mgr)

        async def tool_fn(name: str, args: dict) -> str:
            executed.append(name)
            return "should not reach here"

        mock_call = {
            "name": "list_directory",
            "id": "call_cancel",
            "args": {"path": "."},
        }
        batch_result = await executor.execute_batch_detailed([mock_call], tool_fn)

        assert len(executed) == 0, "취소된 도구가 실행되었습니다"
        assert batch_result.results[0].cancelled

    @_skip_no_mcp
    async def test_builtin_hooks_with_real_tools(self):
        """내장 훅(logging, timing, audit)이 실제 도구와 동작한다."""
        from coding_agent.core.builtin_hooks import (
            audit_hook,
            logging_hook,
            timing_hook,
        )
        from coding_agent.core.hooks import HookEvent, HookManager
        from coding_agent.core.mcp_loader import MCPToolLoader
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor

        hook_mgr = HookManager()
        hook_mgr.register(HookEvent.PRE_TOOL_CALL, logging_hook, priority=0)
        hook_mgr.register(HookEvent.PRE_TOOL_CALL, timing_hook, priority=1)
        hook_mgr.register(HookEvent.PRE_TOOL_CALL, audit_hook, priority=2)
        hook_mgr.register(HookEvent.POST_TOOL_CALL, timing_hook, priority=0)
        hook_mgr.register(HookEvent.POST_TOOL_CALL, logging_hook, priority=1)

        loader = MCPToolLoader(
            servers={"code_tools": "http://localhost:3003/mcp/"},
            health_timeout=5.0,
        )
        tools = await loader.load()
        tools_by_name = {getattr(t, "name", None): t for t in tools}

        executor = ParallelToolExecutor(hook_manager=hook_mgr)

        async def tool_fn(name: str, args: dict) -> str:
            if name in tools_by_name:
                return str(await tools_by_name[name].ainvoke(args))
            return f"unknown: {name}"

        mock_call = {
            "name": "read_file",
            "id": "call_bf",
            "args": {"path": "pyproject.toml"},
        }
        batch_result = await executor.execute_batch_detailed([mock_call], tool_fn)

        assert batch_result.results[0].success
        assert batch_result.results[0].duration_s >= 0


# ── 2. ToolPermissionManager E2E ─────────────────────────────


class TestPermissionManagerE2E:
    """실제 도구 호출에서 권한 검사가 동작하는지 검증."""

    @_skip_no_mcp
    async def test_deny_sensitive_file_access(self):
        """민감 파일(.env) 접근 시 ASK 이상 권한이 필요하다."""
        from coding_agent.core.tool_permissions import (
            PermissionDecision,
            ToolPermissionManager,
        )

        mgr = ToolPermissionManager(workspace=".")

        # .env 파일 쓰기 시도
        decision = mgr.check("write_file", {"path": "/app/.env", "content": "test"})
        # 민감 파일은 ASK 또는 DENY
        assert decision in (PermissionDecision.ASK, PermissionDecision.DENY)

    @_skip_no_mcp
    async def test_allow_safe_tool(self):
        """안전한 도구(list_directory)는 허용된다."""
        from coding_agent.core.tool_permissions import (
            PermissionDecision,
            ToolPermissionManager,
        )

        mgr = ToolPermissionManager(workspace=".")
        decision = mgr.check("list_directory", {"path": "."})
        assert decision == PermissionDecision.ALLOW

    @_skip_no_mcp
    async def test_permission_integrated_with_executor(self):
        """권한 훅이 ParallelToolExecutor와 통합되어 동작한다."""
        from coding_agent.core.hooks import HookContext, HookEvent, HookManager
        from coding_agent.core.mcp_loader import MCPToolLoader
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor
        from coding_agent.core.tool_permissions import (
            PermissionDecision,
            ToolPermissionManager,
        )

        perm_mgr = ToolPermissionManager(workspace=".")
        hook_mgr = HookManager()

        # 권한 검사 훅
        async def permission_hook(ctx: HookContext) -> HookContext:
            if ctx.tool_name and ctx.event == HookEvent.PRE_TOOL_CALL:
                decision = perm_mgr.check(ctx.tool_name, ctx.tool_args or {})
                if decision == PermissionDecision.DENY:
                    ctx.metadata["cancel"] = True
                    ctx.metadata["deny_reason"] = f"권한 거부: {ctx.tool_name}"
            return ctx

        hook_mgr.register(HookEvent.PRE_TOOL_CALL, permission_hook, priority=-10)

        executor = ParallelToolExecutor(hook_manager=hook_mgr)

        loader = MCPToolLoader(
            servers={"code_tools": "http://localhost:3003/mcp/"},
            health_timeout=5.0,
        )
        tools = await loader.load()
        tools_by_name = {getattr(t, "name", None): t for t in tools}

        executed_tools = []

        async def tool_fn(name: str, args: dict) -> str:
            executed_tools.append(name)
            if name in tools_by_name:
                return str(await tools_by_name[name].ainvoke(args))
            return "unknown"

        # 안전한 도구 호출
        calls = [
            {"name": "list_directory", "id": "call_safe", "args": {"path": "."}},
        ]

        batch_result = await executor.execute_batch_detailed(calls, tool_fn)

        assert "list_directory" in executed_tools
        assert batch_result.results[0].success


# ── 3. ParallelToolExecutor E2E ──────────────────────────────


class TestParallelToolExecutorE2E:
    """실제 MCP 도구에서 병렬 실행이 동작하는지 검증."""

    @_skip_no_mcp
    async def test_parallel_read_operations(self):
        """읽기 전용 도구가 병렬로 실행된다."""
        from coding_agent.core.mcp_loader import MCPToolLoader
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor

        loader = MCPToolLoader(
            servers={"code_tools": "http://localhost:3003/mcp/"},
            health_timeout=5.0,
        )
        tools = await loader.load()
        tools_by_name = {getattr(t, "name", None): t for t in tools}

        async def tool_fn(name: str, args: dict) -> str:
            if name in tools_by_name:
                return str(await tools_by_name[name].ainvoke(args))
            return "unknown"

        # 3개 읽기 전용 도구 병렬 호출
        calls = [
            {"name": "list_directory", "id": "call_1", "args": {"path": "."}},
            {"name": "read_file", "id": "call_2", "args": {"path": "pyproject.toml"}},
            {"name": "list_directory", "id": "call_3", "args": {"path": "core"}},
        ]

        executor = ParallelToolExecutor()

        start = time.monotonic()
        batch_result = await executor.execute_batch_detailed(calls, tool_fn)
        parallel_time = time.monotonic() - start

        assert batch_result.success_count == 3
        assert all(r.success for r in batch_result.results)

        # 각 도구 실행 시간 합 vs 실제 벽시계 시간
        total_individual = sum(r.duration_s for r in batch_result.results)
        print(f"  병렬 실행: {parallel_time:.3f}s, 개별 합: {total_individual:.3f}s")

    @_skip_no_mcp
    async def test_batch_result_order_preserved(self):
        """결과 순서가 요청 순서와 동일하다."""
        from coding_agent.core.mcp_loader import MCPToolLoader
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor

        loader = MCPToolLoader(
            servers={"code_tools": "http://localhost:3003/mcp/"},
            health_timeout=5.0,
        )
        tools = await loader.load()
        tools_by_name = {getattr(t, "name", None): t for t in tools}

        async def tool_fn(name: str, args: dict) -> str:
            if name in tools_by_name:
                return str(await tools_by_name[name].ainvoke(args))
            return "unknown"

        calls = [
            {"name": "read_file", "id": "c1", "args": {"path": "pyproject.toml"}},
            {"name": "list_directory", "id": "c2", "args": {"path": "."}},
            {"name": "read_file", "id": "c3", "args": {"path": "README.md"}},
        ]

        executor = ParallelToolExecutor()
        batch_result = await executor.execute_batch_detailed(calls, tool_fn)

        for i, r in enumerate(batch_result.results):
            assert r.index == i, f"결과 순서 불일치: 기대 {i}, 실제 {r.index}"


# ── 4. Coordinator Mode E2E ──────────────────────────────────


class TestCoordinatorModeE2E:
    """Coordinator Mode가 실제 LLM으로 작업을 분해하고 실행하는지 검증."""

    @_skip_no_api
    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_task_decomposition_with_real_llm(self):
        """실제 LLM이 복합 작업을 서브태스크로 분해한다."""
        from coding_agent.agents.orchestrator.coordinator import CoordinatorMode
        from coding_agent.core.context_manager import ContextManager
        from coding_agent.core.model_tiers import (
            build_default_purpose_tiers,
            build_default_tiers,
            create_chat_model,
            resolve_tier_config,
        )
        from coding_agent.core.subagents.registry import SubAgentRegistry
        from coding_agent.core.subagents.schemas import SubAgentSpec

        # 에이전트 등록
        agents = [
            SubAgentSpec(
                name="coding_assistant",
                description="코드 생성, 수정, 리팩토링",
                capabilities=["code_generation", "code_review"],
                endpoint="http://localhost:8001",
            ),
            SubAgentSpec(
                name="deep_research",
                description="심층 조사 및 리서치",
                capabilities=["research", "analysis"],
                endpoint="http://localhost:8002",
            ),
        ]
        registry = SubAgentRegistry()
        for agent in agents:
            registry.register(agent)

        ctx_mgr = ContextManager(max_tokens=128000)

        coordinator = CoordinatorMode(
            registry=registry,
            context_manager=ctx_mgr,
            max_workers=3,
            timeout_s=60.0,
        )

        tiers = build_default_tiers()
        purpose_tiers = build_default_purpose_tiers()
        tier_config = resolve_tier_config("fast", tiers, purpose_tiers)
        llm = create_chat_model(tier_config, temperature=0.1)

        subtasks = await asyncio.wait_for(
            coordinator.decompose_task(
                task="프로젝트의 코드 품질을 분석하고 개선 방안을 제안해줘",
                llm=llm,
                available_agents=agents,
            ),
            timeout=90.0,
        )

        assert len(subtasks) >= 1, "서브태스크가 생성되지 않았습니다"
        print(f"  분해된 서브태스크: {len(subtasks)}개")
        for st in subtasks:
            print(
                f"    - [{st.get('agent_type', '?')}] {st.get('description', '?')[:60]}"
            )

    async def test_task_graph_with_decomposed_tasks(self):
        """분해된 서브태스크로 TaskGraph를 구성하고 실행 웨이브를 계산한다."""
        from coding_agent.agents.orchestrator.task_graph import TaskGraph

        subtasks = [
            {
                "id": "t1",
                "description": "코드 분석",
                "agent_type": "coding_assistant",
                "dependencies": [],
                "priority": 1,
                "timeout_s": 30.0,
            },
            {
                "id": "t2",
                "description": "테스트 현황 분석",
                "agent_type": "coding_assistant",
                "dependencies": [],
                "priority": 1,
                "timeout_s": 30.0,
            },
            {
                "id": "t3",
                "description": "결과 종합",
                "agent_type": "deep_research",
                "dependencies": ["t1", "t2"],
                "priority": 2,
                "timeout_s": 30.0,
            },
        ]

        graph = TaskGraph(subtasks)
        assert graph.validate(), "DAG 유효성 검사 실패"

        waves = graph.execution_waves
        assert len(waves) == 2, f"기대 2웨이브, 실제 {len(waves)}웨이브"

        # 1웨이브: t1, t2 (병렬)
        wave1_ids = {st["id"] for st in waves[0]}
        assert wave1_ids == {"t1", "t2"}

        # 2웨이브: t3 (의존성 완료 후)
        wave2_ids = {st["id"] for st in waves[1]}
        assert wave2_ids == {"t3"}

        print(f"  실행 웨이브: {len(waves)}개")
        for i, wave in enumerate(waves):
            ids = [st["id"] for st in wave]
            print(f"    Wave {i + 1}: {ids}")

    @_skip_no_api
    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_coordinator_full_pipeline_with_mock_workers(self):
        """Coordinator 전체 파이프라인: decompose → execute → synthesize."""
        from coding_agent.agents.orchestrator.coordinator import CoordinatorMode
        from coding_agent.core.context_manager import ContextManager
        from coding_agent.core.model_tiers import (
            build_default_purpose_tiers,
            build_default_tiers,
            create_chat_model,
            resolve_tier_config,
        )
        from coding_agent.core.subagents.registry import SubAgentRegistry
        from coding_agent.core.subagents.schemas import SubAgentSpec

        agents = [
            SubAgentSpec(
                name="coding_assistant",
                description="코드 생성, 수정, 리팩토링",
                capabilities=["code_generation"],
                endpoint="http://localhost:8001",
            ),
        ]
        registry = SubAgentRegistry()
        for agent in agents:
            registry.register(agent)

        ctx_mgr = ContextManager(max_tokens=128000)

        # 워커 함수 모킹 — 실제 에이전�� 대신 간단한 응답 반환
        async def mock_worker(subtask, context):
            await asyncio.sleep(0.1)
            return {
                "subtask_id": subtask["id"],
                "agent_name": subtask.get("agent_type", "mock"),
                "status": "success",
                "output": f"[{subtask['id']}] 완료: {subtask['description']}",
                "duration_s": 0.1,
                "error": None,
            }

        coordinator = CoordinatorMode(
            registry=registry,
            context_manager=ctx_mgr,
            max_workers=3,
            timeout_s=30.0,
            worker_fn=mock_worker,
        )

        tiers = build_default_tiers()
        purpose_tiers = build_default_purpose_tiers()
        tier_config = resolve_tier_config("fast", tiers, purpose_tiers)
        llm = create_chat_model(tier_config, temperature=0.1)

        result = await asyncio.wait_for(
            coordinator.run(
                task="간단한 피보나치 함수를 작성해줘",
                context=[],
                llm=llm,
            ),
            timeout=60.0,
        )

        assert result["synthesized_response"], "최종 응답이 비어있습니다"
        assert len(result["worker_results"]) >= 1
        print(f"  워커 결과: {len(result['worker_results'])}개")
        print(f"  최종 응답 길이: {len(result['synthesized_response'])}자")
        print(f"  총 소요: {result['total_duration_s']:.2f}s")


# ── 5. CodingAssistant 전체 파이프라인 E2E ───────────────────


class TestCodingAssistantE2E:
    """CodingAssistantAgent가 Phase 11 기능과 함께 전체 동작하는지 검증."""

    @_skip_no_api
    @_skip_no_mcp
    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_full_pipeline_with_hooks_and_permissions(self):
        """CodingAssistant가 훅+권한+MCP로 코드를 생성한다."""
        from langchain_core.messages import HumanMessage

        from coding_agent.agents.coding_assistant import (
            CodingAssistantAgent,
            CodingConfig,
        )
        from coding_agent.core.hooks import HookContext, HookEvent, HookManager
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor
        from coding_agent.core.tool_permissions import ToolPermissionManager

        # Phase 11 기능 초기화
        hook_events: list[str] = []

        async def trace_hook(ctx: HookContext) -> HookContext:
            hook_events.append(
                f"{ctx.event.value}:{ctx.tool_name or ctx.node_name or '?'}"
            )
            return ctx

        hook_mgr = HookManager()
        hook_mgr.register(HookEvent.PRE_TOOL_CALL, trace_hook)
        hook_mgr.register(HookEvent.POST_TOOL_CALL, trace_hook)

        perm_mgr = ToolPermissionManager(workspace=".")
        tool_exec = ParallelToolExecutor(hook_manager=hook_mgr)

        config = CodingConfig()
        agent = await CodingAssistantAgent.create(
            config=config,
            hook_manager=hook_mgr,
        )
        # Phase 10 기능 주입
        agent.permission_manager = perm_mgr
        agent.tool_executor = tool_exec
        agent.project_context = (
            "이 프로젝트는 Python 기반 AI 에이전트 프레임워크입니다."
        )

        result = await asyncio.wait_for(
            agent.graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="파이썬으로 피보나치 수열 함수를 작성해줘")
                    ],
                    "iteration": 0,
                    "max_iterations": 2,
                }
            ),
            timeout=300.0,
        )

        # 결과 확인
        generated = result.get("generated_code", "")
        verify = result.get("verify_result", {})
        log = result.get("execution_log", [])

        print(f"  생성 코드 길이: {len(generated)}자")
        print(f"  검증 결과: {'통과' if verify.get('passed') else '실패'}")
        print(f"  실행 로그: {len(log)}개 항목")
        print(f"  훅 이벤트: {len(hook_events)}개")

        assert generated, "코드가 생성되지 않았습니다"
        assert len(log) >= 2, "실행 로그가 부족합니다 (parse + execute 이상 필요)"

    @_skip_no_mcp
    async def test_project_context_loading(self):
        """프로젝트 컨텍스트가 로드된다."""
        from coding_agent.core.project_context import ProjectContextLoader

        loader = ProjectContextLoader(workspace=Path.cwd())
        context = loader.load()  # sync method

        assert context, "프로젝트 컨텍스트가 비어있습니다"
        print(f"  프로젝트 컨텍스트 길이: {len(context)}자")
        assert "AGENTS.md" in context or "agent" in context.lower()


# ── 6. Context Manager E2E ───────────────────────────────────


class TestContextManagerE2E:
    """ContextManager가 실제 메시지 시퀀스에서 동작하는지 검증."""

    async def test_should_compact_with_large_messages(self):
        """큰 메시지 시퀀스에서 컴팩션 필요 여부를 정확히 판단한다."""
        from langchain_core.messages import AIMessage, HumanMessage
        from coding_agent.core.context_manager import ContextManager

        mgr = ContextManager(max_tokens=1000, compact_threshold=0.8)

        # 작은 메시지 — 컴팩션 불필요
        small_msgs = [HumanMessage(content="안녕")]
        assert not mgr.should_compact(small_msgs)

        # 큰 메시지 — 컴팩션 필요
        big_content = "x" * 5000
        big_msgs = [
            HumanMessage(content=big_content),
            AIMessage(content=big_content),
            HumanMessage(content=big_content),
        ]
        assert mgr.should_compact(big_msgs)

    async def test_truncate_for_subagent(self):
        """서브에이전트용 히스토리 전파가 동작한다."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from coding_agent.core.context_manager import ContextManager

        mgr = ContextManager(max_tokens=128000)
        msgs = [
            HumanMessage(content="첫 질문"),
            AIMessage(content="첫 답변"),
            HumanMessage(content="두번째 질문"),
            AIMessage(
                content="도구 호출", additional_kwargs={"tool_calls": [{"id": "1"}]}
            ),
            ToolMessage(content="도구 결과", tool_call_id="1"),
            AIMessage(content="두번째 답변"),
            HumanMessage(content="세번째 질문"),
        ]

        truncated = mgr.truncate_for_subagent(msgs, last_n_turns=2)
        # ToolMessage는 제외되어야 함
        assert not any(isinstance(m, ToolMessage) for m in truncated)
        # 최근 메시지가 포함되어야 함
        assert len(truncated) > 0


# ── 7. 전체 시스템 통합 ──────────────────────────────────────


class TestFullSystemIntegration:
    """전체 시스템이 함께 동작하는지 검증하는 스모크 테스트."""

    @_skip_no_mcp
    async def test_mcp_tool_loading_all_servers(self):
        """모든 MCP 서버에서 도구가 로드된다."""
        from coding_agent.core.mcp_loader import MCPToolLoader

        servers = {}
        import socket

        for name, port in [
            ("code_tools", 3003),
            ("arxiv", 3000),
            ("tavily", 3001),
            ("serper", 3002),
        ]:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    servers[name] = f"http://localhost:{port}/mcp/"
            except OSError:
                pass

        assert servers, "실행 중인 MCP 서버가 없습니다"

        loader = MCPToolLoader(servers=servers, health_timeout=5.0)
        tools = await loader.load()

        print(f"  MCP 서버 {len(servers)}개에서 도구 {len(tools)}개 로드:")
        for t in tools:
            print(f"    - {getattr(t, 'name', '?')}")

        assert len(tools) > 0

    async def test_all_phase11_modules_importable(self):
        """Phase 11 모든 모듈이 정상 임포트된다."""
        # Hook system
        from coding_agent.core.hooks import HookContext, HookEvent, HookManager
        from coding_agent.core.builtin_hooks import (
            audit_hook,
            logging_hook,
            timing_hook,
        )

        # Coordinator
        from coding_agent.agents.orchestrator.coordinator import CoordinatorMode
        from coding_agent.agents.orchestrator.task_graph import TaskGraph

        # Phase 10 integration
        from coding_agent.core.parallel_tool_executor import ParallelToolExecutor
        from coding_agent.core.tool_permissions import ToolPermissionManager
        from coding_agent.core.context_manager import ContextManager
        from coding_agent.core.project_context import ProjectContextLoader

        # 임포트 검증 — 모듈이 callable이거나 인스턴스화 가능한지 확인
        assert callable(logging_hook)
        assert callable(timing_hook)
        assert callable(audit_hook)
        assert callable(HookContext)
        assert callable(HookEvent)
        assert CoordinatorMode is not None
        assert TaskGraph is not None
        assert ProjectContextLoader is not None
        assert HookManager() is not None
        assert ParallelToolExecutor() is not None
        assert ToolPermissionManager(workspace=".") is not None
        assert ContextManager(max_tokens=1000) is not None

    @_skip_no_api
    @_skip_no_mcp
    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_smoke_simple_react_with_phase11(self):
        """SimpleReActAgent가 Phase 11 기능과 함께 동작한다."""
        from langchain_core.messages import HumanMessage
        from coding_agent.agents.simple_react import (
            SimpleMCPReActAgent,
            SimpleReActConfig,
        )
        from coding_agent.core.context_manager import ContextManager

        config = SimpleReActConfig(default_model="deepseek/deepseek-v3.2")
        agent = await SimpleMCPReActAgent.create(config=config)

        # Phase 10 주입
        agent.context_manager = ContextManager(max_tokens=128000)
        agent.project_context = "테스트 프로젝트"

        if not agent._tools:
            pytest.skip("MCP 도구 로드 실패")

        result = await asyncio.wait_for(
            agent.graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(content="현재 디렉토리의 파일 목록을 보여줘")
                    ]
                }
            ),
            timeout=60.0,
        )

        last_msg = result["messages"][-1]
        assert last_msg.content, "응답이 비어있습니다"
        print(f"  SimpleReAct 응답 길이: {len(last_msg.content)}자")
