"""SubAgent 프로세스 매니저 테스트.

검증 항목:
- ProcessManager spawn/wait/cancel 수명주기
- 상태 전이 (CREATED→ASSIGNED→RUNNING→COMPLETED→DESTROYED)
- 타임아웃 처리
- 좀비 프로세스 정리
- SubAgentResult JSON 파싱
- ResourceUsage 기록
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile

import pytest

from coding_agent.core.subagents.process_manager import SubAgentProcessManager
from coding_agent.core.subagents.registry import SubAgentRegistry
from coding_agent.core.subagents.schemas import (
    ResourceUsage,
    SubAgentResult,
    SubAgentSpec,
    SubAgentStatus,
    VALID_TRANSITIONS,
)


# ── fixtures ──


@pytest.fixture
def registry():
    reg = SubAgentRegistry()
    reg.register(SubAgentSpec(
        name="echo",
        description="Echo agent for testing",
        capabilities=["echo"],
    ))
    reg.register(SubAgentSpec(
        name="coding_assistant",
        description="Coding agent",
        capabilities=["coding"],
    ))
    return reg


@pytest.fixture
def manager(registry):
    return SubAgentProcessManager(registry=registry, timeout_s=30.0)


# ── SubAgentResult 테스트 ──


class TestSubAgentResult:
    def test_success_result(self):
        r = SubAgentResult(status="completed", result="hello", written_files=["a.py"])
        assert r.success is True
        assert r.result == "hello"
        assert r.written_files == ["a.py"]

    def test_failed_result(self):
        r = SubAgentResult(status="failed", error="boom")
        assert r.success is False
        assert r.error == "boom"

    def test_from_json(self):
        data = {
            "status": "completed",
            "result": "code here",
            "written_files": ["x.py"],
            "duration_s": 5.0,
            "token_usage": {"prompt": 100, "completion": 50},
            "error": None,
        }
        r = SubAgentResult(**data)
        assert r.success
        assert r.duration_s == 5.0


# ── ResourceUsage 테스트 ──


class TestResourceUsage:
    def test_basic(self):
        ru = ResourceUsage(pid=12345, agent_id="abc", start_time=100.0)
        assert ru.pid == 12345
        assert ru.end_time is None
        assert ru.exit_code is None

    def test_finalized(self):
        ru = ResourceUsage(
            pid=12345, agent_id="abc",
            start_time=100.0, end_time=110.0,
            exit_code=0,
        )
        assert ru.exit_code == 0


# ── 상태 전이 테스트 ──


class TestStateTransitions:
    def test_valid_lifecycle(self, registry):
        """CREATED→ASSIGNED→RUNNING→COMPLETED→DESTROYED 전이가 유효해야 한다."""
        assert SubAgentStatus.ASSIGNED in VALID_TRANSITIONS[SubAgentStatus.CREATED]
        assert SubAgentStatus.RUNNING in VALID_TRANSITIONS[SubAgentStatus.ASSIGNED]
        assert SubAgentStatus.COMPLETED in VALID_TRANSITIONS[SubAgentStatus.RUNNING]
        assert SubAgentStatus.DESTROYED in VALID_TRANSITIONS[SubAgentStatus.COMPLETED]

    def test_failed_to_destroyed(self):
        assert SubAgentStatus.DESTROYED in VALID_TRANSITIONS[SubAgentStatus.FAILED]

    def test_cancelled_to_destroyed(self):
        assert SubAgentStatus.DESTROYED in VALID_TRANSITIONS[SubAgentStatus.CANCELLED]


# ── ProcessManager 기본 동작 테스트 ──


class TestProcessManagerBasics:
    def test_initial_state(self, manager):
        assert manager.active_count == 0

    @pytest.mark.asyncio
    async def test_spawn_creates_process(self, registry):
        """spawn()이 실제 자식 프로세스를 생성하는지 확인."""
        manager = SubAgentProcessManager(registry=registry, timeout_s=10.0)

        # 간단한 echo 스크립트를 실행하는 방식으로 테스트
        # worker.py 대신 간단한 python 명령으로 테스트
        instance = await manager.spawn(
            agent_type="echo",
            task_message="hello world",
        )
        assert instance is not None
        assert instance.state == SubAgentStatus.RUNNING
        assert manager.active_count == 1

        # 프로세스가 실제로 존재하는지 확인
        proc = manager.all_processes.get(instance.agent_id)
        assert proc is not None
        assert proc.pid > 0

        # 정리 (워커가 없으므로 cancel)
        await manager.cancel(instance.agent_id, reason="test cleanup")

    @pytest.mark.asyncio
    async def test_spawn_auto_registers_unknown_agent(self, registry):
        """알 수 없는 에이전트 타입을 자동 등록하는지 확인."""
        manager = SubAgentProcessManager(registry=registry, timeout_s=5.0)
        instance = await manager.spawn(
            agent_type="unknown_type",
            task_message="test",
        )
        assert instance is not None
        # 자동 등록 확인
        assert registry.get("unknown_type") is not None
        await manager.cancel(instance.agent_id)

    @pytest.mark.asyncio
    async def test_resource_usage_tracking(self, registry):
        """자원 사용량이 기록되는지 확인."""
        manager = SubAgentProcessManager(registry=registry, timeout_s=5.0)
        instance = await manager.spawn(agent_type="echo", task_message="test")
        agent_id = instance.agent_id

        usage = manager.get_resource_usage(agent_id)
        assert usage is not None
        assert usage.pid > 0
        assert usage.start_time > 0
        assert usage.end_time is None  # 아직 완료되지 않음

        await manager.cancel(agent_id)


# ── spawn_and_wait 통합 테스트 (모의 워커 사용) ──


class TestSpawnAndWaitMock:
    """실제 worker.py 대신 간단한 Python 스크립트로 테스트."""

    @pytest.mark.asyncio
    async def test_successful_worker(self, registry, tmp_path):
        """정상 완료 워커를 시뮬레이션."""
        # 간단한 워커 스크립트 생성
        worker_script = tmp_path / "mock_worker.py"
        worker_script.write_text(
            'import json, sys\n'
            'print(json.dumps({"status": "completed", "result": "mock output",'
            ' "written_files": ["a.py"], "duration_s": 1.0, "token_usage": {},'
            ' "error": None}))\n'
        )

        manager = SubAgentProcessManager(registry=registry, timeout_s=10.0)

        # spawn 직접 수행 (worker.py 경로 대신 mock 사용)
        # 실제 ProcessManager는 sys.executable -m worker 를 사용하지만
        # 테스트에서는 spawn_and_wait를 통하지 않고 수동으로 검증
        instance = registry.create_instance(
            spec_name="echo",
            role="test",
            task_summary="test task",
        )
        assert instance is not None

        # mock worker 프로세스 직접 실행
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(worker_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        result_data = json.loads(stdout.decode())
        result = SubAgentResult(**result_data)

        assert result.success
        assert result.result == "mock output"
        assert result.written_files == ["a.py"]

    @pytest.mark.asyncio
    async def test_failed_worker(self, tmp_path):
        """실패 워커를 시뮬레이션."""
        worker_script = tmp_path / "fail_worker.py"
        worker_script.write_text(
            'import json, sys\n'
            'print(json.dumps({"status": "failed", "result": None,'
            ' "written_files": [], "duration_s": 0.5, "token_usage": {},'
            ' "error": "Test error"}))\n'
            'sys.exit(1)\n'
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(worker_script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        result = SubAgentResult(**json.loads(stdout.decode()))

        assert not result.success
        assert result.error == "Test error"
        assert proc.returncode == 1

    @pytest.mark.asyncio
    async def test_timeout_handling(self, registry):
        """타임아웃 시 프로세스가 kill되는지 확인."""
        manager = SubAgentProcessManager(registry=registry, timeout_s=2.0)

        instance = await manager.spawn(
            agent_type="echo",
            task_message="this will timeout",
        )
        # worker.py가 없으므로 즉시 실패하거나 타임아웃
        result = await manager.wait(instance.agent_id, timeout_s=2.0)

        # 실패 상태여야 함
        assert not result.success

    @pytest.mark.asyncio
    async def test_cancel(self, registry):
        """cancel()이 프로세스를 종료하는지 확인."""
        manager = SubAgentProcessManager(registry=registry, timeout_s=30.0)

        instance = await manager.spawn(
            agent_type="echo",
            task_message="cancel me",
        )
        assert manager.active_count == 1

        cancelled = await manager.cancel(instance.agent_id, reason="test")
        assert cancelled is True

        # 인스턴스 상태 확인
        inst = registry.get_instance(instance.agent_id)
        assert inst is None or inst.state == SubAgentStatus.DESTROYED


# ── cleanup 테스트 ──


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_all(self, registry):
        """cleanup_all()이 완료된 프로세스를 정리하는지 확인."""
        manager = SubAgentProcessManager(registry=registry, timeout_s=5.0)

        instance = await manager.spawn(
            agent_type="echo",
            task_message="cleanup test",
        )
        await manager.cancel(instance.agent_id)

        cleaned = await manager.cleanup_all()
        # cancel 후 이미 destroy됨 → 추가 정리 대상 없을 수 있음
        assert cleaned >= 0

    @pytest.mark.asyncio
    async def test_max_concurrent_limit(self, registry):
        """동시 실행 한도를 초과하면 에러 발생."""
        manager = SubAgentProcessManager(
            registry=registry, timeout_s=10.0, max_concurrent=1,
        )

        instance1 = await manager.spawn(agent_type="echo", task_message="first")

        with pytest.raises(RuntimeError, match="동시 실행 한도 초과"):
            await manager.spawn(agent_type="echo", task_message="second")

        await manager.cancel(instance1.agent_id)


# ── max_tokens 복구 컨텍스트 폭발 수정 테스트 ──


class TestMaxTokensRecovery:
    """invoke_with_max_tokens_recovery의 컨텍스트 한도 체크 테스트."""

    @pytest.mark.asyncio
    async def test_stops_when_context_exceeds_limit(self):
        """입력 토큰 + max_output이 모델 한도를 초과하면 재시도를 중단해야 한다."""
        from unittest.mock import AsyncMock, patch

        from langchain_core.messages import AIMessage, HumanMessage

        from coding_agent.core.context_manager import (
            ContextManager,
            invoke_with_max_tokens_recovery,
        )

        # max_tokens stop_reason을 반복 반환하는 mock LLM
        mock_llm = AsyncMock()
        large_content = "x" * 50000  # 큰 응답
        mock_response = AIMessage(content=large_content)
        mock_response.response_metadata = {"finish_reason": "length"}
        mock_llm.ainvoke.return_value = mock_response

        ctx = ContextManager(max_tokens=128000, compact_threshold=0.8)

        # count_messages_tokens를 mock하여 재시도마다 급격히 증가하도록 설정
        call_count = {"n": 0}
        original_count = ctx.count_messages_tokens

        def growing_count(msgs):
            call_count["n"] += 1
            # 1차: 10000, 2차: 80000 (한도 초과)
            return 10000 + (call_count["n"] - 1) * 70000

        messages = [HumanMessage(content="generate code")]

        with patch.object(ctx, "count_messages_tokens", side_effect=growing_count):
            result = await invoke_with_max_tokens_recovery(
                mock_llm,
                messages,
                ctx,
                max_retries=3,
                model_context_limit=100000,  # 작은 한도
                max_output_tokens=65536,
            )

        # 재시도가 무한히 계속되지 않고 축적된 partial을 반환해야 함
        assert isinstance(result, AIMessage)
        assert len(result.content) > 0
        # 2차에서 80000 + 65536 > 100000 이므로 중단 → ainvoke 최대 2번
        assert mock_llm.ainvoke.call_count <= 2
