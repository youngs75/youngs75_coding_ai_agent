"""Hook System 테스트.

HookManager 등록/해제, 이벤트 발행, 우선순위, 핸들러 체이닝,
취소 동작, 에러 격리, ParallelToolExecutor 연동, 동기/비동기 혼합을 검증한다.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from coding_agent.core.hooks import HookContext, HookEvent, HookManager
from coding_agent.core.builtin_hooks import audit_hook, logging_hook, timing_hook
from coding_agent.core.parallel_tool_executor import ParallelToolExecutor


# ─────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────


async def _noop_handler(ctx: HookContext) -> HookContext:
    """아무것도 하지 않는 핸들러."""
    return ctx


async def _mark_handler(ctx: HookContext) -> HookContext:
    """메타데이터에 마크를 남기는 핸들러."""
    ctx.metadata["marked"] = True
    return ctx


def _sync_mark_handler(ctx: HookContext) -> HookContext:
    """동기 마크 핸들러."""
    ctx.metadata["sync_marked"] = True
    return ctx


async def _append_handler_factory(label: str):
    """순서 추적용 핸들러 팩토리."""

    async def handler(ctx: HookContext) -> HookContext:
        order = ctx.metadata.setdefault("order", [])
        order.append(label)
        return ctx

    handler.__name__ = f"handler_{label}"
    return handler


async def _failing_handler(ctx: HookContext) -> HookContext:
    """항상 예외를 던지는 핸들러."""
    raise RuntimeError("핸들러 실패!")


async def _cancel_handler(ctx: HookContext) -> HookContext:
    """실행 취소를 설정하는 핸들러."""
    ctx.metadata["cancel"] = True
    return ctx


async def _modify_args_handler(ctx: HookContext) -> HookContext:
    """도구 인자를 수정하는 핸들러."""
    if ctx.tool_args is not None:
        ctx.tool_args["modified"] = True
    return ctx


async def _modify_result_handler(ctx: HookContext) -> HookContext:
    """도구 결과를 수정하는 핸들러."""
    if ctx.event == HookEvent.POST_TOOL_CALL:
        ctx.tool_result = f"modified:{ctx.tool_result}"
    return ctx


async def _mock_tool_executor(name: str, args: dict) -> str:
    """테스트용 도구 실행기."""
    return f"result:{name}:{args}"


def _make_tool_call(name: str, args: dict | None = None, call_id: str | None = None):
    """테스트용 도구 호출 dict 생성."""
    return {
        "name": name,
        "args": args or {},
        "id": call_id or f"call_{name}",
    }


# ─────────────────────────────────────────────────────────────
# 1. HookManager — 등록/해제
# ─────────────────────────────────────────────────────────────


class TestRegistration:
    """핸들러 등록 및 해제 테스트."""

    def test_register_returns_handler_id(self):
        """register()가 고유 handler_id를 반환한다."""
        manager = HookManager()
        hid = manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        assert isinstance(hid, str)
        assert len(hid) > 0

    def test_register_multiple_returns_unique_ids(self):
        """여러 핸들러 등록 시 고유 ID가 반환된다."""
        manager = HookManager()
        id1 = manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        id2 = manager.register(HookEvent.PRE_TOOL_CALL, _mark_handler)
        assert id1 != id2

    def test_unregister_removes_handler(self):
        """unregister()가 핸들러를 제거한다."""
        manager = HookManager()
        hid = manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        assert manager.get_handler_count(HookEvent.PRE_TOOL_CALL) == 1
        result = manager.unregister(hid)
        assert result is True
        assert manager.get_handler_count(HookEvent.PRE_TOOL_CALL) == 0

    def test_unregister_nonexistent_returns_false(self):
        """존재하지 않는 ID로 unregister() 시 False를 반환한다."""
        manager = HookManager()
        result = manager.unregister("nonexistent-id")
        assert result is False

    def test_get_handler_count_total(self):
        """get_handler_count(None)이 전체 핸들러 수를 반환한다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        manager.register(HookEvent.POST_TOOL_CALL, _noop_handler)
        manager.register(HookEvent.ON_ERROR, _noop_handler)
        assert manager.get_handler_count() == 3

    def test_get_handler_count_per_event(self):
        """get_handler_count(event)가 특정 이벤트의 핸들러 수를 반환한다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        manager.register(HookEvent.PRE_TOOL_CALL, _mark_handler)
        manager.register(HookEvent.POST_TOOL_CALL, _noop_handler)
        assert manager.get_handler_count(HookEvent.PRE_TOOL_CALL) == 2
        assert manager.get_handler_count(HookEvent.POST_TOOL_CALL) == 1

    def test_clear_specific_event(self):
        """clear(event)가 특정 이벤트의 핸들러만 제거한다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        manager.register(HookEvent.POST_TOOL_CALL, _noop_handler)
        manager.clear(HookEvent.PRE_TOOL_CALL)
        assert manager.get_handler_count(HookEvent.PRE_TOOL_CALL) == 0
        assert manager.get_handler_count(HookEvent.POST_TOOL_CALL) == 1

    def test_clear_all(self):
        """clear()가 모든 핸들러를 제거한다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _noop_handler)
        manager.register(HookEvent.POST_TOOL_CALL, _noop_handler)
        manager.register(HookEvent.ON_ERROR, _noop_handler)
        manager.clear()
        assert manager.get_handler_count() == 0


# ─────────────────────────────────────────────────────────────
# 2. HookManager — 이벤트 발행 + 핸들러 호출
# ─────────────────────────────────────────────────────────────


class TestEmit:
    """이벤트 발행 및 핸들러 호출 테스트."""

    async def test_emit_calls_registered_handler(self):
        """등록된 핸들러가 emit()에 의해 호출된다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _mark_handler)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="test_tool")
        result = await manager.emit(ctx)
        assert result.metadata.get("marked") is True

    async def test_emit_no_handlers_returns_context(self):
        """핸들러가 없으면 원래 context가 그대로 반환된다."""
        manager = HookManager()
        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="test_tool")
        result = await manager.emit(ctx)
        assert result is ctx

    async def test_emit_only_calls_matching_event(self):
        """다른 이벤트의 핸들러는 호출되지 않는다."""
        manager = HookManager()
        manager.register(HookEvent.POST_TOOL_CALL, _mark_handler)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="test_tool")
        result = await manager.emit(ctx)
        assert "marked" not in result.metadata

    async def test_emit_returns_handler_none_keeps_context(self):
        """핸들러가 None을 반환하면 원래 context를 유지한다."""

        async def return_none_handler(ctx: HookContext) -> None:
            ctx.metadata["visited"] = True
            return None

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, return_none_handler)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        # None 반환이므로 원래 ctx 유지 (하지만 in-place 수정은 반영됨)
        assert result.metadata.get("visited") is True


# ─────────────────────────────────────────────────────────────
# 3. 우선순위 순서 검증
# ─────────────────────────────────────────────────────────────


class TestPriority:
    """핸들러 우선순위 실행 순서 테스트."""

    async def test_lower_priority_runs_first(self):
        """낮은 우선순위 숫자의 핸들러가 먼저 실행된다."""
        manager = HookManager()

        handler_c = await _append_handler_factory("C")
        handler_a = await _append_handler_factory("A")
        handler_b = await _append_handler_factory("B")

        manager.register(HookEvent.PRE_TOOL_CALL, handler_c, priority=20)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_a, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_b, priority=10)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata["order"] == ["A", "B", "C"]

    async def test_same_priority_preserves_registration_order(self):
        """동일 우선순위의 핸들러는 등록 순서대로 실행된다."""
        manager = HookManager()

        handler_x = await _append_handler_factory("X")
        handler_y = await _append_handler_factory("Y")
        handler_z = await _append_handler_factory("Z")

        manager.register(HookEvent.PRE_TOOL_CALL, handler_x, priority=5)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_y, priority=5)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_z, priority=5)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata["order"] == ["X", "Y", "Z"]

    async def test_negative_priority(self):
        """음수 우선순위가 양수보다 먼저 실행된다."""
        manager = HookManager()

        handler_first = await _append_handler_factory("first")
        handler_second = await _append_handler_factory("second")

        manager.register(HookEvent.PRE_TOOL_CALL, handler_second, priority=1)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_first, priority=-10)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata["order"] == ["first", "second"]


# ─────────────────────────────────────────────────────────────
# 4. 핸들러 체이닝 (context 수정)
# ─────────────────────────────────────────────────────────────


class TestChaining:
    """핸들러 체이닝 — context 수정 전파 테스트."""

    async def test_args_modification_propagates(self):
        """앞 핸들러의 args 수정이 뒤 핸들러에 전파된다."""

        async def add_key_handler(ctx: HookContext) -> HookContext:
            if ctx.tool_args is not None:
                ctx.tool_args["added_by_first"] = True
            return ctx

        async def check_key_handler(ctx: HookContext) -> HookContext:
            if ctx.tool_args and ctx.tool_args.get("added_by_first"):
                ctx.metadata["chaining_works"] = True
            return ctx

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, add_key_handler, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, check_key_handler, priority=10)

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name="test",
            tool_args={"original": "value"},
        )
        result = await manager.emit(ctx)
        assert result.metadata["chaining_works"] is True
        assert result.tool_args["added_by_first"] is True
        assert result.tool_args["original"] == "value"

    async def test_metadata_accumulation(self):
        """여러 핸들러가 metadata를 누적한다."""

        async def handler_a(ctx: HookContext) -> HookContext:
            ctx.metadata["a"] = 1
            return ctx

        async def handler_b(ctx: HookContext) -> HookContext:
            ctx.metadata["b"] = 2
            return ctx

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, handler_a, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_b, priority=1)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata == {"a": 1, "b": 2}


# ─────────────────────────────────────────────────────────────
# 5. 취소 동작
# ─────────────────────────────────────────────────────────────


class TestCancellation:
    """pre 훅에서 cancel=True 설정 시 동작 테스트."""

    async def test_cancel_stops_propagation_to_subsequent_action(self):
        """cancel 설정이 context에 전파된다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _cancel_handler, priority=0)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="test")
        result = await manager.emit(ctx)
        assert result.metadata.get("cancel") is True

    async def test_cancel_in_early_handler_still_runs_later_hooks(self):
        """cancel이 설정되어도 나머지 훅은 계속 실행된다 (훅 레벨에서는 체이닝)."""
        manager = HookManager()

        handler_after = await _append_handler_factory("after_cancel")
        manager.register(HookEvent.PRE_TOOL_CALL, _cancel_handler, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_after, priority=10)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="test")
        result = await manager.emit(ctx)
        assert result.metadata.get("cancel") is True
        # cancel 이후의 핸들러도 실행됨 (cancel은 실제 실행만 막음)
        assert "after_cancel" in result.metadata.get("order", [])


# ─────────────────────────────────────────────────────────────
# 6. 에러 격리
# ─────────────────────────────────────────────────────────────


class TestErrorIsolation:
    """핸들러 예외 시 에러 격리 테스트."""

    async def test_failing_handler_does_not_stop_others(self):
        """핸들러 예외가 후속 핸들러 실행을 막지 않는다."""
        manager = HookManager()

        handler_after = await _append_handler_factory("after_error")
        manager.register(HookEvent.PRE_TOOL_CALL, _failing_handler, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, handler_after, priority=10)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        # 실패 핸들러 이후에도 다음 핸들러가 실행됨
        assert "after_error" in result.metadata.get("order", [])

    async def test_failing_handler_does_not_raise(self):
        """핸들러 예외가 emit() 자체를 실패시키지 않는다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _failing_handler)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        # 예외가 발생하지 않아야 함
        result = await manager.emit(ctx)
        assert result is ctx

    async def test_multiple_failing_handlers(self):
        """여러 핸들러가 모두 실패해도 계속 진행한다."""

        async def fail_a(ctx: HookContext) -> HookContext:
            raise ValueError("A 실패")

        async def fail_b(ctx: HookContext) -> HookContext:
            raise TypeError("B 실패")

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, fail_a, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, fail_b, priority=1)
        manager.register(HookEvent.PRE_TOOL_CALL, _mark_handler, priority=2)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        # 마지막 핸들러까지 실행됨
        assert result.metadata.get("marked") is True


# ─────────────────────────────────────────────────────────────
# 7. 비동기/동기 핸들러 혼합
# ─────────────────────────────────────────────────────────────


class TestAsyncSyncMix:
    """동기/비동기 핸들러 혼합 실행 테스트."""

    async def test_sync_handler_works(self):
        """동기 핸들러가 정상 호출된다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _sync_mark_handler)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata.get("sync_marked") is True

    async def test_mixed_sync_async_handlers(self):
        """동기/비동기 핸들러가 혼합되어도 정상 동작한다."""
        manager = HookManager()

        async def async_handler(ctx: HookContext) -> HookContext:
            ctx.metadata["async_called"] = True
            return ctx

        def sync_handler(ctx: HookContext) -> HookContext:
            ctx.metadata["sync_called"] = True
            return ctx

        manager.register(HookEvent.PRE_TOOL_CALL, sync_handler, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, async_handler, priority=1)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata["sync_called"] is True
        assert result.metadata["async_called"] is True

    async def test_sync_handler_returning_none(self):
        """동기 핸들러가 None을 반환해도 정상 동작한다."""

        def sync_none_handler(ctx: HookContext) -> None:
            ctx.metadata["sync_none"] = True

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, sync_none_handler)

        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        result = await manager.emit(ctx)
        assert result.metadata.get("sync_none") is True


# ─────────────────────────────────────────────────────────────
# 8. ParallelToolExecutor 훅 연동
# ─────────────────────────────────────────────────────────────


class TestParallelExecutorHookIntegration:
    """ParallelToolExecutor와 훅 시스템 연동 테스트."""

    async def test_pre_hook_called_before_execution(self):
        """PRE_TOOL_CALL 훅이 도구 실행 전에 호출된다."""
        call_log: list[str] = []

        async def pre_logger(ctx: HookContext) -> HookContext:
            call_log.append(f"pre:{ctx.tool_name}")
            return ctx

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, pre_logger)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("read_file", {"path": "test.py"})]
        await executor.execute_batch(calls, _mock_tool_executor)

        assert "pre:read_file" in call_log

    async def test_post_hook_called_after_execution(self):
        """POST_TOOL_CALL 훅이 도구 실행 후에 호출된다."""
        call_log: list[str] = []

        async def post_logger(ctx: HookContext) -> HookContext:
            call_log.append(f"post:{ctx.tool_name}:{ctx.tool_result}")
            return ctx

        manager = HookManager()
        manager.register(HookEvent.POST_TOOL_CALL, post_logger)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("read_file", {"path": "test.py"})]
        await executor.execute_batch(calls, _mock_tool_executor)

        assert len(call_log) == 1
        assert call_log[0].startswith("post:read_file:")

    async def test_pre_hook_modifies_args(self):
        """PRE_TOOL_CALL 훅에서 수정한 인자가 실제 실행에 반영된다."""
        actual_args: list[dict] = []

        async def capture_executor(name: str, args: dict) -> str:
            actual_args.append(dict(args))
            return f"ok:{name}"

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _modify_args_handler)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("write_file", {"path": "a.py"})]
        await executor.execute_batch(calls, capture_executor)

        assert len(actual_args) == 1
        assert actual_args[0]["modified"] is True
        assert actual_args[0]["path"] == "a.py"

    async def test_cancel_in_pre_hook_skips_execution(self):
        """PRE_TOOL_CALL 훅에서 cancel 시 도구가 실행되지 않는다."""
        executed: list[str] = []

        async def tracking_executor(name: str, args: dict) -> str:
            executed.append(name)
            return f"result:{name}"

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _cancel_handler)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("write_file", {"path": "x.py"})]
        results = await executor.execute_batch(calls, tracking_executor)

        # 도구가 실행되지 않아야 함
        assert len(executed) == 0
        # 취소 메시지가 포함되어야 함
        assert "취소" in results[0].content

    async def test_cancel_returns_cancelled_result(self):
        """취소된 도구의 ToolExecutionResult에 cancelled=True가 설정된다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, _cancel_handler)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("write_file", {})]
        batch = await executor.execute_batch_detailed(calls, _mock_tool_executor)

        assert len(batch.results) == 1
        assert batch.results[0].cancelled is True

    async def test_post_hook_modifies_result(self):
        """POST_TOOL_CALL 훅에서 수정한 결과가 반영된다."""
        manager = HookManager()
        manager.register(HookEvent.POST_TOOL_CALL, _modify_result_handler)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("read_file", {"path": "test.py"})]
        results = await executor.execute_batch(calls, _mock_tool_executor)

        assert results[0].content.startswith("modified:")

    async def test_hooks_with_parallel_execution(self):
        """병렬 실행 도구에도 훅이 정상 동작한다."""
        pre_calls: list[str] = []
        post_calls: list[str] = []

        async def pre_tracker(ctx: HookContext) -> HookContext:
            pre_calls.append(ctx.tool_name or "unknown")
            return ctx

        async def post_tracker(ctx: HookContext) -> HookContext:
            post_calls.append(ctx.tool_name or "unknown")
            return ctx

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, pre_tracker)
        manager.register(HookEvent.POST_TOOL_CALL, post_tracker)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [
            _make_tool_call("read_file", {"path": "a.py"}),
            _make_tool_call("search_code", {"pattern": "foo"}),
            _make_tool_call("list_directory", {"path": "."}),
        ]
        results = await executor.execute_batch(calls, _mock_tool_executor)

        assert len(results) == 3
        assert sorted(pre_calls) == ["list_directory", "read_file", "search_code"]
        assert sorted(post_calls) == ["list_directory", "read_file", "search_code"]

    async def test_default_hook_manager_created(self):
        """hook_manager를 지정하지 않으면 기본 HookManager가 생성된다."""
        executor = ParallelToolExecutor()
        assert isinstance(executor.hook_manager, HookManager)

    async def test_pre_hook_metadata_passed_to_post_hook(self):
        """PRE 훅에서 설정한 metadata가 POST 훅으로 전달된다."""
        captured_metadata: list[dict] = []

        async def pre_set_meta(ctx: HookContext) -> HookContext:
            ctx.metadata["trace_id"] = "abc123"
            return ctx

        async def post_check_meta(ctx: HookContext) -> HookContext:
            captured_metadata.append(dict(ctx.metadata))
            return ctx

        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, pre_set_meta)
        manager.register(HookEvent.POST_TOOL_CALL, post_check_meta)

        executor = ParallelToolExecutor(hook_manager=manager)
        calls = [_make_tool_call("read_file", {})]
        await executor.execute_batch(calls, _mock_tool_executor)

        assert len(captured_metadata) == 1
        assert captured_metadata[0].get("trace_id") == "abc123"


# ─────────────────────────────────────────────────────────────
# 9. BaseGraphAgent 훅 통합
# ─────────────────────────────────────────────────────────────


class TestBaseAgentHookIntegration:
    """BaseGraphAgent 훅 통합 테스트."""

    def test_base_agent_has_hook_manager(self):
        """BaseGraphAgent가 hook_manager 속성을 가진다."""
        from coding_agent.core.base_agent import BaseGraphAgent

        # auto_build=False (state_schema 없이 빌드 불가)
        agent = BaseGraphAgent(auto_build=False)
        assert isinstance(agent.hook_manager, HookManager)

    def test_base_agent_custom_hook_manager(self):
        """커스텀 HookManager를 전달할 수 있다."""
        from coding_agent.core.base_agent import BaseGraphAgent

        custom_manager = HookManager()
        custom_manager.register(HookEvent.PRE_NODE, _noop_handler)

        agent = BaseGraphAgent(auto_build=False, hook_manager=custom_manager)
        assert agent.hook_manager is custom_manager
        assert agent.hook_manager.get_handler_count(HookEvent.PRE_NODE) == 1

    async def test_wrap_node_emits_pre_and_post(self):
        """_wrap_node가 PRE_NODE와 POST_NODE 훅을 발행한다."""
        from coding_agent.core.base_agent import BaseGraphAgent

        events_emitted: list[str] = []

        async def track_events(ctx: HookContext) -> HookContext:
            events_emitted.append(f"{ctx.event.value}:{ctx.node_name}")
            return ctx

        manager = HookManager()
        manager.register(HookEvent.PRE_NODE, track_events)
        manager.register(HookEvent.POST_NODE, track_events)

        agent = BaseGraphAgent(auto_build=False, hook_manager=manager)

        def my_node(state):
            return {"result": "ok"}

        result = await agent._wrap_node("parse", my_node, {"input": "test"})
        assert result == {"result": "ok"}
        assert "pre_node:parse" in events_emitted
        assert "post_node:parse" in events_emitted

    async def test_wrap_node_cancel_skips_execution(self):
        """PRE_NODE 훅에서 cancel 시 노드가 실행되지 않는다."""
        from coding_agent.core.base_agent import BaseGraphAgent

        executed = False

        def my_node(state):
            nonlocal executed
            executed = True
            return {"result": "should_not_reach"}

        manager = HookManager()
        manager.register(HookEvent.PRE_NODE, _cancel_handler)

        agent = BaseGraphAgent(auto_build=False, hook_manager=manager)
        result = await agent._wrap_node("execute", my_node, {"input": "test"})

        assert executed is False
        assert result == {"input": "test"}

    async def test_wrap_node_error_emits_on_error(self):
        """노드에서 예외 발생 시 ON_ERROR 훅이 발행된다."""
        from coding_agent.core.base_agent import BaseGraphAgent

        error_captured: list[Exception] = []

        async def error_handler(ctx: HookContext) -> HookContext:
            if ctx.error:
                error_captured.append(ctx.error)
            return ctx

        manager = HookManager()
        manager.register(HookEvent.ON_ERROR, error_handler)

        agent = BaseGraphAgent(auto_build=False, hook_manager=manager)

        def failing_node(state):
            raise ValueError("노드 실패!")

        with pytest.raises(ValueError, match="노드 실패"):
            await agent._wrap_node("verify", failing_node, {"input": "test"})

        assert len(error_captured) == 1
        assert str(error_captured[0]) == "노드 실패!"

    async def test_wrap_node_async_func(self):
        """비동기 노드 함수도 _wrap_node로 감쌀 수 있다."""
        from coding_agent.core.base_agent import BaseGraphAgent

        agent = BaseGraphAgent(auto_build=False)

        async def async_node(state):
            await asyncio.sleep(0.01)
            return {"async_result": True}

        result = await agent._wrap_node("async_parse", async_node, {})
        assert result == {"async_result": True}


# ─────────────────────────────────────────────────────────────
# 10. 내장 훅 테스트
# ─────────────────────────────────────────────────────────────


class TestBuiltinHooks:
    """내장 훅 (logging, timing, audit) 테스트."""

    async def test_logging_hook_pre_tool(self):
        """logging_hook이 PRE_TOOL_CALL에서 정상 동작한다."""
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name="read_file",
            tool_args={"path": "test.py"},
        )
        result = await logging_hook(ctx)
        assert result is ctx  # context가 그대로 반환됨

    async def test_logging_hook_post_tool(self):
        """logging_hook이 POST_TOOL_CALL에서 정상 동작한다."""
        ctx = HookContext(
            event=HookEvent.POST_TOOL_CALL,
            tool_name="read_file",
            tool_result="file content here",
        )
        result = await logging_hook(ctx)
        assert result is ctx

    async def test_timing_hook_measures_duration(self):
        """timing_hook이 PRE/POST 사이 소요 시간을 측정한다."""
        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="test")
        ctx = await timing_hook(ctx)
        assert "start_time" in ctx.metadata

        # 약간의 시간 경과
        await asyncio.sleep(0.05)

        ctx.event = HookEvent.POST_TOOL_CALL
        ctx = await timing_hook(ctx)
        assert "duration_s" in ctx.metadata
        assert ctx.metadata["duration_s"] >= 0.04

    async def test_audit_hook_logs_sensitive_tool(self):
        """audit_hook이 민감 도구 사용을 기록한다."""
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name="bash",
            tool_args={"command": "rm -rf /"},
        )
        result = await audit_hook(ctx)
        assert result.metadata.get("audit_logged") is True

    async def test_audit_hook_ignores_safe_tool(self):
        """audit_hook이 안전한 도구는 무시한다."""
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name="read_file",
            tool_args={"path": "test.py"},
        )
        result = await audit_hook(ctx)
        assert "audit_logged" not in result.metadata

    async def test_builtin_hooks_integration_with_manager(self):
        """내장 훅이 HookManager에 등록되어 정상 동작한다."""
        manager = HookManager()
        manager.register(HookEvent.PRE_TOOL_CALL, logging_hook, priority=0)
        manager.register(HookEvent.PRE_TOOL_CALL, timing_hook, priority=10)
        manager.register(HookEvent.PRE_TOOL_CALL, audit_hook, priority=20)

        ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name="bash",
            tool_args={"command": "ls"},
        )
        result = await manager.emit(ctx)
        assert "start_time" in result.metadata
        assert result.metadata.get("audit_logged") is True


# ─────────────────────────────────────────────────────────────
# 11. 설정 파일 로드
# ─────────────────────────────────────────────────────────────


class TestConfigLoading:
    """설정 파일 기반 훅 로딩 테스트."""

    def test_load_from_nonexistent_file(self):
        """존재하지 않는 파일 경로에서 로드 시 에러 없이 진행한다."""
        from pathlib import Path

        manager = HookManager()
        manager.load_from_config(Path("/nonexistent/hooks.yaml"))
        assert manager.get_handler_count() == 0

    def test_load_from_json_config(self, tmp_path):
        """JSON 설정 파일에서 훅을 로드한다."""
        config = {
            "hooks": [
                {
                    "event": "pre_tool_call",
                    "handler": "coding_agent.core.builtin_hooks.logging_hook",
                    "priority": 0,
                },
                {
                    "event": "post_tool_call",
                    "handler": "coding_agent.core.builtin_hooks.timing_hook",
                    "priority": 5,
                },
            ]
        }
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        manager = HookManager()
        manager.load_from_config(config_path)
        assert manager.get_handler_count(HookEvent.PRE_TOOL_CALL) == 1
        assert manager.get_handler_count(HookEvent.POST_TOOL_CALL) == 1

    def test_load_from_json_with_invalid_handler(self, tmp_path):
        """존재하지 않는 핸들러 경로는 무시한다."""
        config = {
            "hooks": [
                {
                    "event": "pre_tool_call",
                    "handler": "nonexistent.module.handler",
                    "priority": 0,
                },
            ]
        }
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        manager = HookManager()
        manager.load_from_config(config_path)
        assert manager.get_handler_count() == 0

    def test_load_from_json_with_invalid_event(self, tmp_path):
        """잘못된 이벤트 타입은 무시한다."""
        config = {
            "hooks": [
                {
                    "event": "invalid_event",
                    "handler": "coding_agent.core.builtin_hooks.logging_hook",
                    "priority": 0,
                },
            ]
        }
        config_path = tmp_path / "hooks.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        manager = HookManager()
        manager.load_from_config(config_path)
        assert manager.get_handler_count() == 0


# ─────────────────────────────────────────────────────────────
# 12. HookContext 데이터 모델
# ─────────────────────────────────────────────────────────────


class TestHookContext:
    """HookContext 데이터 모델 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정된다."""
        ctx = HookContext(event=HookEvent.PRE_TOOL_CALL)
        assert ctx.tool_name is None
        assert ctx.tool_args is None
        assert ctx.tool_result is None
        assert ctx.node_name is None
        assert ctx.state is None
        assert ctx.error is None
        assert ctx.metadata == {}

    def test_metadata_is_independent_per_instance(self):
        """각 인스턴스의 metadata가 독립적이다."""
        ctx1 = HookContext(event=HookEvent.PRE_TOOL_CALL)
        ctx2 = HookContext(event=HookEvent.POST_TOOL_CALL)
        ctx1.metadata["key"] = "value"
        assert "key" not in ctx2.metadata

    def test_all_event_types_valid(self):
        """모든 HookEvent 값이 유효하다."""
        expected_events = {
            "pre_tool_call",
            "post_tool_call",
            "pre_node",
            "post_node",
            "on_error",
            "pre_llm_call",
            "post_llm_call",
        }
        actual_events = {e.value for e in HookEvent}
        assert actual_events == expected_events
