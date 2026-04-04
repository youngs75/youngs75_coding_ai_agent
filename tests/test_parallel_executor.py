"""ParallelToolExecutor + apply_patch + str_replace 테스트.

최소 20개 테스트를 포함:
- 병렬 실행기: 분류, 병렬/순차/혼합 실행, 순서 보장, 에러 처리, 타임아웃
- apply_patch: 정상 패치, workspace 밖 차단, 잘못된 diff
- str_replace: 정상 교체, 미발견, 다중 매치, workspace 밖 차단
"""

from __future__ import annotations

import asyncio
import time

import pytest

from youngs75_a2a.core.parallel_tool_executor import (
    CONCURRENCY_SAFE_TOOLS,
    BatchToolResult,
    ParallelToolExecutor,
)


# ─────────────────────────────────────────────────────────────
# 헬퍼: 가짜 도구 실행기
# ─────────────────────────────────────────────────────────────


async def _mock_tool_executor(name: str, args: dict) -> str:
    """테스트용 도구 실행기 — 이름과 인자를 반환."""
    return f"result:{name}:{args}"


async def _slow_tool_executor(name: str, args: dict) -> str:
    """지연이 있는 테스트용 도구 실행기."""
    delay = args.get("delay", 0.1)
    await asyncio.sleep(delay)
    return f"slow_result:{name}"


async def _failing_tool_executor(name: str, args: dict) -> str:
    """항상 예외를 던지는 도구 실행기."""
    raise RuntimeError(f"도구 실행 실패: {name}")


def _make_tool_call(name: str, args: dict | None = None, call_id: str | None = None):
    """테스트용 도구 호출 dict 생성."""
    return {
        "name": name,
        "args": args or {},
        "id": call_id or f"call_{name}",
    }


# ─────────────────────────────────────────────────────────────
# 1. ParallelToolExecutor — 분류 테스트
# ─────────────────────────────────────────────────────────────


class TestClassification:
    """도구 호출 분류 테스트."""

    def test_safe_tool_classified_correctly(self):
        """읽기 전용 도구가 safe로 분류된다."""
        executor = ParallelToolExecutor()
        for tool_name in ["read_file", "search_code", "list_directory"]:
            assert executor.is_concurrency_safe(tool_name), f"{tool_name}은 safe여야 함"

    def test_unsafe_tool_classified_correctly(self):
        """쓰기 도구가 non-concurrent로 분류된다."""
        executor = ParallelToolExecutor()
        for tool_name in ["write_file", "run_python", "apply_patch", "str_replace"]:
            assert not executor.is_concurrency_safe(tool_name), (
                f"{tool_name}은 non-concurrent여야 함"
            )

    def test_none_tool_name_is_unsafe(self):
        """None 도구 이름은 non-concurrent로 분류된다."""
        executor = ParallelToolExecutor()
        assert not executor.is_concurrency_safe(None)

    def test_classify_tool_calls_splits_correctly(self):
        """도구 호출 리스트가 safe/non-concurrent로 올바르게 분리된다."""
        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("read_file", {"path": "a.py"}),
            _make_tool_call("write_file", {"path": "b.py", "content": "x"}),
            _make_tool_call("search_code", {"pattern": "foo"}),
            _make_tool_call("run_python", {"code": "print(1)"}),
        ]
        safe, non_concurrent = executor.classify_tool_calls(calls)
        assert len(safe) == 2  # read_file, search_code
        assert len(non_concurrent) == 2  # write_file, run_python

    def test_custom_safe_tools(self):
        """커스텀 safe 도구 집합으로 분류가 바뀐다."""
        executor = ParallelToolExecutor(concurrency_safe_tools={"my_tool"})
        assert executor.is_concurrency_safe("my_tool")
        assert not executor.is_concurrency_safe("read_file")

    def test_default_safe_tools_set(self):
        """기본 CONCURRENCY_SAFE_TOOLS 상수가 올바른 도구를 포함한다."""
        expected = {
            "read_file",
            "search_code",
            "list_directory",
            "search_web",
            "search_papers",
            "search_recent_papers",
        }
        assert CONCURRENCY_SAFE_TOOLS == expected


# ─────────────────────────────────────────────────────────────
# 2. ParallelToolExecutor — 실행 테스트
# ─────────────────────────────────────────────────────────────


class TestExecution:
    """도구 실행 동작 테스트."""

    async def test_empty_batch_returns_empty(self):
        """빈 tool_calls는 빈 리스트를 반환한다."""
        executor = ParallelToolExecutor()
        results = await executor.execute_batch([], _mock_tool_executor)
        assert results == []

    async def test_single_safe_tool(self):
        """safe 도구 1개가 정상 실행된다."""
        executor = ParallelToolExecutor()
        calls = [_make_tool_call("read_file", {"path": "test.py"})]
        results = await executor.execute_batch(calls, _mock_tool_executor)
        assert len(results) == 1
        assert "result:read_file" in results[0].content

    async def test_single_unsafe_tool(self):
        """non-concurrent 도구 1개가 정상 실행된다."""
        executor = ParallelToolExecutor()
        calls = [_make_tool_call("write_file", {"path": "x.py", "content": "y"})]
        results = await executor.execute_batch(calls, _mock_tool_executor)
        assert len(results) == 1
        assert "result:write_file" in results[0].content

    async def test_parallel_execution_is_faster(self):
        """safe 도구 2개가 병렬 실행되면 순차보다 빠르다."""
        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("read_file", {"delay": 0.15}),
            _make_tool_call("search_code", {"delay": 0.15}),
        ]
        start = time.perf_counter()
        results = await executor.execute_batch(calls, _slow_tool_executor)
        elapsed = time.perf_counter() - start

        assert len(results) == 2
        # 병렬이면 ~0.15s, 순차면 ~0.30s — 0.25s 미만이면 병렬
        assert elapsed < 0.25, f"병렬 실행이 예상보다 느림: {elapsed:.2f}s"

    async def test_sequential_execution_for_unsafe(self):
        """non-concurrent 도구 2개는 순차 실행된다."""
        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("write_file", {"delay": 0.1}),
            _make_tool_call("run_python", {"delay": 0.1}),
        ]
        start = time.perf_counter()
        results = await executor.execute_batch(calls, _slow_tool_executor)
        elapsed = time.perf_counter() - start

        assert len(results) == 2
        # 순차 실행이면 ~0.2s 이상
        assert elapsed >= 0.18, f"순차 실행이 예상보다 빠름: {elapsed:.2f}s"

    async def test_mixed_execution_order_preserved(self):
        """safe + non-concurrent 혼합 시 결과가 원래 순서대로 반환된다."""
        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("read_file", {"path": "1"}),  # 0: safe
            _make_tool_call("write_file", {"path": "2"}),  # 1: unsafe
            _make_tool_call("search_code", {"pattern": "3"}),  # 2: safe
            _make_tool_call("run_python", {"code": "4"}),  # 3: unsafe
            _make_tool_call("list_directory", {"path": "5"}),  # 4: safe
        ]
        results = await executor.execute_batch(calls, _mock_tool_executor)
        assert len(results) == 5

        # 결과 순서가 원래 호출 순서와 일치하는지 확인
        assert "read_file" in results[0].content
        assert "write_file" in results[1].content
        assert "search_code" in results[2].content
        assert "run_python" in results[3].content
        assert "list_directory" in results[4].content

    async def test_tool_call_id_preserved(self):
        """tool_call_id가 결과에 올바르게 전달된다."""
        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("read_file", {}, call_id="my_unique_id_123"),
        ]
        results = await executor.execute_batch(calls, _mock_tool_executor)
        assert results[0].tool_call_id == "my_unique_id_123"

    async def test_tool_failure_returns_error_message(self):
        """도구 실행 실패 시 에러 메시지가 ToolMessage에 포함된다."""
        executor = ParallelToolExecutor()
        calls = [_make_tool_call("read_file", {})]
        results = await executor.execute_batch(calls, _failing_tool_executor)
        assert len(results) == 1
        assert "도구 실행 오류" in results[0].content

    async def test_partial_failure_in_batch(self):
        """배치에서 일부 실패해도 나머지는 성공한다."""
        call_count = 0

        async def _partial_fail_executor(name: str, args: dict) -> str:
            nonlocal call_count
            call_count += 1
            if name == "search_code":
                raise RuntimeError("검색 실패")
            return f"ok:{name}"

        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("read_file", {}),
            _make_tool_call("search_code", {}),
            _make_tool_call("list_directory", {}),
        ]
        results = await executor.execute_batch(calls, _partial_fail_executor)
        assert len(results) == 3
        assert "ok:read_file" in results[0].content
        assert "도구 실행 오류" in results[1].content
        assert "ok:list_directory" in results[2].content

    async def test_timeout_handling(self):
        """타임아웃 초과 시 에러 메시지가 반환된다."""
        executor = ParallelToolExecutor(timeout_s=0.05)

        async def _very_slow(name: str, args: dict) -> str:
            await asyncio.sleep(5)
            return "should not reach"

        calls = [_make_tool_call("read_file", {})]
        results = await executor.execute_batch(calls, _very_slow)
        assert len(results) == 1
        assert "도구 실행 오류" in results[0].content

    async def test_detailed_batch_result(self):
        """execute_batch_detailed가 상세 결과를 반환한다."""
        executor = ParallelToolExecutor()
        calls = [
            _make_tool_call("read_file", {}),  # safe
            _make_tool_call("write_file", {}),  # unsafe
        ]
        batch_result = await executor.execute_batch_detailed(calls, _mock_tool_executor)
        assert isinstance(batch_result, BatchToolResult)
        assert batch_result.parallel_count == 1
        assert batch_result.sequential_count == 1
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 0
        assert len(batch_result.tool_messages) == 2

    async def test_detailed_empty_returns_empty_result(self):
        """빈 호출에 대해 execute_batch_detailed가 빈 BatchToolResult를 반환한다."""
        executor = ParallelToolExecutor()
        batch_result = await executor.execute_batch_detailed([], _mock_tool_executor)
        assert isinstance(batch_result, BatchToolResult)
        assert len(batch_result.results) == 0


# ─────────────────────────────────────────────────────────────
# 3. apply_patch MCP 도구 테스트
# ─────────────────────────────────────────────────────────────


class TestApplyPatch:
    """apply_patch MCP 도구 테스트."""

    @pytest.fixture(autouse=True)
    def _setup_workspace(self, tmp_path, monkeypatch):
        """테스트용 workspace를 임시 디렉토리로 설정한다."""
        self.workspace = tmp_path
        monkeypatch.setattr(
            "youngs75_a2a.mcp_servers.code_tools.server._WORKSPACE", str(tmp_path)
        )

    def test_apply_patch_normal(self):
        """정상적인 unified diff 패치가 올바르게 적용된다."""
        from youngs75_a2a.mcp_servers.code_tools.server import apply_patch

        # 원본 파일 생성
        target = self.workspace / "hello.py"
        target.write_text("line1\nold_line\nline3\n", encoding="utf-8")

        patch = (
            "--- a/hello.py\n"
            "+++ b/hello.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-old_line\n"
            "+new_line\n"
            " line3\n"
        )
        result = apply_patch(patch)
        assert "✓" in result or "OK" in result
        content = target.read_text(encoding="utf-8")
        assert "new_line" in content
        assert "old_line" not in content

    def test_apply_patch_add_lines(self):
        """패치로 새 줄을 추가할 수 있다."""
        from youngs75_a2a.mcp_servers.code_tools.server import apply_patch

        target = self.workspace / "add.py"
        target.write_text("line1\nline2\n", encoding="utf-8")

        patch = (
            "--- a/add.py\n"
            "+++ b/add.py\n"
            "@@ -1,2 +1,4 @@\n"
            " line1\n"
            "+added_a\n"
            "+added_b\n"
            " line2\n"
        )
        result = apply_patch(patch)
        assert "✓" in result or "OK" in result
        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines == ["line1", "added_a", "added_b", "line2"]

    def test_apply_patch_workspace_outside_blocked(self):
        """workspace 밖 경로 접근이 차단된다."""
        from youngs75_a2a.mcp_servers.code_tools.server import apply_patch

        patch = (
            "--- a/../../../etc/passwd\n"
            "+++ b/../../../etc/passwd\n"
            "@@ -1,1 +1,1 @@\n"
            "-root\n"
            "+hacked\n"
        )
        with pytest.raises(ValueError, match="접근 거부"):
            apply_patch(patch)

    def test_apply_patch_invalid_format(self):
        """잘못된 diff 형식에 대해 에러를 반환한다."""
        from youngs75_a2a.mcp_servers.code_tools.server import apply_patch

        result = apply_patch("이건 유효한 패치가 아닙니다")
        assert "Error" in result
        assert "대상 파일" in result or "+++ " in result

    def test_apply_patch_no_hunk(self):
        """hunk 헤더가 없는 패치는 에러를 반환한다."""
        from youngs75_a2a.mcp_servers.code_tools.server import apply_patch

        target = self.workspace / "nohunk.py"
        target.write_text("hello\n", encoding="utf-8")

        patch = "--- a/nohunk.py\n+++ b/nohunk.py\n"
        result = apply_patch(patch)
        assert "Error" in result


# ─────────────────────────────────────────────────────────────
# 4. str_replace MCP 도구 테스트
# ─────────────────────────────────────────────────────────────


class TestStrReplace:
    """str_replace MCP 도구 테스트."""

    @pytest.fixture(autouse=True)
    def _setup_workspace(self, tmp_path, monkeypatch):
        """테스트용 workspace를 임시 디렉토리로 설정한다."""
        self.workspace = tmp_path
        monkeypatch.setattr(
            "youngs75_a2a.mcp_servers.code_tools.server._WORKSPACE", str(tmp_path)
        )

    def test_str_replace_normal(self):
        """정상적인 문자열 교체가 동작한다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        target = self.workspace / "example.py"
        target.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

        result = str_replace("example.py", "return 'world'", "return 'universe'")
        assert "✓" in result or "OK" in result
        content = target.read_text(encoding="utf-8")
        assert "return 'universe'" in content
        assert "return 'world'" not in content

    def test_str_replace_not_found(self):
        """old_str이 파일에 없으면 에러를 반환한다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        target = self.workspace / "missing.py"
        target.write_text("hello world\n", encoding="utf-8")

        result = str_replace("missing.py", "nonexistent text", "replacement")
        assert "Error" in result
        assert "찾을 수 없" in result

    def test_str_replace_multiple_matches(self):
        """old_str이 2번 이상 존재하면 모호한 매치 에러를 반환한다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        target = self.workspace / "ambiguous.py"
        target.write_text("foo\nbar\nfoo\nbaz\n", encoding="utf-8")

        result = str_replace("ambiguous.py", "foo", "qux")
        assert "Error" in result
        assert "모호한 매치" in result
        assert "2번" in result

    def test_str_replace_workspace_outside_blocked(self):
        """workspace 밖 경로 접근이 차단된다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        with pytest.raises(ValueError, match="접근 거부"):
            str_replace("../../../etc/passwd", "root", "hacked")

    def test_str_replace_file_not_exists(self):
        """존재하지 않는 파일에 대해 에러를 반환한다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        result = str_replace("no_such_file.py", "old", "new")
        assert "Error" in result
        assert "존재하지 않습니다" in result

    def test_str_replace_multiline(self):
        """여러 줄에 걸친 문자열도 교체할 수 있다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        target = self.workspace / "multi.py"
        target.write_text("def foo():\n    x = 1\n    return x\n", encoding="utf-8")

        result = str_replace(
            "multi.py",
            "    x = 1\n    return x",
            "    x = 42\n    y = x * 2\n    return y",
        )
        assert "✓" in result or "OK" in result
        content = target.read_text(encoding="utf-8")
        assert "x = 42" in content
        assert "y = x * 2" in content

    def test_str_replace_preserves_rest(self):
        """교체 대상 외의 내용이 보존된다."""
        from youngs75_a2a.mcp_servers.code_tools.server import str_replace

        original = "aaa\nbbb\nccc\nddd\n"
        target = self.workspace / "preserve.py"
        target.write_text(original, encoding="utf-8")

        result = str_replace("preserve.py", "bbb", "BBB")
        assert "✓" in result or "OK" in result
        content = target.read_text(encoding="utf-8")
        assert content == "aaa\nBBB\nccc\nddd\n"
