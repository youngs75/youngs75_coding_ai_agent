"""병렬 도구 실행기 — Claude Code StreamingToolExecutor 패턴의 Python/asyncio 구현.

도구 호출을 동시성 안전성에 따라 분류하고, 안전한 도구는 병렬로,
안전하지 않은 도구는 순차적으로 실행한다.
결과는 항상 원래 요청 순서대로 반환된다.

사용 예시:
    from youngs75_a2a.core.parallel_tool_executor import ParallelToolExecutor

    executor = ParallelToolExecutor()
    results = await executor.execute_batch(tool_calls, tool_executor_fn)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import ToolMessage

from youngs75_a2a.core.hooks import HookContext, HookEvent, HookManager
from youngs75_a2a.core.tool_call_utils import tc_args, tc_id, tc_name

logger = logging.getLogger(__name__)


# 병렬 실행이 안전한 읽기 전용 도구 목록
CONCURRENCY_SAFE_TOOLS: set[str] = {
    "read_file",
    "search_code",
    "list_directory",
    "search_web",
    "search_papers",
    "search_recent_papers",
}


@dataclass
class ToolExecutionResult:
    """개별 도구 실행 결과.

    Attributes:
        index: 원래 tool_calls 리스트에서의 위치 (순서 복원용)
        tool_message: LangChain ToolMessage 객체
        duration_s: 실행 소요 시간 (초)
        success: 실행 성공 여부
        cancelled: 훅에 의해 실행이 취소되었는지 여부
    """

    index: int
    tool_message: ToolMessage
    duration_s: float = 0.0
    success: bool = True
    cancelled: bool = False


@dataclass
class BatchToolResult:
    """배치 도구 실행 전체 결과.

    Attributes:
        results: 도구별 실행 결과 (원래 순서대로 정렬)
        total_duration_s: 전체 배치 소요 시간
        parallel_count: 병렬로 실행된 도구 수
        sequential_count: 순차로 실행된 도구 수
    """

    results: list[ToolExecutionResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    parallel_count: int = 0
    sequential_count: int = 0

    @property
    def tool_messages(self) -> list[ToolMessage]:
        """원래 순서대로 정렬된 ToolMessage 리스트."""
        sorted_results = sorted(self.results, key=lambda r: r.index)
        return [r.tool_message for r in sorted_results]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)


class ParallelToolExecutor:
    """병렬 안전 도구 실행기.

    - concurrency_safe 도구끼리는 asyncio.gather로 병렬 실행
    - non-concurrent 도구는 독점 실행 (다른 도구 완료 후)
    - 결과는 요청 순서대로 반환

    Args:
        concurrency_safe_tools: 병렬 실행 가능한 도구 이름 집합.
            기본값은 모듈 수준 CONCURRENCY_SAFE_TOOLS.
        max_parallel: 동시 실행 최대 수 (세마포어). 0이면 제한 없음.
        timeout_s: 개별 도구 실행 타임아웃 (초). None이면 제한 없음.
    """

    def __init__(
        self,
        *,
        concurrency_safe_tools: set[str] | None = None,
        max_parallel: int = 10,
        timeout_s: float | None = 60.0,
        hook_manager: HookManager | None = None,
    ) -> None:
        self._safe_tools = (
            concurrency_safe_tools
            if concurrency_safe_tools is not None
            else CONCURRENCY_SAFE_TOOLS
        )
        self._max_parallel = max_parallel
        self._timeout_s = timeout_s
        self.hook_manager = hook_manager or HookManager()

    def is_concurrency_safe(self, tool_name: str | None) -> bool:
        """도구가 병렬 실행 안전한지 판단한다."""
        if tool_name is None:
            return False
        return tool_name in self._safe_tools

    def classify_tool_calls(
        self, tool_calls: list[Any]
    ) -> tuple[list[tuple[int, Any]], list[tuple[int, Any]]]:
        """도구 호출을 safe(병렬)와 non-concurrent(순차)로 분류한다.

        Returns:
            (safe_calls, non_concurrent_calls): 각각 (원래 인덱스, tool_call) 튜플 리스트
        """
        safe: list[tuple[int, Any]] = []
        non_concurrent: list[tuple[int, Any]] = []

        for i, call in enumerate(tool_calls):
            name = tc_name(call)
            if self.is_concurrency_safe(name):
                safe.append((i, call))
            else:
                non_concurrent.append((i, call))

        return safe, non_concurrent

    async def _execute_single_tool(
        self,
        index: int,
        tool_call: Any,
        tool_executor: Callable[..., Any],
        semaphore: asyncio.Semaphore | None = None,
    ) -> ToolExecutionResult:
        """단일 도구를 실행하고 ToolExecutionResult를 반환한다.

        PRE_TOOL_CALL 훅에서 context를 수정하면 변경된 인자로 실행된다.
        PRE_TOOL_CALL 훅에서 cancel=True 설정 시 실행을 스킵한다.

        Args:
            index: 원래 tool_calls 리스트에서의 인덱스
            tool_call: 도구 호출 객체
            tool_executor: 도구를 실행할 callable (name, args) -> str
            semaphore: 동시성 제한 세마포어 (선택)
        """
        name = tc_name(tool_call) or "unknown"
        args = tc_args(tool_call)
        call_id = tc_id(tool_call) or f"call_{index}"

        # PRE_TOOL_CALL 훅 발행
        pre_ctx = HookContext(
            event=HookEvent.PRE_TOOL_CALL,
            tool_name=name,
            tool_args=dict(args) if args else {},
        )
        pre_ctx = await self.hook_manager.emit(pre_ctx)

        # 훅에 의한 취소 처리
        if pre_ctx.metadata.get("cancel"):
            logger.info("[ParallelToolExecutor] %s 훅에 의해 취소됨", name)
            return ToolExecutionResult(
                index=index,
                tool_message=ToolMessage(
                    content=f"도구 실행 취소됨 (훅): {name}",
                    tool_call_id=call_id,
                    name=name,
                ),
                duration_s=0.0,
                success=True,
                cancelled=True,
            )

        # 훅이 수정했을 수 있는 인자/이름 사용
        effective_name = pre_ctx.tool_name or name
        effective_args = pre_ctx.tool_args if pre_ctx.tool_args is not None else args

        start = time.perf_counter()
        try:
            if semaphore is not None:
                async with semaphore:
                    result = await self._invoke_with_timeout(
                        tool_executor, effective_name, effective_args
                    )
            else:
                result = await self._invoke_with_timeout(
                    tool_executor, effective_name, effective_args
                )

            duration = time.perf_counter() - start
            logger.debug(
                "[ParallelToolExecutor] %s 완료 (%.2fs)", effective_name, duration
            )

            # POST_TOOL_CALL 훅 발행
            post_ctx = HookContext(
                event=HookEvent.POST_TOOL_CALL,
                tool_name=effective_name,
                tool_args=effective_args,
                tool_result=result,
                metadata=dict(pre_ctx.metadata),
            )
            post_ctx = await self.hook_manager.emit(post_ctx)

            return ToolExecutionResult(
                index=index,
                tool_message=ToolMessage(
                    content=str(post_ctx.tool_result)
                    if post_ctx.tool_result
                    else "실행 완료 (출력 없음)",
                    tool_call_id=call_id,
                    name=effective_name,
                ),
                duration_s=duration,
                success=True,
            )
        except Exception as e:
            duration = time.perf_counter() - start
            logger.warning(
                "[ParallelToolExecutor] %s 실패 (%.2fs): %s",
                effective_name,
                duration,
                e,
            )
            return ToolExecutionResult(
                index=index,
                tool_message=ToolMessage(
                    content=f"도구 실행 오류: {e}",
                    tool_call_id=call_id,
                    name=effective_name,
                ),
                duration_s=duration,
                success=False,
            )

    async def _invoke_with_timeout(
        self, tool_executor: Callable[..., Any], name: str, args: dict
    ) -> Any:
        """타임아웃을 적용하여 도구를 실행한다."""
        coro = tool_executor(name, args)
        if self._timeout_s is not None:
            return await asyncio.wait_for(coro, timeout=self._timeout_s)
        return await coro

    async def execute_batch(
        self,
        tool_calls: list[Any],
        tool_executor: Callable[..., Any],
    ) -> list[ToolMessage]:
        """여러 도구 호출을 최적으로 실행한다.

        1. tool_calls를 concurrency_safe 그룹과 non-concurrent로 분류
        2. safe 그룹은 asyncio.gather로 병렬 실행
        3. non-concurrent는 순차 실행 (앞의 safe 그룹 완료 후)
        4. 결과를 원래 순서대로 정렬하여 반환

        Args:
            tool_calls: 도구 호출 객체 리스트
            tool_executor: async callable (name: str, args: dict) -> str

        Returns:
            원래 순서대로 정렬된 ToolMessage 리스트
        """
        if not tool_calls:
            return []

        batch_start = time.perf_counter()
        safe_calls, non_concurrent_calls = self.classify_tool_calls(tool_calls)

        all_results: list[ToolExecutionResult] = []

        # 세마포어 생성 (병렬 실행 제한)
        semaphore = (
            asyncio.Semaphore(self._max_parallel) if self._max_parallel > 0 else None
        )

        # 1단계: safe 도구들을 병렬 실행
        if safe_calls:
            safe_coros = [
                self._execute_single_tool(idx, call, tool_executor, semaphore)
                for idx, call in safe_calls
            ]
            safe_results = await asyncio.gather(*safe_coros)
            all_results.extend(safe_results)

            logger.info(
                "[ParallelToolExecutor] 병렬 실행 완료: %d개 도구", len(safe_calls)
            )

        # 2단계: non-concurrent 도구들을 순차 실행
        for idx, call in non_concurrent_calls:
            result = await self._execute_single_tool(
                idx, call, tool_executor, semaphore=None
            )
            all_results.append(result)

        if non_concurrent_calls:
            logger.info(
                "[ParallelToolExecutor] 순차 실행 완료: %d개 도구",
                len(non_concurrent_calls),
            )

        batch_duration = time.perf_counter() - batch_start

        # 결과를 원래 순서대로 정렬
        all_results.sort(key=lambda r: r.index)

        batch_result = BatchToolResult(
            results=all_results,
            total_duration_s=batch_duration,
            parallel_count=len(safe_calls),
            sequential_count=len(non_concurrent_calls),
        )

        logger.info(
            "[ParallelToolExecutor] 배치 완료: 병렬=%d, 순차=%d, 성공=%d/%d, %.2fs",
            batch_result.parallel_count,
            batch_result.sequential_count,
            batch_result.success_count,
            len(all_results),
            batch_duration,
        )

        return batch_result.tool_messages

    async def execute_batch_detailed(
        self,
        tool_calls: list[Any],
        tool_executor: Callable[..., Any],
    ) -> BatchToolResult:
        """execute_batch와 동일하지만 상세 결과(BatchToolResult)를 반환한다.

        디버깅이나 관측성이 필요할 때 사용한다.
        """
        if not tool_calls:
            return BatchToolResult()

        batch_start = time.perf_counter()
        safe_calls, non_concurrent_calls = self.classify_tool_calls(tool_calls)

        all_results: list[ToolExecutionResult] = []

        semaphore = (
            asyncio.Semaphore(self._max_parallel) if self._max_parallel > 0 else None
        )

        # 병렬 실행
        if safe_calls:
            safe_coros = [
                self._execute_single_tool(idx, call, tool_executor, semaphore)
                for idx, call in safe_calls
            ]
            safe_results = await asyncio.gather(*safe_coros)
            all_results.extend(safe_results)

        # 순차 실행
        for idx, call in non_concurrent_calls:
            result = await self._execute_single_tool(
                idx, call, tool_executor, semaphore=None
            )
            all_results.append(result)

        batch_duration = time.perf_counter() - batch_start
        all_results.sort(key=lambda r: r.index)

        return BatchToolResult(
            results=all_results,
            total_duration_s=batch_duration,
            parallel_count=len(safe_calls),
            sequential_count=len(non_concurrent_calls),
        )
