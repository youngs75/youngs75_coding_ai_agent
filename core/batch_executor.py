"""배치/병렬 실행 최적화 모듈.

여러 독립적인 LLM 호출이나 비동기 작업을 병렬로 실행한다.
asyncio.gather + 세마포어 기반으로 동시 실행 수를 제한하고,
일부 실패 시에도 나머지 결과를 반환한다.

사용 예시:
    from youngs75_a2a.core.batch_executor import BatchExecutor

    executor = BatchExecutor(max_concurrency=5)

    # 여러 LLM 호출을 병렬 실행
    tasks = [
        lambda: llm.ainvoke(prompt_1),
        lambda: llm.ainvoke(prompt_2),
        lambda: llm.ainvoke(prompt_3),
    ]
    results = await executor.execute(tasks)

    # 결과 확인 (성공/실패 혼재 가능)
    for r in results:
        if r.success:
            print(r.value)
        else:
            print(f"실패: {r.error}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """개별 태스크 실행 결과.

    Attributes:
        index: 태스크 인덱스 (입력 순서)
        success: 성공 여부
        value: 성공 시 반환값 (실패 시 None)
        error: 실패 시 예외 (성공 시 None)
        duration_s: 실행 소요 시간 (초)
    """

    index: int
    success: bool
    value: Any = None
    error: Exception | None = None
    duration_s: float = 0.0


@dataclass
class BatchResult:
    """배치 실행 전체 결과.

    Attributes:
        results: 태스크별 결과 리스트 (입력 순서 유지)
        total_duration_s: 배치 전체 소요 시간 (wall-clock)
    """

    results: list[TaskResult] = field(default_factory=list)
    total_duration_s: float = 0.0

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self.results)

    @property
    def values(self) -> list[Any]:
        """성공한 태스크의 반환값만 순서대로 반환한다.

        실패한 태스크는 None으로 표시된다.
        """
        return [r.value for r in self.results]

    @property
    def successful_values(self) -> list[Any]:
        """성공한 태스크의 반환값만 반환한다 (실패 항목 제외)."""
        return [r.value for r in self.results if r.success]

    @property
    def errors(self) -> list[tuple[int, Exception]]:
        """실패한 태스크의 (인덱스, 예외) 리스트."""
        return [(r.index, r.error) for r in self.results if not r.success and r.error]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_duration_s": round(self.total_duration_s, 4),
            "all_succeeded": self.all_succeeded,
        }


class BatchExecutor:
    """비동기 태스크를 배치로 병렬 실행한다.

    Args:
        max_concurrency: 최대 동시 실행 수 (세마포어).
            0이면 제한 없음.
        timeout_s: 개별 태스크 타임아웃 (초). None이면 제한 없음.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 5,
        timeout_s: float | None = None,
    ) -> None:
        self._max_concurrency = max_concurrency
        self._timeout_s = timeout_s

    async def execute(
        self,
        tasks: list[Callable[[], Awaitable[Any]]],
    ) -> BatchResult:
        """태스크 리스트를 병렬 실행한다.

        각 태스크는 인자 없는 비동기 callable이어야 한다.
        예: [lambda: llm.ainvoke(msg), ...]

        일부 태스크가 실패해도 나머지는 정상 완료된다.

        Args:
            tasks: 비동기 callable 리스트

        Returns:
            BatchResult: 전체 배치 결과
        """
        if not tasks:
            return BatchResult()

        semaphore = (
            asyncio.Semaphore(self._max_concurrency)
            if self._max_concurrency > 0
            else None
        )

        start = time.perf_counter()

        async def _run_one(index: int, task: Callable[[], Awaitable[Any]]) -> TaskResult:
            task_start = time.perf_counter()
            try:
                if semaphore is not None:
                    async with semaphore:
                        result = await self._execute_with_timeout(task)
                else:
                    result = await self._execute_with_timeout(task)

                duration = time.perf_counter() - task_start
                return TaskResult(
                    index=index,
                    success=True,
                    value=result,
                    duration_s=duration,
                )
            except Exception as e:
                duration = time.perf_counter() - task_start
                logger.warning(
                    "[BatchExecutor] 태스크 %d 실패 (%.2fs): %s",
                    index,
                    duration,
                    str(e),
                )
                return TaskResult(
                    index=index,
                    success=False,
                    error=e,
                    duration_s=duration,
                )

        # 모든 태스크를 동시 실행
        coros = [_run_one(i, task) for i, task in enumerate(tasks)]
        task_results = await asyncio.gather(*coros)

        # 입력 순서대로 정렬
        sorted_results = sorted(task_results, key=lambda r: r.index)

        total_duration = time.perf_counter() - start
        batch_result = BatchResult(
            results=sorted_results,
            total_duration_s=total_duration,
        )

        logger.info(
            "[BatchExecutor] 완료: %d/%d 성공, %.2fs",
            batch_result.success_count,
            batch_result.total_count,
            total_duration,
        )

        return batch_result

    async def execute_map(
        self,
        func: Callable[[Any], Awaitable[Any]],
        items: list[Any],
    ) -> BatchResult:
        """동일한 함수를 여러 입력에 대해 병렬 실행한다.

        map-style API로, 하나의 함수를 여러 입력에 적용한다.

        Args:
            func: 각 항목에 적용할 비동기 함수
            items: 입력 항목 리스트

        Returns:
            BatchResult: 전체 배치 결과
        """
        tasks = [lambda item=item: func(item) for item in items]
        return await self.execute(tasks)

    async def _execute_with_timeout(
        self, task: Callable[[], Awaitable[Any]]
    ) -> Any:
        """타임아웃을 적용하여 태스크를 실행한다."""
        if self._timeout_s is not None:
            return await asyncio.wait_for(task(), timeout=self._timeout_s)
        return await task()
