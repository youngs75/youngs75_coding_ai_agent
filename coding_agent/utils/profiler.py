"""성능 프로파일링 유틸리티.

에이전트 실행 시간, 노드별 성능, 토큰 사용량을 추적하고
구조화된 리포트를 생성한다.

기존 AgentMetricsCollector(eval_pipeline/observability/callback_handler.py)와
호환되며, 독립적으로도 사용 가능하다.

사용 예시:
    from coding_agent.utils.profiler import Profiler, profile_async

    # 데코레이터
    @profile_async(name="parse_request")
    async def parse_request(state):
        ...

    # 컨텍스트 매니저
    profiler = Profiler()
    with profiler.measure("execute_code"):
        result = await execute(...)

    profiler.record_tokens("execute_code", input_tokens=500, output_tokens=200)
    print(profiler.report())
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

logger = logging.getLogger(__name__)


@dataclass
class NodeProfile:
    """단일 노드의 프로파일 데이터."""

    name: str
    call_count: int = 0
    total_duration_s: float = 0.0
    min_duration_s: float = float("inf")
    max_duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    @property
    def avg_duration_s(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_duration_s / self.call_count

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record_duration(self, duration_s: float) -> None:
        """실행 시간을 기록한다."""
        self.call_count += 1
        self.total_duration_s += duration_s
        self.min_duration_s = min(self.min_duration_s, duration_s)
        self.max_duration_s = max(self.max_duration_s, duration_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_duration_s": round(self.total_duration_s, 4),
            "avg_duration_s": round(self.avg_duration_s, 4),
            "min_duration_s": round(self.min_duration_s, 4)
            if self.call_count > 0
            else 0,
            "max_duration_s": round(self.max_duration_s, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
        }


class Profiler:
    """에이전트 실행 프로파일러.

    노드별 실행 시간과 토큰 사용량을 추적하고,
    구조화된 리포트를 생성한다.
    """

    def __init__(self, *, name: str = "agent") -> None:
        self.name = name
        self._nodes: dict[str, NodeProfile] = {}
        self._start_time: float = time.perf_counter()
        self._end_time: float | None = None
        # 진행 중인 측정 스택
        self._active: dict[str, float] = {}

    def _get_or_create_node(self, node_name: str) -> NodeProfile:
        if node_name not in self._nodes:
            self._nodes[node_name] = NodeProfile(name=node_name)
        return self._nodes[node_name]

    @contextmanager
    def measure(self, node_name: str) -> Generator[NodeProfile, None, None]:
        """노드 실행 시간을 측정하는 컨텍스트 매니저.

        사용 예시:
            with profiler.measure("parse_request") as node:
                result = await parse(state)
            # node.call_count, node.total_duration_s 등 업데이트됨
        """
        node = self._get_or_create_node(node_name)
        start = time.perf_counter()
        error_occurred = False
        try:
            yield node
        except Exception:
            error_occurred = True
            node.errors += 1
            raise
        finally:
            duration = time.perf_counter() - start
            node.record_duration(duration)
            level = logging.WARNING if error_occurred else logging.DEBUG
            logger.log(
                level,
                "[Profiler] %s: %.4fs%s",
                node_name,
                duration,
                " (ERROR)" if error_occurred else "",
            )

    def start_node(self, node_name: str) -> None:
        """노드 측정을 시작한다 (명시적 시작/종료 패턴)."""
        self._active[node_name] = time.perf_counter()

    def end_node(self, node_name: str, *, error: bool = False) -> float:
        """노드 측정을 종료하고 소요 시간을 반환한다.

        Returns:
            소요 시간 (초)

        Raises:
            KeyError: start_node()가 호출되지 않은 경우
        """
        start = self._active.pop(node_name, None)
        if start is None:
            raise KeyError(f"'{node_name}'에 대한 start_node()가 호출되지 않았습니다")

        duration = time.perf_counter() - start
        node = self._get_or_create_node(node_name)
        node.record_duration(duration)
        if error:
            node.errors += 1

        return duration

    def record_tokens(
        self,
        node_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """노드의 토큰 사용량을 기록한다."""
        node = self._get_or_create_node(node_name)
        node.input_tokens += input_tokens
        node.output_tokens += output_tokens

    def finalize(self) -> None:
        """프로파일링을 완료한다."""
        self._end_time = time.perf_counter()

    @property
    def total_duration_s(self) -> float:
        """전체 프로파일링 소요 시간 (초)."""
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @property
    def total_input_tokens(self) -> int:
        return sum(n.input_tokens for n in self._nodes.values())

    @property
    def total_output_tokens(self) -> int:
        return sum(n.output_tokens for n in self._nodes.values())

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def nodes(self) -> dict[str, NodeProfile]:
        """노드별 프로파일 데이터."""
        return dict(self._nodes)

    def report(self) -> dict[str, Any]:
        """구조화된 프로파일 리포트를 반환한다.

        AgentMetricsCollector.to_dict()와 호환되는 필드를 포함한다.
        """
        return {
            "profiler_name": self.name,
            "total_duration_s": round(self.total_duration_s, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "nodes": {name: node.to_dict() for name, node in self._nodes.items()},
        }

    def report_text(self) -> str:
        """사람이 읽기 쉬운 텍스트 리포트를 반환한다."""
        data = self.report()
        lines = [
            f"=== 프로파일 리포트: {data['profiler_name']} ===",
            f"전체 소요 시간: {data['total_duration_s']:.4f}s",
            f"전체 토큰: {data['total_tokens']:,} "
            f"(입력: {data['total_input_tokens']:,}, 출력: {data['total_output_tokens']:,})",
            "",
            "노드별 상세:",
        ]

        for name, node_data in data["nodes"].items():
            lines.append(
                f"  {name:20s}: "
                f"호출={node_data['call_count']}, "
                f"평균={node_data['avg_duration_s']:.4f}s, "
                f"합계={node_data['total_duration_s']:.4f}s, "
                f"토큰={node_data['total_tokens']:,}"
                + (f", 오류={node_data['errors']}" if node_data["errors"] else "")
            )

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_agent_metrics_dict(self) -> dict[str, Any]:
        """AgentMetricsCollector.to_dict() 호환 딕셔너리를 반환한다.

        eval_pipeline/observability/callback_handler.py의
        AgentMetricsCollector가 사용하는 필드명과 호환된다.
        """
        total_errors = sum(n.errors for n in self._nodes.values())
        total_calls = sum(n.call_count for n in self._nodes.values())
        error_rate = total_errors / total_calls if total_calls > 0 else 0.0

        return {
            "agent_name": self.name,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_input_tokens,
            "completion_tokens": self.total_output_tokens,
            "duration_ms": round(self.total_duration_s * 1000, 2),
            "error_count": total_errors,
            "error_rate": round(error_rate, 4),
            "nodes": {
                name: {
                    "call_count": n.call_count,
                    "avg_duration_ms": round(n.avg_duration_s * 1000, 2),
                    "error_count": n.errors,
                    "error_rate": round(n.errors / n.call_count, 4)
                    if n.call_count > 0
                    else 0.0,
                }
                for name, n in self._nodes.items()
            },
        }


# ── 데코레이터 ──


def profile_sync(
    name: str | None = None,
    *,
    profiler: Profiler | None = None,
) -> Any:
    """동기 함수의 실행 시간을 프로파일링하는 데코레이터.

    profiler가 제공되지 않으면 로그만 남긴다.

    Args:
        name: 프로파일 이름 (None이면 함수 이름 사용)
        profiler: Profiler 인스턴스
    """

    def decorator(func: Any) -> Any:
        node_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if profiler is not None:
                with profiler.measure(node_name):
                    return func(*args, **kwargs)
            else:
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    logger.debug("[profile] %s: %.4fs", node_name, duration)

        return wrapper

    return decorator


def profile_async(
    name: str | None = None,
    *,
    profiler: Profiler | None = None,
) -> Any:
    """비동기 함수의 실행 시간을 프로파일링하는 데코레이터.

    Args:
        name: 프로파일 이름 (None이면 함수 이름 사용)
        profiler: Profiler 인스턴스
    """

    def decorator(func: Any) -> Any:
        node_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if profiler is not None:
                # 비동기 함수에서는 컨텍스트 매니저를 직접 사용
                node = profiler._get_or_create_node(node_name)
                start = time.perf_counter()
                error_occurred = False
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception:
                    error_occurred = True
                    node.errors += 1
                    raise
                finally:
                    duration = time.perf_counter() - start
                    node.record_duration(duration)
                    level = logging.WARNING if error_occurred else logging.DEBUG
                    logger.log(level, "[profile] %s: %.4fs", node_name, duration)
            else:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    logger.debug("[profile] %s: %.4fs", node_name, duration)

        return wrapper

    return decorator
