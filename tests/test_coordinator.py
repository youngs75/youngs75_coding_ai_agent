"""Coordinator Mode 유닛 테스트.

TaskGraph: 의존성 해석, 순환 감지, 실행 웨이브 계산
CoordinatorMode: decompose → parallel execute → synthesize 파이프라인
타임아웃 처리, 부분 실패 핸들링, 병렬 효율성 메트릭
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from coding_agent.agents.orchestrator.coordinator import CoordinatorMode
from coding_agent.agents.orchestrator.schemas import (
    SubTask,
    WorkerResult,
)
from coding_agent.agents.orchestrator.task_graph import TaskGraph
from coding_agent.core.context_manager import ContextManager
from coding_agent.core.subagents.registry import SubAgentRegistry
from coding_agent.core.subagents.schemas import SubAgentSpec

# ─────────────────────────────────────────────────────────────
# 헬퍼: 테스트용 SubTask 생성
# ─────────────────────────────────────────────────────────────


def _make_subtask(
    task_id: str,
    *,
    deps: list[str] | None = None,
    priority: int = 1,
    timeout_s: float = 60.0,
    agent_type: str = "coder",
) -> SubTask:
    """테스트용 서브태스크를 간편하게 생성한다."""
    return SubTask(
        id=task_id,
        description=f"{task_id}에 대한 작업",
        agent_type=agent_type,
        dependencies=deps or [],
        priority=priority,
        timeout_s=timeout_s,
    )


def _make_registry() -> SubAgentRegistry:
    """테스트용 레지스트리를 생성한다."""
    reg = SubAgentRegistry(cost_sensitivity=0.3)
    reg.register(
        SubAgentSpec(
            name="coder",
            description="코드 생성 에이전트",
            capabilities=["code_generation"],
            cost_weight=1.0,
        )
    )
    reg.register(
        SubAgentSpec(
            name="reviewer",
            description="코드 리뷰 에이전트",
            capabilities=["code_review"],
            cost_weight=0.5,
        )
    )
    reg.register(
        SubAgentSpec(
            name="researcher",
            description="리서치 에이전트",
            capabilities=["research"],
            cost_weight=0.8,
        )
    )
    return reg


def _make_mock_llm(response_text: str = "mock response") -> AsyncMock:
    """테스트용 Mock LLM을 생성한다."""
    mock_llm = AsyncMock()
    mock_response = AsyncMock()
    mock_response.content = response_text
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


# ─────────────────────────────────────────────────────────────
# 1. TaskGraph 테스트
# ─────────────────────────────────────────────────────────────


class TestTaskGraphBasic:
    """TaskGraph 기본 기능 테스트."""

    def test_empty_graph(self):
        """빈 서브태스크 리스트로 TaskGraph를 생성할 수 있다."""
        graph = TaskGraph([])
        assert graph.subtask_count == 0
        assert graph.validate()
        assert graph.execution_waves == []

    def test_single_task(self):
        """단일 태스크 그래프가 올바르게 생성된다."""
        tasks = [_make_subtask("t1")]
        graph = TaskGraph(tasks)
        assert graph.subtask_count == 1
        assert graph.validate()
        waves = graph.execution_waves
        assert len(waves) == 1
        assert waves[0][0]["id"] == "t1"

    def test_independent_tasks(self):
        """의존성 없는 태스크들이 단일 웨이브로 묶인다."""
        tasks = [
            _make_subtask("t1"),
            _make_subtask("t2"),
            _make_subtask("t3"),
        ]
        graph = TaskGraph(tasks)
        assert graph.validate()
        waves = graph.execution_waves
        assert len(waves) == 1
        assert len(waves[0]) == 3

    def test_get_subtask(self):
        """ID로 서브태스크를 조회할 수 있다."""
        tasks = [_make_subtask("t1"), _make_subtask("t2")]
        graph = TaskGraph(tasks)
        assert graph.get_subtask("t1") is not None
        assert graph.get_subtask("t1")["id"] == "t1"
        assert graph.get_subtask("nonexistent") is None


class TestTaskGraphDependencies:
    """TaskGraph 의존성 관리 테스트."""

    def test_linear_dependency(self):
        """순차 의존성 (t1 → t2 → t3)이 올바르게 해석된다."""
        tasks = [
            _make_subtask("t1"),
            _make_subtask("t2", deps=["t1"]),
            _make_subtask("t3", deps=["t2"]),
        ]
        graph = TaskGraph(tasks)
        assert graph.validate()
        waves = graph.execution_waves
        assert len(waves) == 3
        assert waves[0][0]["id"] == "t1"
        assert waves[1][0]["id"] == "t2"
        assert waves[2][0]["id"] == "t3"

    def test_diamond_dependency(self):
        """다이아몬드 의존성 (t1 → t2, t3 → t4)이 올바르게 해석된다."""
        tasks = [
            _make_subtask("t1"),
            _make_subtask("t2", deps=["t1"]),
            _make_subtask("t3", deps=["t1"]),
            _make_subtask("t4", deps=["t2", "t3"]),
        ]
        graph = TaskGraph(tasks)
        assert graph.validate()
        waves = graph.execution_waves
        assert len(waves) == 3  # [t1], [t2, t3], [t4]
        assert waves[0][0]["id"] == "t1"
        assert len(waves[1]) == 2  # t2, t3 병렬
        wave1_ids = {w["id"] for w in waves[1]}
        assert wave1_ids == {"t2", "t3"}
        assert waves[2][0]["id"] == "t4"

    def test_mixed_dependency(self):
        """혼합 의존성 — 일부 독립, 일부 의존."""
        tasks = [
            _make_subtask("t1"),
            _make_subtask("t2"),
            _make_subtask("t3", deps=["t1"]),
            _make_subtask("t4", deps=["t1", "t2"]),
        ]
        graph = TaskGraph(tasks)
        assert graph.validate()
        waves = graph.execution_waves
        assert len(waves) == 2  # [t1, t2], [t3, t4]
        wave0_ids = {w["id"] for w in waves[0]}
        assert wave0_ids == {"t1", "t2"}
        wave1_ids = {w["id"] for w in waves[1]}
        assert wave1_ids == {"t3", "t4"}

    def test_nonexistent_dependency_ignored(self):
        """존재하지 않는 의존성 ID는 무시된다."""
        tasks = [
            _make_subtask("t1", deps=["nonexistent"]),
            _make_subtask("t2"),
        ]
        graph = TaskGraph(tasks)
        assert graph.validate()
        waves = graph.execution_waves
        # 존재하지 않는 의존성은 무시되므로 t1, t2 모두 첫 웨이브
        assert len(waves) == 1
        assert len(waves[0]) == 2


class TestTaskGraphCycleDetection:
    """TaskGraph 순환 의존성 감지 테스트."""

    def test_simple_cycle(self):
        """단순 순환 (t1 → t2 → t1)을 감지한다."""
        tasks = [
            _make_subtask("t1", deps=["t2"]),
            _make_subtask("t2", deps=["t1"]),
        ]
        graph = TaskGraph(tasks)
        assert not graph.validate()

    def test_self_cycle(self):
        """자기 참조 순환 (t1 → t1)을 감지한다."""
        tasks = [_make_subtask("t1", deps=["t1"])]
        graph = TaskGraph(tasks)
        assert not graph.validate()

    def test_three_node_cycle(self):
        """3노드 순환 (t1 → t2 → t3 → t1)을 감지한다."""
        tasks = [
            _make_subtask("t1", deps=["t3"]),
            _make_subtask("t2", deps=["t1"]),
            _make_subtask("t3", deps=["t2"]),
        ]
        graph = TaskGraph(tasks)
        assert not graph.validate()

    def test_cycle_raises_on_execution_waves(self):
        """순환 그래프에서 execution_waves 접근 시 ValueError."""
        tasks = [
            _make_subtask("t1", deps=["t2"]),
            _make_subtask("t2", deps=["t1"]),
        ]
        graph = TaskGraph(tasks)
        with pytest.raises(ValueError, match="순환 의존성"):
            _ = graph.execution_waves


class TestTaskGraphReadyTasks:
    """TaskGraph get_ready_tasks 테스트."""

    def test_all_ready_when_no_deps(self):
        """의존성이 없으면 모든 태스크가 준비 상태."""
        tasks = [_make_subtask("t1"), _make_subtask("t2")]
        graph = TaskGraph(tasks)
        ready = graph.get_ready_tasks(completed=set())
        assert len(ready) == 2

    def test_ready_after_deps_completed(self):
        """의존성 완료 후 태스크가 준비 상태가 된다."""
        tasks = [
            _make_subtask("t1"),
            _make_subtask("t2", deps=["t1"]),
        ]
        graph = TaskGraph(tasks)
        # t1 미완료 시
        ready = graph.get_ready_tasks(completed=set())
        assert len(ready) == 1
        assert ready[0]["id"] == "t1"
        # t1 완료 후
        ready = graph.get_ready_tasks(completed={"t1"})
        assert len(ready) == 1
        assert ready[0]["id"] == "t2"

    def test_completed_tasks_excluded(self):
        """이미 완료된 태스크는 제외된다."""
        tasks = [_make_subtask("t1"), _make_subtask("t2")]
        graph = TaskGraph(tasks)
        ready = graph.get_ready_tasks(completed={"t1"})
        assert len(ready) == 1
        assert ready[0]["id"] == "t2"

    def test_priority_ordering(self):
        """준비된 태스크가 우선순위 높은 순으로 정렬된다."""
        tasks = [
            _make_subtask("t1", priority=1),
            _make_subtask("t2", priority=5),
            _make_subtask("t3", priority=3),
        ]
        graph = TaskGraph(tasks)
        ready = graph.get_ready_tasks(completed=set())
        assert [r["id"] for r in ready] == ["t2", "t3", "t1"]


class TestTaskGraphExecutionWaves:
    """TaskGraph execution_waves 속성 테스트."""

    def test_priority_within_wave(self):
        """웨이브 내에서 우선순위 순으로 정렬된다."""
        tasks = [
            _make_subtask("t1", priority=1),
            _make_subtask("t2", priority=10),
            _make_subtask("t3", priority=5),
        ]
        graph = TaskGraph(tasks)
        waves = graph.execution_waves
        assert len(waves) == 1
        ids = [w["id"] for w in waves[0]]
        assert ids == ["t2", "t3", "t1"]

    def test_complex_dag(self):
        """복잡한 DAG의 웨이브가 올바르게 계산된다.

        구조:
            t1, t2 (독립)
            t3 (← t1)
            t4 (← t2)
            t5 (← t3, t4)
        """
        tasks = [
            _make_subtask("t1"),
            _make_subtask("t2"),
            _make_subtask("t3", deps=["t1"]),
            _make_subtask("t4", deps=["t2"]),
            _make_subtask("t5", deps=["t3", "t4"]),
        ]
        graph = TaskGraph(tasks)
        waves = graph.execution_waves
        assert len(waves) == 3
        # 웨이브 0: t1, t2
        assert {w["id"] for w in waves[0]} == {"t1", "t2"}
        # 웨이브 1: t3, t4
        assert {w["id"] for w in waves[1]} == {"t3", "t4"}
        # 웨이브 2: t5
        assert {w["id"] for w in waves[2]} == {"t5"}


# ─────────────────────────────────────────────────────────────
# 2. CoordinatorMode 테스트
# ─────────────────────────────────────────────────────────────


class TestCoordinatorDecompose:
    """CoordinatorMode.decompose_task 테스트."""

    async def test_decompose_parses_json(self):
        """LLM의 JSON 응답을 서브태스크로 파싱한다."""
        llm_response = """```json
[
  {"id": "t1", "description": "코드 작성", "agent_type": "coder", "dependencies": [], "priority": 2, "timeout_s": 60.0},
  {"id": "t2", "description": "코드 리뷰", "agent_type": "reviewer", "dependencies": ["t1"], "priority": 1, "timeout_s": 30.0}
]
```"""
        mock_llm = _make_mock_llm(llm_response)

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
        )
        subtasks = await coordinator.decompose_task(
            task="코드를 작성하고 리뷰해줘",
            llm=mock_llm,
            available_agents=_make_registry().list_available(),
        )

        assert len(subtasks) == 2
        assert subtasks[0]["id"] == "t1"
        assert subtasks[0]["agent_type"] == "coder"
        assert subtasks[1]["dependencies"] == ["t1"]

    async def test_decompose_fallback_on_invalid_json(self):
        """잘못된 JSON 응답 시 폴백 태스크를 생성한다."""
        mock_llm = _make_mock_llm("이건 유효한 JSON이 아닙니다")

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
        )
        subtasks = await coordinator.decompose_task(
            task="작업 설명",
            llm=mock_llm,
            available_agents=[],
        )

        assert len(subtasks) == 1
        assert subtasks[0]["id"] == "task_fallback"

    async def test_decompose_plain_json(self):
        """```json 블록 없이 순수 JSON도 파싱한다."""
        llm_response = '[{"id": "t1", "description": "테스트", "agent_type": "coder", "dependencies": [], "priority": 1, "timeout_s": 60.0}]'
        mock_llm = _make_mock_llm(llm_response)

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
        )
        subtasks = await coordinator.decompose_task(
            task="테스트",
            llm=mock_llm,
            available_agents=[],
        )
        assert len(subtasks) == 1
        assert subtasks[0]["id"] == "t1"


class TestCoordinatorExecuteParallel:
    """CoordinatorMode.execute_parallel 테스트."""

    async def test_parallel_execution_basic(self):
        """독립 태스크가 병렬로 실행된다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            await asyncio.sleep(0.05)
            return f"결과: {subtask['id']}"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask("t1"), _make_subtask("t2"), _make_subtask("t3")]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        assert len(result["worker_results"]) == 3
        for wr in result["worker_results"]:
            assert wr["status"] == "success"
            assert "결과:" in wr["output"]

    async def test_parallel_faster_than_sequential(self):
        """병렬 실행이 순차 실행보다 빠르다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            await asyncio.sleep(0.1)
            return "done"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask("t1"), _make_subtask("t2"), _make_subtask("t3")]
        start = time.perf_counter()
        result = await coordinator.execute_parallel(subtasks, parent_context=[])
        elapsed = time.perf_counter() - start

        # 3개 × 0.1s = 0.3s 순차, 병렬이면 ~0.1s
        assert elapsed < 0.25, f"병렬 실행이 예상보다 느림: {elapsed:.2f}s"
        assert result["parallel_efficiency"] < 0.8  # 효율적

    async def test_dag_execution_order(self):
        """DAG 의존성에 따라 웨이브 순서대로 실행된다."""
        execution_order: list[str] = []

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            execution_order.append(subtask["id"])
            await asyncio.sleep(0.01)
            return f"done: {subtask['id']}"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        # t1 → t2, t3 (t2, t3은 t1에 의존)
        subtasks = [
            _make_subtask("t1"),
            _make_subtask("t2", deps=["t1"]),
            _make_subtask("t3", deps=["t1"]),
        ]
        await coordinator.execute_parallel(subtasks, parent_context=[])

        # t1이 t2, t3보다 먼저 실행되어야 함
        t1_idx = execution_order.index("t1")
        t2_idx = execution_order.index("t2")
        t3_idx = execution_order.index("t3")
        assert t1_idx < t2_idx
        assert t1_idx < t3_idx

    async def test_partial_failure_handling(self):
        """일부 워커 실패 시에도 나머지 결과가 반환된다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            if subtask["id"] == "t2":
                raise RuntimeError("t2 실행 실패")
            await asyncio.sleep(0.01)
            return f"결과: {subtask['id']}"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask("t1"), _make_subtask("t2"), _make_subtask("t3")]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        assert len(result["worker_results"]) == 3
        statuses = {wr["subtask_id"]: wr["status"] for wr in result["worker_results"]}
        assert statuses["t1"] == "success"
        assert statuses["t2"] == "failed"
        assert statuses["t3"] == "success"

        # 실패한 워커의 에러 메시지 확인
        t2_result = next(
            wr for wr in result["worker_results"] if wr["subtask_id"] == "t2"
        )
        assert t2_result["error"] is not None
        assert "t2 실행 실패" in t2_result["error"]

    async def test_timeout_handling(self):
        """타임아웃 초과 시 적절한 상태가 반환된다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            if subtask["id"] == "t_slow":
                await asyncio.sleep(10)
            return "done"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            timeout_s=0.1,  # 매우 짧은 타임아웃
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask("t_slow")]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        assert len(result["worker_results"]) == 1
        wr = result["worker_results"][0]
        assert wr["status"] == "timeout"
        assert wr["error"] is not None

    async def test_cyclic_dependency_fallback(self):
        """순환 의존성이 있으면 의존성 무시 후 병렬 실행한다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            return f"결과: {subtask['id']}"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        # 순환 의존성
        subtasks = [
            _make_subtask("t1", deps=["t2"]),
            _make_subtask("t2", deps=["t1"]),
        ]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        # 의존성 무시하고 모두 실행됨
        assert len(result["worker_results"]) == 2
        for wr in result["worker_results"]:
            assert wr["status"] == "success"


class TestCoordinatorSynthesize:
    """CoordinatorMode.synthesize 테스트."""

    async def test_synthesize_calls_llm(self):
        """synthesize가 LLM을 호출하여 결과를 통합한다."""
        mock_llm = _make_mock_llm("통합된 최종 응답입니다.")

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
        )

        results = [
            WorkerResult(
                subtask_id="t1",
                agent_name="coder",
                status="success",
                output="코드 작성 완료",
                duration_s=1.0,
                error=None,
            ),
            WorkerResult(
                subtask_id="t2",
                agent_name="reviewer",
                status="success",
                output="리뷰 완료",
                duration_s=0.5,
                error=None,
            ),
        ]

        synthesized = await coordinator.synthesize(
            original_task="코드를 작성하고 리뷰해줘",
            results=results,
            llm=mock_llm,
        )

        assert synthesized == "통합된 최종 응답입니다."
        mock_llm.ainvoke.assert_called_once()

    async def test_synthesize_includes_failed_results(self):
        """synthesize가 실패한 워커 결과도 포함하여 LLM에 전달한다."""
        mock_llm = _make_mock_llm("부분 결과를 포함한 응답")

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
        )

        results = [
            WorkerResult(
                subtask_id="t1",
                agent_name="coder",
                status="success",
                output="성공",
                duration_s=1.0,
                error=None,
            ),
            WorkerResult(
                subtask_id="t2",
                agent_name="reviewer",
                status="failed",
                output="",
                duration_s=0.5,
                error="에러 발생",
            ),
        ]

        await coordinator.synthesize(
            original_task="작업",
            results=results,
            llm=mock_llm,
        )

        # LLM에 전달된 메시지에 실패 정보가 포함되었는지 확인
        call_args = mock_llm.ainvoke.call_args[0][0]
        messages_text = str(call_args)
        assert "실패" in messages_text or "failed" in messages_text.lower()


class TestCoordinatorEfficiency:
    """병렬 효율성 메트릭 테스트."""

    async def test_efficiency_ratio_calculation(self):
        """병렬 효율성 비율이 올바르게 계산된다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            await asyncio.sleep(0.05)
            return "done"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask("t1"), _make_subtask("t2")]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        # 2개 태스크 × 0.05s = 0.1s 순차, 병렬 ~0.05s
        # 효율성 비율은 1.0 미만이어야 함 (병렬이 더 빠르므로)
        assert result["parallel_efficiency"] < 1.0
        assert result["total_duration_s"] > 0

    async def test_sequential_dep_efficiency(self):
        """순차 의존성이 있으면 효율성이 1.0에 가깝다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            await asyncio.sleep(0.05)
            return "done"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        # 완전히 순차적인 의존성
        subtasks = [
            _make_subtask("t1"),
            _make_subtask("t2", deps=["t1"]),
        ]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        # 순차 실행이므로 효율성은 ~1.0 (또는 오버헤드로 1.0보다 약간 클 수 있음)
        assert result["parallel_efficiency"] > 0.5


class TestCoordinatorFullPipeline:
    """CoordinatorMode 전체 파이프라인 (run) 테스트."""

    async def test_full_pipeline(self):
        """decompose → execute_parallel → synthesize 전체 파이프라인이 동작한다."""
        decompose_response = """```json
[
  {"id": "t1", "description": "코드 작성", "agent_type": "coder", "dependencies": [], "priority": 2, "timeout_s": 60.0},
  {"id": "t2", "description": "리서치", "agent_type": "researcher", "dependencies": [], "priority": 1, "timeout_s": 60.0},
  {"id": "t3", "description": "코드 리뷰", "agent_type": "reviewer", "dependencies": ["t1"], "priority": 1, "timeout_s": 60.0}
]
```"""
        synthesize_response = (
            "코드를 작성하고, 관련 연구를 조사하고, 리뷰를 완료했습니다."
        )

        # LLM이 decompose와 synthesize에서 각각 다른 응답을 반환하도록 설정
        call_count = 0

        async def mock_ainvoke(messages: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            if call_count == 1:
                resp.content = decompose_response
            else:
                resp.content = synthesize_response
            return resp

        mock_llm = AsyncMock()
        mock_llm.ainvoke = mock_ainvoke

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            await asyncio.sleep(0.01)
            return f"{subtask['id']} 완료"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        result = await coordinator.run(
            task="코드를 작성하고 리서치해서 리뷰까지 해줘",
            context=[],
            llm=mock_llm,
        )

        assert result["synthesized_response"] == synthesize_response
        assert len(result["worker_results"]) == 3
        assert result["total_duration_s"] > 0

        # t1, t2는 병렬 (웨이브 1), t3는 t1 이후 (웨이브 2)
        success_count = sum(
            1 for wr in result["worker_results"] if wr["status"] == "success"
        )
        assert success_count == 3

    async def test_pipeline_with_all_failures(self):
        """모든 워커가 실패해도 파이프라인이 정상 종료된다."""
        decompose_response = '[{"id": "t1", "description": "실패할 작업", "agent_type": "coder", "dependencies": [], "priority": 1, "timeout_s": 60.0}]'
        synthesize_response = "모든 서브태스크가 실패했습니다."

        call_count = 0

        async def mock_ainvoke(messages: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            if call_count == 1:
                resp.content = decompose_response
            else:
                resp.content = synthesize_response
            return resp

        mock_llm = AsyncMock()
        mock_llm.ainvoke = mock_ainvoke

        async def failing_worker(subtask: SubTask, context: Any) -> str:
            raise RuntimeError("전체 실패")

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=failing_worker,
        )

        result = await coordinator.run(
            task="실패할 작업",
            context=[],
            llm=mock_llm,
        )

        assert result["synthesized_response"] == synthesize_response
        assert all(wr["status"] == "failed" for wr in result["worker_results"])

    async def test_pipeline_empty_agents(self):
        """에이전트가 없어도 파이프라인이 정상 동작한다."""
        decompose_response = '[{"id": "t1", "description": "작업", "agent_type": "default", "dependencies": [], "priority": 1, "timeout_s": 60.0}]'
        synthesize_response = "완료"

        call_count = 0

        async def mock_ainvoke(messages: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            if call_count == 1:
                resp.content = decompose_response
            else:
                resp.content = synthesize_response
            return resp

        mock_llm = AsyncMock()
        mock_llm.ainvoke = mock_ainvoke

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            return "완료"

        # 빈 레지스트리
        empty_registry = SubAgentRegistry()

        coordinator = CoordinatorMode(
            registry=empty_registry,
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        result = await coordinator.run(
            task="작업",
            context=[],
            llm=mock_llm,
        )

        assert result["synthesized_response"] == "완료"


class TestCoordinatorMiscellaneous:
    """CoordinatorMode 기타 테스트."""

    async def test_max_workers_respected(self):
        """max_workers 설정이 동시 실행 수를 제한한다."""
        concurrent_count = 0
        max_concurrent = 0

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return "done"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            max_workers=2,
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask(f"t{i}") for i in range(5)]
        await coordinator.execute_parallel(subtasks, parent_context=[])

        assert max_concurrent <= 2

    async def test_worker_result_duration_tracked(self):
        """워커 결과에 실행 시간이 기록된다."""

        async def worker_fn(subtask: SubTask, context: Any) -> str:
            await asyncio.sleep(0.05)
            return "done"

        coordinator = CoordinatorMode(
            registry=_make_registry(),
            context_manager=ContextManager(),
            worker_fn=worker_fn,
        )

        subtasks = [_make_subtask("t1")]
        result = await coordinator.execute_parallel(subtasks, parent_context=[])

        wr = result["worker_results"][0]
        assert wr["duration_s"] >= 0.04  # 약간의 마진


# ─────────────────────────────────────────────────────────────
# 3. OrchestratorAgent 통합 테스트 — _route_after_classify
# ─────────────────────────────────────────────────────────────


class TestRouteAfterClassify:
    """classify 이후 라우팅 결정 테스트."""

    def test_route_to_coordinator(self):
        """selected_agent가 __coordinator__이면 coordinate로 라우팅."""
        from coding_agent.agents.orchestrator.agent import _route_after_classify

        state: dict[str, Any] = {"selected_agent": "__coordinator__"}
        assert _route_after_classify(state) == "coordinate"  # type: ignore[arg-type]

    def test_route_to_plan_for_coding(self):
        """selected_agent가 코딩 에이전트 + is_complex이면 plan으로 라우팅."""
        from coding_agent.agents.orchestrator.agent import _route_after_classify

        state: dict[str, Any] = {"selected_agent": "coder", "is_complex": True}
        assert _route_after_classify(state) == "plan"  # type: ignore[arg-type]

        state2: dict[str, Any] = {"selected_agent": "coding_assistant", "is_complex": True}
        assert _route_after_classify(state2) == "plan"  # type: ignore[arg-type]

    def test_route_to_delegate_for_simple_coding(self):
        """selected_agent가 코딩 에이전트지만 is_complex=False이면 delegate로 직행."""
        from coding_agent.agents.orchestrator.agent import _route_after_classify

        state: dict[str, Any] = {"selected_agent": "coder"}
        assert _route_after_classify(state) == "delegate"  # type: ignore[arg-type]

        state2: dict[str, Any] = {"selected_agent": "coder", "is_complex": False}
        assert _route_after_classify(state2) == "delegate"  # type: ignore[arg-type]

    def test_route_to_delegate_for_non_coding(self):
        """selected_agent가 비코딩 에이전트이면 delegate로 직접 라우팅."""
        from coding_agent.agents.orchestrator.agent import _route_after_classify

        state: dict[str, Any] = {"selected_agent": "deep_research"}
        assert _route_after_classify(state) == "delegate"  # type: ignore[arg-type]

    def test_route_none_to_delegate(self):
        """selected_agent가 none이면 delegate로 라우팅 (delegate에서 처리)."""
        from coding_agent.agents.orchestrator.agent import _route_after_classify

        state: dict[str, Any] = {"selected_agent": "none"}
        assert _route_after_classify(state) == "delegate"  # type: ignore[arg-type]


class TestOrchestratorAgentGraph:
    """OrchestratorAgent 그래프 구성 테스트."""

    def test_graph_has_coordinate_node(self):
        """OrchestratorAgent 그래프에 coordinate 노드가 포함된다."""
        from coding_agent.agents.orchestrator.agent import OrchestratorAgent
        from coding_agent.agents.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()
        agent = OrchestratorAgent(config=config)
        assert agent.graph is not None

        # 그래프에 coordinate 노드가 있는지 확인
        node_names = set(agent.graph.nodes.keys())
        assert "coordinate" in node_names
        assert "classify" in node_names
        assert "delegate" in node_names
        assert "respond" in node_names

    def test_node_names_include_coordinate(self):
        """NODE_NAMES에 COORDINATE가 포함된다."""
        from coding_agent.agents.orchestrator.agent import OrchestratorAgent

        assert "COORDINATE" in OrchestratorAgent.NODE_NAMES
        assert OrchestratorAgent.NODE_NAMES["COORDINATE"] == "coordinate"
