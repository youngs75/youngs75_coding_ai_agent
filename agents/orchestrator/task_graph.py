"""DAG 기반 태스크 스케줄링.

서브태스크 간 의존성을 DAG(Directed Acyclic Graph)로 관리하고
위상 정렬(topological sort)을 통해 실행 순서를 결정한다.

사용 예시:
    from coding_agent.agents.orchestrator.task_graph import TaskGraph

    graph = TaskGraph(subtasks)
    if not graph.validate():
        raise ValueError("순환 의존성 감지")

    for wave in graph.execution_waves:
        # 각 웨이브 내 태스크는 병렬 실행 가능
        await execute_parallel(wave)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

from .schemas import SubTask

logger = logging.getLogger(__name__)


class TaskGraph:
    """서브태스크 간 의존성을 DAG로 관리하고 실행 순서를 결정한다.

    위상 정렬(Kahn's algorithm)을 사용하여:
    1. 순환 의존성 검사
    2. 실행 가능한 태스크 파악
    3. 병렬 실행 웨이브 계산
    """

    def __init__(self, subtasks: list[SubTask]) -> None:
        self._subtasks: dict[str, SubTask] = {st["id"]: st for st in subtasks}
        # 인접 리스트: task_id → 이 태스크에 의존하는 태스크 ID 집합
        self._successors: dict[str, set[str]] = defaultdict(set)
        # 각 태스크의 진입 차수 (의존하는 선행 태스크 수)
        self._in_degree: dict[str, int] = {}

        self._build_graph()

    def _build_graph(self) -> None:
        """서브태스크 목록으로부터 DAG 구조를 구축한다."""
        for task_id in self._subtasks:
            self._in_degree.setdefault(task_id, 0)

        for task_id, subtask in self._subtasks.items():
            dep_count = 0
            for dep_id in subtask["dependencies"]:
                if dep_id in self._subtasks:
                    self._successors[dep_id].add(task_id)
                    dep_count += 1
                else:
                    logger.warning(
                        "[TaskGraph] 태스크 '%s'의 의존성 '%s'가 존재하지 않음 — 무시",
                        task_id,
                        dep_id,
                    )
            self._in_degree[task_id] = dep_count

    def validate(self) -> bool:
        """순환 의존성이 없는지 검사한다.

        Kahn's algorithm으로 위상 정렬을 시도하여
        모든 노드가 처리되면 True (DAG), 아니면 False (순환 존재).

        Returns:
            순환 의존성이 없으면 True
        """
        in_degree = dict(self._in_degree)
        queue: deque[str] = deque()
        processed = 0

        # 진입 차수가 0인 노드로 시작
        for task_id, degree in in_degree.items():
            if degree == 0:
                queue.append(task_id)

        while queue:
            current = queue.popleft()
            processed += 1
            for successor in self._successors.get(current, set()):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        is_valid = processed == len(self._subtasks)
        if not is_valid:
            logger.error(
                "[TaskGraph] 순환 의존성 감지: %d/%d 태스크만 처리됨",
                processed,
                len(self._subtasks),
            )
        return is_valid

    def get_ready_tasks(self, completed: set[str]) -> list[SubTask]:
        """의존성이 모두 완료된 실행 가능 태스크를 반환한다.

        Args:
            completed: 이미 완료된 태스크 ID 집합

        Returns:
            현재 실행 가능한 서브태스크 리스트 (우선순위 높은 순)
        """
        ready: list[SubTask] = []
        for task_id, subtask in self._subtasks.items():
            if task_id in completed:
                continue
            # 모든 의존성이 완료되었는지 확인
            deps = subtask["dependencies"]
            existing_deps = [d for d in deps if d in self._subtasks]
            if all(dep_id in completed for dep_id in existing_deps):
                ready.append(subtask)

        # 우선순위 높은 순(값이 클수록 높은 우선순위)으로 정렬
        ready.sort(key=lambda t: t["priority"], reverse=True)
        return ready

    @property
    def execution_waves(self) -> list[list[SubTask]]:
        """위상 정렬 기반 실행 웨이브를 계산한다.

        각 웨이브 내의 태스크는 서로 의존성이 없으므로 병렬 실행 가능.
        웨이브 간에는 순차적으로 실행해야 한다.

        Returns:
            웨이브 리스트 (각 웨이브는 병렬 실행 가능한 SubTask 리스트)

        Raises:
            ValueError: 순환 의존성이 존재하는 경우
        """
        if not self.validate():
            raise ValueError("순환 의존성이 존재하여 실행 웨이브를 계산할 수 없습니다.")

        waves: list[list[SubTask]] = []
        in_degree = dict(self._in_degree)
        remaining = set(self._subtasks.keys())

        while remaining:
            # 현재 진입 차수가 0인 노드가 이번 웨이브
            wave_ids = [
                task_id for task_id in remaining if in_degree.get(task_id, 0) == 0
            ]
            if not wave_ids:
                # 남은 노드가 있지만 진입 차수 0인 노드가 없다면 순환
                raise ValueError("순환 의존성이 존재합니다.")

            wave = [self._subtasks[tid] for tid in wave_ids]
            # 우선순위 순 정렬
            wave.sort(key=lambda t: t["priority"], reverse=True)
            waves.append(wave)

            # 처리된 노드 제거 및 후속 노드의 진입 차수 감소
            for task_id in wave_ids:
                remaining.discard(task_id)
                for successor in self._successors.get(task_id, set()):
                    in_degree[successor] -= 1

        return waves

    @property
    def subtask_count(self) -> int:
        """등록된 서브태스크 수."""
        return len(self._subtasks)

    def get_subtask(self, task_id: str) -> SubTask | None:
        """ID로 서브태스크 조회."""
        return self._subtasks.get(task_id)
