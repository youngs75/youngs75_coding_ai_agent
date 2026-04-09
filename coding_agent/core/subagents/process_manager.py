"""동적 SubAgent 프로세스 매니저.

Claude Code 스타일로 에이전트를 별도 asyncio subprocess로 spawn하고,
작업 완료 시 프로세스를 종료하며 자원을 회수한다.

수명주기:
    spawn() → 자식 프로세스 생성 (CREATED → ASSIGNED → RUNNING)
    ↓
    자식 프로세스에서 에이전트 실행 (worker.py 엔트리포인트)
    ↓
    결과 수신 (stdout JSON) → COMPLETED / FAILED
    ↓
    프로세스 종료 확인 → DESTROYED (자원 회수)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import SubAgentRegistry
from .schemas import ResourceUsage, SubAgentInstance, SubAgentResult, SubAgentStatus

logger = logging.getLogger(__name__)

# 활성 상태 집합 (좀비 프로세스 감지에 사용)
_ACTIVE_STATES = {
    SubAgentStatus.CREATED,
    SubAgentStatus.ASSIGNED,
    SubAgentStatus.RUNNING,
    SubAgentStatus.BLOCKED,
}

# 종료 대기 시간 (SIGTERM 후 SIGKILL까지)
_GRACEFUL_SHUTDOWN_S = 3.0


class SubAgentProcessManager:
    """SubAgent 프로세스 수명주기 관리자.

    asyncio subprocess로 에이전트를 spawn하고, 작업 완료 시
    프로세스를 종료하며 자원을 회수한다. 타임아웃, 동시 실행 제한,
    자동 재시도, 좀비 프로세스 정리를 지원한다.

    Args:
        registry: SubAgentRegistry 인스턴스 (인스턴스 생성/상태 관리).
        timeout_s: 프로세스별 기본 타임아웃(초).
        max_concurrent: 최대 동시 실행 프로세스 수.
    """

    def __init__(
        self,
        registry: SubAgentRegistry,
        timeout_s: float = 300.0,
        max_concurrent: int = 5,
    ):
        self._registry = registry
        self._timeout_s = timeout_s
        self._max_concurrent = max_concurrent
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._resource_usage: dict[str, ResourceUsage] = {}
        self._temp_files: dict[str, list[str]] = {}  # agent_id → 임시파일 경로들

    async def spawn(
        self,
        agent_type: str,
        task_message: str,
        *,
        parent_id: str | None = None,
        task_plan: str | None = None,
        timeout_s: float | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SubAgentInstance:
        """에이전트를 별도 프로세스로 spawn한다.

        1. registry.create_instance()로 인스턴스 생성 (CREATED)
        2. asyncio.create_subprocess_exec()로 자식 프로세스 생성
        3. 상태 전이: CREATED → ASSIGNED → RUNNING

        Args:
            agent_type: 에이전트 타입 이름 (SubAgentSpec.name).
            task_message: 에이전트에게 전달할 작업 메시지.
            parent_id: 부모 에이전트/오케스트레이터 ID.
            task_plan: 작업 계획 텍스트 (있으면 임시파일로 전달).
            timeout_s: 개별 타임아웃 오버라이드(초).
            env_overrides: 환경변수 오버라이드 딕셔너리.

        Returns:
            생성된 SubAgentInstance.

        Raises:
            RuntimeError: 동시 실행 한도 초과 또는 인스턴스 생성 실패 시.
        """
        # 동시 실행 제한 확인
        if self.active_count >= self._max_concurrent:
            await self._reap_zombies()
            if self.active_count >= self._max_concurrent:
                raise RuntimeError(
                    f"동시 실행 한도 초과: {self.active_count}/{self._max_concurrent}"
                )

        # 1. 인스턴스 생성
        instance = self._registry.create_instance(
            spec_name=agent_type,
            role=f"{agent_type} worker",
            task_summary=task_message[:200],
            parent_id=parent_id,
        )
        if instance is None:
            # spec이 없으면 자동 등록 후 재시도
            from .schemas import SubAgentSpec

            self._registry.register(SubAgentSpec(
                name=agent_type,
                description=f"Dynamic {agent_type} agent",
                capabilities=[agent_type],
            ))
            instance = self._registry.create_instance(
                spec_name=agent_type,
                role=f"{agent_type} worker",
                task_summary=task_message[:200],
                parent_id=parent_id,
            )
            if instance is None:
                raise RuntimeError(f"인스턴스 생성 실패: {agent_type}")

        agent_id = instance.agent_id

        # 2. 임시파일로 태스크 메시지 전달 (메시지가 길 수 있으므로)
        temp_files: list[str] = []
        task_msg_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="subagent_task_"
        )
        task_msg_file.write(task_message)
        task_msg_file.close()
        temp_files.append(task_msg_file.name)

        cmd = [
            sys.executable, "-m", "coding_agent.core.subagents.worker",
            "--agent-type", agent_type,
            "--task-message-file", task_msg_file.name,
            "--parent-id", parent_id or "",
        ]

        if task_plan:
            plan_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, prefix="subagent_plan_"
            )
            plan_file.write(task_plan)
            plan_file.close()
            temp_files.append(plan_file.name)
            cmd.extend(["--task-plan-file", plan_file.name])

        self._temp_files[agent_id] = temp_files

        # 3. 환경변수: 현재 프로세스 환경 상속 + 오버라이드
        env = {**os.environ, **(env_overrides or {})}

        # 4. 상태 전이: CREATED → ASSIGNED
        self._registry.transition_state(
            agent_id, SubAgentStatus.ASSIGNED,
            reason=f"Task assigned: {agent_type}",
        )

        # 5. 프로세스 spawn
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._processes[agent_id] = proc
        self._resource_usage[agent_id] = ResourceUsage(
            pid=proc.pid,
            agent_id=agent_id,
            start_time=time.time(),
        )

        # 6. ASSIGNED → RUNNING
        self._registry.transition_state(
            agent_id, SubAgentStatus.RUNNING,
            reason=f"Process spawned (pid={proc.pid})",
        )

        logger.info(
            "[ProcessManager] spawn: agent_id=%s, type=%s, pid=%d",
            agent_id[:12], agent_type, proc.pid,
        )

        return instance

    async def wait(self, agent_id: str, timeout_s: float | None = None) -> SubAgentResult:
        """프로세스 완료를 대기하고 결과를 반환한다.

        Args:
            agent_id: 대기할 에이전트 인스턴스 ID.
            timeout_s: 타임아웃 오버라이드(초). None이면 기본값 사용.

        Returns:
            SubAgentResult (성공/실패 정보 포함).
        """
        proc = self._processes.get(agent_id)
        if proc is None:
            return SubAgentResult(
                status="failed",
                error=f"프로세스를 찾을 수 없음: {agent_id}",
            )

        effective_timeout = timeout_s or self._timeout_s

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            # 타임아웃: SIGTERM → 대기 → SIGKILL
            logger.warning(
                "[ProcessManager] 타임아웃 (%ds): agent_id=%s, pid=%d",
                effective_timeout, agent_id[:12], proc.pid,
            )
            await self._kill_process(proc)
            self._registry.transition_state(
                agent_id, SubAgentStatus.FAILED,
                error_message=f"Process timeout ({effective_timeout}s)",
            )
            self._finalize_resource(agent_id, proc.returncode)
            return SubAgentResult(
                status="failed",
                error=f"타임아웃: {effective_timeout}초 초과",
                duration_s=effective_timeout,
            )

        # 자원 사용량 기록
        self._finalize_resource(agent_id, proc.returncode)

        # stderr 로그 출력
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if stderr_text:
            for line in stderr_text.split("\n")[-20:]:  # 마지막 20줄만
                logger.debug("[worker:%s] %s", agent_id[:8], line)

        # stdout에서 JSON 결과 파싱
        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not stdout_text:
            self._registry.transition_state(
                agent_id, SubAgentStatus.FAILED,
                error_message="Worker produced no output",
            )
            return SubAgentResult(status="failed", error="워커 출력 없음")

        try:
            result_data = json.loads(stdout_text)
            result = SubAgentResult(**result_data)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(
                "[ProcessManager] JSON 파싱 실패: %s (stdout=%s)",
                e, stdout_text[:200],
            )
            self._registry.transition_state(
                agent_id, SubAgentStatus.FAILED,
                error_message=f"JSON parse error: {e}",
            )
            return SubAgentResult(status="failed", error=f"결과 파싱 실패: {e}")

        # 상태 전이
        if result.success:
            self._registry.transition_state(
                agent_id, SubAgentStatus.COMPLETED,
                result_summary=str(result.result)[:500] if result.result else "",
            )
        else:
            self._registry.transition_state(
                agent_id, SubAgentStatus.FAILED,
                error_message=result.error or "Unknown error",
            )

        logger.info(
            "[ProcessManager] 완료: agent_id=%s, status=%s, duration=%.1fs",
            agent_id[:12], result.status, result.duration_s,
        )

        # 결과 영속화
        instance = self._registry.get_instance(agent_id)
        self._persist_result(agent_id, instance, result)

        return result

    async def spawn_and_wait(
        self,
        agent_type: str,
        task_message: str,
        **kwargs: Any,
    ) -> SubAgentResult:
        """spawn + wait를 한번에 수행하는 편의 메서드.

        실패 시 retry_count < max_retries이면 자동 재시도한다.

        Args:
            agent_type: 에이전트 타입 이름.
            task_message: 작업 메시지.
            **kwargs: spawn()에 전달할 추가 인자.

        Returns:
            최종 SubAgentResult.
        """
        instance = await self.spawn(agent_type, task_message, **kwargs)
        result = await self.wait(instance.agent_id, kwargs.get("timeout_s"))

        # 실패 시 재시도 로직
        while not result.success and instance.retry_count < instance.max_retries:
            instance.retry_count += 1
            logger.info(
                "[ProcessManager] 재시도 %d/%d: agent_id=%s, type=%s",
                instance.retry_count, instance.max_retries,
                instance.agent_id[:12], agent_type,
            )

            # FAILED → ASSIGNED → RUNNING (새 프로세스 spawn)
            self._registry.transition_state(
                instance.agent_id, SubAgentStatus.ASSIGNED,
                reason=f"Retry {instance.retry_count}/{instance.max_retries}",
            )

            # 기존 프로세스 정리 후 새 프로세스 생성
            self._processes.pop(instance.agent_id, None)

            cmd = self._build_cmd(agent_type, task_message, kwargs)
            env = {**os.environ, **(kwargs.get("env_overrides") or {})}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._processes[instance.agent_id] = proc
            self._resource_usage[instance.agent_id] = ResourceUsage(
                pid=proc.pid,
                agent_id=instance.agent_id,
                start_time=time.time(),
            )

            self._registry.transition_state(
                instance.agent_id, SubAgentStatus.RUNNING,
                reason=f"Retry process spawned (pid={proc.pid})",
            )

            result = await self.wait(instance.agent_id, kwargs.get("timeout_s"))

        # 완료 후 자원 정리 (DESTROYED 전이)
        await self._destroy_instance(instance.agent_id)
        return result

    async def cancel(self, agent_id: str, reason: str = "user_cancel") -> bool:
        """실행 중인 프로세스를 취소한다.

        Args:
            agent_id: 취소할 에이전트 인스턴스 ID.
            reason: 취소 사유.

        Returns:
            취소 성공 시 True, 프로세스 미존재 또는 이미 종료 시 False.
        """
        proc = self._processes.get(agent_id)
        if proc is None or proc.returncode is not None:
            return False

        logger.info("[ProcessManager] cancel: agent_id=%s, reason=%s", agent_id[:12], reason)

        # SIGTERM → 대기 → SIGKILL
        await self._kill_process(proc)

        self._registry.transition_state(
            agent_id, SubAgentStatus.CANCELLED,
            reason=reason,
        )
        self._finalize_resource(agent_id, proc.returncode)
        await self._destroy_instance(agent_id)
        return True

    async def cleanup_all(self) -> int:
        """모든 완료/실패 인스턴스의 프로세스를 정리한다.

        Returns:
            정리된 프로세스 수.
        """
        cleaned = 0
        for agent_id in list(self._processes.keys()):
            proc = self._processes[agent_id]
            if proc.returncode is not None:
                await self._destroy_instance(agent_id)
                cleaned += 1
        return cleaned

    async def _reap_zombies(self) -> None:
        """좀비 프로세스를 정리한다."""
        for agent_id, proc in list(self._processes.items()):
            if proc.returncode is not None:
                instance = self._registry.get_instance(agent_id)
                if instance and instance.state in _ACTIVE_STATES:
                    self._registry.transition_state(
                        agent_id, SubAgentStatus.FAILED,
                        error_message=f"Process died unexpectedly (exit={proc.returncode})",
                    )
                await self._destroy_instance(agent_id)

    async def _destroy_instance(self, agent_id: str) -> None:
        """인스턴스를 DESTROYED로 전이하고 자원을 정리한다."""
        instance = self._registry.get_instance(agent_id)
        if instance and instance.state != SubAgentStatus.DESTROYED:
            # DESTROYED로 전이 가능한 상태인지 확인
            from .schemas import VALID_TRANSITIONS
            if SubAgentStatus.DESTROYED in VALID_TRANSITIONS.get(instance.state, set()):
                self._registry.transition_state(
                    agent_id, SubAgentStatus.DESTROYED,
                    reason="Process cleanup",
                )

        # 프로세스 객체 제거
        self._processes.pop(agent_id, None)

        # 임시파일 정리
        for path in self._temp_files.pop(agent_id, []):
            try:
                os.unlink(path)
            except OSError:
                pass

    def _finalize_resource(self, agent_id: str, exit_code: int | None) -> None:
        """자원 사용량을 확정한다."""
        usage = self._resource_usage.get(agent_id)
        if usage:
            usage.end_time = time.time()
            usage.exit_code = exit_code

    def _build_cmd(
        self,
        agent_type: str,
        task_message: str,
        kwargs: dict[str, Any],
    ) -> list[str]:
        """워커 프로세스 실행 명령어를 생성한다."""
        # 임시파일로 태스크 메시지 전달
        task_msg_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="subagent_task_"
        )
        task_msg_file.write(task_message)
        task_msg_file.close()

        cmd = [
            sys.executable, "-m", "coding_agent.core.subagents.worker",
            "--agent-type", agent_type,
            "--task-message-file", task_msg_file.name,
            "--parent-id", kwargs.get("parent_id") or "",
        ]

        task_plan = kwargs.get("task_plan")
        if task_plan:
            plan_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, prefix="subagent_plan_"
            )
            plan_file.write(task_plan)
            plan_file.close()
            cmd.extend(["--task-plan-file", plan_file.name])

        return cmd

    def _persist_result(
        self,
        agent_id: str,
        instance: SubAgentInstance | None,
        result: SubAgentResult,
    ) -> None:
        """SubAgentResult를 workspace의 .ai/subagent_results/results.jsonl에 영속화한다."""
        workspace = os.environ.get("WORKSPACE", os.getcwd())
        results_dir = Path(workspace) / ".ai" / "subagent_results"
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "agent_id": agent_id,
                "spec_name": instance.spec_name if instance else "unknown",
                "task_summary": instance.task_summary if instance else "",
                "status": result.status,
                "result_summary": (result.result or "")[:500],
                "duration_s": result.duration_s,
                "token_usage": result.token_usage,
                "error": result.error,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(results_dir / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[ProcessManager] 결과 영속화 실패: %s", e)

    @staticmethod
    async def _kill_process(proc: asyncio.subprocess.Process) -> None:
        """SIGTERM → 대기 → SIGKILL 순서로 프로세스를 종료한다."""
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=_GRACEFUL_SHUTDOWN_S)
        except (asyncio.TimeoutError, ProcessLookupError):
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

    def get_resource_usage(self, agent_id: str) -> ResourceUsage | None:
        """프로세스의 자원 사용량을 조회한다.

        Args:
            agent_id: 조회할 에이전트 인스턴스 ID.

        Returns:
            ResourceUsage 또는 None (미존재 시).
        """
        return self._resource_usage.get(agent_id)

    @property
    def active_count(self) -> int:
        """현재 실행 중인 프로세스 수."""
        return sum(1 for p in self._processes.values() if p.returncode is None)

    @property
    def all_processes(self) -> dict[str, asyncio.subprocess.Process]:
        """등록된 모든 프로세스 (디버깅용)."""
        return dict(self._processes)
