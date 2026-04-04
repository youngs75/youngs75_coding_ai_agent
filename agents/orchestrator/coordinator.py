"""Coordinator Mode — 로컬 병렬 워커 오케스트레이션.

복잡한 작업을 서브태스크로 분해하고 병렬 워커에 위임하는 코디네이터.
Claude Code의 Agent 팀 패턴 참고.

파이프라인:
    decompose (LLM 기반 작업 분해)
    → execute_parallel (DAG 기반 병렬 실행)
    → synthesize (결과 통합)

사용 예시:
    from youngs75_a2a.agents.orchestrator.coordinator import CoordinatorMode

    coordinator = CoordinatorMode(registry=registry, context_manager=ctx_mgr)
    result = await coordinator.run(task="...", context=messages, llm=llm)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from youngs75_a2a.core.batch_executor import BatchExecutor
from youngs75_a2a.core.context_manager import ContextManager
from youngs75_a2a.core.subagents.registry import SubAgentRegistry
from youngs75_a2a.core.subagents.schemas import SubAgentSpec

from .schemas import CoordinatorResult, SubTask, WorkerResult
from .task_graph import TaskGraph

logger = logging.getLogger(__name__)

# ── 프롬프트 ────────────────────────────────────────────────

_DECOMPOSE_SYSTEM_PROMPT = """\
당신은 복합 작업을 병렬 실행 가능한 서브태스크로 분해하는 전문가입니다.

사용 가능한 에이전트:
{agent_descriptions}

규칙:
1. 작업을 독립적으로 실행 가능한 서브태스크로 분해하세요.
2. 각 서브태스크에 가장 적합한 에이전트 타입을 할당하세요.
3. 서브태스크 간 의존성이 있으면 dependencies에 선행 태스크 ID를 명시하세요.
4. 가능한 한 많은 태스크가 병렬 실행될 수 있도록 분해하세요.
5. 서브태스크 수는 최소 1개, 최대 10개로 제한하세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
```json
[
  {{
    "id": "task_1",
    "description": "서브태스크 설명",
    "agent_type": "에이전트 이름",
    "dependencies": [],
    "priority": 1,
    "timeout_s": 60.0
  }}
]
```
"""

_SYNTHESIZE_SYSTEM_PROMPT = """\
당신은 여러 워커의 결과를 통합하여 최종 응답을 생성하는 전문가입니다.

원래 작업: {original_task}

아래는 각 워커의 실행 결과입니다. 모든 결과를 종합하여 사용자에게 명확하고 일관된 응답을 생성하세요.
실패한 워커가 있다면 그 부분을 보완하거나 실패 사실을 언급하세요.
"""


class CoordinatorMode:
    """복잡한 작업을 서브태스크로 분해하고 병렬 워커에 위임하는 코디네이터.

    Claude Code의 Agent 팀 패턴을 참고하여,
    DAG 기반 태스크 스케줄링과 asyncio 병렬 실행을 결합한다.

    Args:
        registry: 서브에이전트 레지스트리 (에이전트 선택 + 성과 추적)
        context_manager: 컨텍스트 매니저 (서브에이전트용 히스토리 필터링)
        max_workers: 최대 동시 워커 수
        timeout_s: 전체 코디네이터 타임아웃 (초)
        worker_fn: 워커 실행 함수 (테스트용 주입 포인트).
            None이면 기본 A2A 위임 사용.
    """

    def __init__(
        self,
        registry: SubAgentRegistry,
        context_manager: ContextManager,
        max_workers: int = 5,
        timeout_s: float = 300.0,
        worker_fn: Any | None = None,
    ) -> None:
        self._registry = registry
        self._context_manager = context_manager
        self._max_workers = max_workers
        self._timeout_s = timeout_s
        self._worker_fn = worker_fn
        self._executor = BatchExecutor(
            max_concurrency=max_workers,
            timeout_s=timeout_s,
        )

    # ── 1단계: 태스크 분해 ──────────────────────────────────

    async def decompose_task(
        self,
        task: str,
        llm: BaseChatModel,
        available_agents: list[SubAgentSpec],
    ) -> list[SubTask]:
        """LLM을 사용해 복합 작업을 병렬 가능한 서브태스크로 분해한다.

        Args:
            task: 사용자의 원래 작업 설명
            llm: 분해에 사용할 LLM
            available_agents: 사용 가능한 에이전트 목록

        Returns:
            분해된 서브태스크 리스트
        """
        agent_descriptions = "\n".join(
            f"- {a.name}: {a.description} (능력: {', '.join(a.capabilities)})"
            for a in available_agents
        )

        system_prompt = _DECOMPOSE_SYSTEM_PROMPT.format(
            agent_descriptions=agent_descriptions
        )

        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"작업: {task}"),
            ]
        )

        return self._parse_subtasks(response.content)

    def _parse_subtasks(self, llm_output: str | Any) -> list[SubTask]:
        """LLM 출력에서 서브태스크 리스트를 파싱한다."""
        if not isinstance(llm_output, str):
            llm_output = str(llm_output)

        # JSON 블록 추출 (```json ... ``` 또는 순수 JSON)
        text = llm_output.strip()
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + len("```")
            end = text.index("```", start)
            text = text[start:end].strip()

        try:
            raw_tasks = json.loads(text)
        except json.JSONDecodeError:
            logger.error("[CoordinatorMode] 서브태스크 파싱 실패: %s", text[:200])
            # 파싱 실패 시 단일 태스크로 폴백
            return [
                SubTask(
                    id="task_fallback",
                    description=llm_output[:500] if isinstance(llm_output, str) else "작업 실행",
                    agent_type="default",
                    dependencies=[],
                    priority=1,
                    timeout_s=60.0,
                )
            ]

        subtasks: list[SubTask] = []
        for raw in raw_tasks:
            subtask = SubTask(
                id=raw.get("id", f"task_{uuid.uuid4().hex[:8]}"),
                description=raw.get("description", ""),
                agent_type=raw.get("agent_type", "default"),
                dependencies=raw.get("dependencies", []),
                priority=raw.get("priority", 1),
                timeout_s=raw.get("timeout_s", 60.0),
            )
            subtasks.append(subtask)

        return subtasks if subtasks else [
            SubTask(
                id="task_fallback",
                description="작업 실행",
                agent_type="default",
                dependencies=[],
                priority=1,
                timeout_s=60.0,
            )
        ]

    # ── 2단계: 병렬 실행 ───────────────────────────────────

    async def execute_parallel(
        self,
        subtasks: list[SubTask],
        parent_context: list[BaseMessage],
    ) -> CoordinatorResult:
        """서브태스크를 적절한 워커에 배분하고 병렬 실행한다.

        DAG 의존성을 고려하여 웨이브 단위로 실행한다.
        각 웨이브 내 태스크는 병렬 실행, 웨이브 간에는 순차 실행.

        Args:
            subtasks: 실행할 서브태스크 리스트
            parent_context: 부모 컨텍스트 메시지 (서브에이전트에 전달)

        Returns:
            CoordinatorResult (synthesized_response 제외 — synthesize에서 채움)
        """
        start_time = time.perf_counter()

        # DAG 구성 및 검증
        task_graph = TaskGraph(subtasks)
        if not task_graph.validate():
            logger.error("[CoordinatorMode] 순환 의존성 감지 — 의존성 무시하고 모두 병렬 실행")
            # 순환 의존성 시 모든 의존성 제거 후 단일 웨이브로 실행
            for st in subtasks:
                st["dependencies"] = []
            task_graph = TaskGraph(subtasks)

        waves = task_graph.execution_waves
        all_results: list[WorkerResult] = []
        completed: set[str] = set()

        # 서브에이전트용 컨텍스트 준비
        truncated_context = self._context_manager.truncate_for_subagent(
            parent_context, last_n_turns=3
        )

        for wave_idx, wave in enumerate(waves):
            logger.info(
                "[CoordinatorMode] 웨이브 %d/%d 실행: %d개 태스크",
                wave_idx + 1,
                len(waves),
                len(wave),
            )

            # 웨이브 내 태스크를 병렬 실행
            wave_results = await self._execute_wave(wave, truncated_context)
            all_results.extend(wave_results)

            # 완료된 태스크 추적
            for wr in wave_results:
                if wr["status"] == "success":
                    completed.add(wr["subtask_id"])

        total_duration = time.perf_counter() - start_time

        # 순차 실행 소요 시간 추정 (각 워커의 실행 시간 합)
        sequential_duration = sum(wr["duration_s"] for wr in all_results)
        parallel_efficiency = (
            total_duration / sequential_duration
            if sequential_duration > 0
            else 1.0
        )

        return CoordinatorResult(
            synthesized_response="",  # synthesize 단계에서 채움
            worker_results=all_results,
            total_duration_s=total_duration,
            parallel_efficiency=parallel_efficiency,
        )

    async def _execute_wave(
        self,
        wave: list[SubTask],
        context: list[BaseMessage],
    ) -> list[WorkerResult]:
        """단일 웨이브의 태스크를 병렬 실행한다."""
        if not wave:
            return []

        # 각 태스크를 비동기 callable로 변환
        tasks = [
            self._make_worker_task(subtask, context)
            for subtask in wave
        ]

        batch_result = await self._executor.execute(tasks)

        results: list[WorkerResult] = []
        for i, task_result in enumerate(batch_result.results):
            subtask = wave[i]
            if task_result.success:
                results.append(
                    WorkerResult(
                        subtask_id=subtask["id"],
                        agent_name=subtask["agent_type"],
                        status="success",
                        output=str(task_result.value) if task_result.value else "",
                        duration_s=task_result.duration_s,
                        error=None,
                    )
                )
            else:
                error_msg = str(task_result.error) if task_result.error else "알 수 없는 오류"
                # 타임아웃 구분
                status = "timeout" if "timeout" in error_msg.lower() or isinstance(
                    task_result.error, asyncio.TimeoutError
                ) else "failed"
                results.append(
                    WorkerResult(
                        subtask_id=subtask["id"],
                        agent_name=subtask["agent_type"],
                        status=status,
                        output="",
                        duration_s=task_result.duration_s,
                        error=error_msg,
                    )
                )

        return results

    def _make_worker_task(
        self,
        subtask: SubTask,
        context: list[BaseMessage],
    ) -> Any:
        """서브태스크를 위한 비동기 callable을 생성한다."""
        async def _execute() -> str:
            if self._worker_fn is not None:
                # 테스트용 커스텀 워커 함수
                return await self._worker_fn(subtask, context)

            # 기본 구현: 레지스트리에서 에이전트를 선택하여 실행
            selection = self._registry.select(
                task_type=subtask["agent_type"],
            )
            if selection is None:
                raise ValueError(
                    f"서브태스크 '{subtask['id']}'에 적합한 에이전트를 찾을 수 없습니다."
                )

            agent = selection.agent
            logger.info(
                "[CoordinatorMode] 서브태스크 '%s' → 에이전트 '%s' (점수: %.2f)",
                subtask["id"],
                agent.name,
                selection.score,
            )

            # A2A 프로토콜로 위임 (기존 delegate 패턴 재사용)
            if agent.endpoint:
                return await self._delegate_to_agent(
                    agent, subtask["description"], context
                )
            else:
                # 로컬 에이전트 — 설명만 반환 (실제 구현은 에이전트에 위임)
                return f"[{agent.name}] {subtask['description']} — 처리 완료"

        return _execute

    async def _delegate_to_agent(
        self,
        agent: SubAgentSpec,
        task_description: str,
        context: list[BaseMessage],
    ) -> str:
        """A2A 프로토콜로 에이전트에 작업을 위임한다."""
        import httpx
        from a2a.client import A2AClient
        from a2a.client.helpers import create_text_message_object
        from a2a.types import MessageSendParams, SendMessageRequest

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout_s)
        ) as hc:
            client = A2AClient(httpx_client=hc, url=agent.endpoint)
            msg = create_text_message_object(content=task_description)
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(message=msg),
            )
            response = await client.send_message(request)

        # 응답에서 텍스트 추출
        result = response.root
        if hasattr(result, "result"):
            obj = result.result
            if hasattr(obj, "artifacts") and obj.artifacts:
                for artifact in obj.artifacts:
                    for part in artifact.parts or []:
                        root = getattr(part, "root", part)
                        if hasattr(root, "text") and root.text and len(root.text) > 5:
                            return root.text

        return "[에이전트 응답 파싱 실패]"

    # ── 3단계: 결과 통합 ───────────────────────────────────

    async def synthesize(
        self,
        original_task: str,
        results: list[WorkerResult],
        llm: BaseChatModel,
    ) -> str:
        """개별 워커 결과를 통합하여 최종 응답을 생성한다.

        Args:
            original_task: 사용자의 원래 작업
            results: 워커 실행 결과 리스트
            llm: 통합에 사용할 LLM

        Returns:
            통합된 최종 응답 문자열
        """
        system_prompt = _SYNTHESIZE_SYSTEM_PROMPT.format(
            original_task=original_task
        )

        # 워커 결과를 텍스트로 포맷팅
        results_text_parts: list[str] = []
        for wr in results:
            status_emoji = {
                "success": "[성공]",
                "failed": "[실패]",
                "timeout": "[타임아웃]",
                "cancelled": "[취소]",
            }.get(wr["status"], "[알 수 없음]")

            part = f"### 서브태스크: {wr['subtask_id']} ({wr['agent_name']}) {status_emoji}\n"
            if wr["status"] == "success":
                part += wr["output"]
            else:
                part += f"오류: {wr['error'] or '알 수 없는 오류'}"
            results_text_parts.append(part)

        results_text = "\n\n".join(results_text_parts)

        response = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=results_text),
            ]
        )

        content = response.content
        return content if isinstance(content, str) else str(content)

    # ── 전체 파이프라인 ────────────────────────────────────

    async def run(
        self,
        task: str,
        context: list[BaseMessage],
        llm: BaseChatModel,
    ) -> CoordinatorResult:
        """전체 파이프라인: decompose → execute_parallel → synthesize.

        Args:
            task: 사용자의 원래 작업 설명
            context: 대화 컨텍스트 메시지
            llm: 사용할 LLM

        Returns:
            CoordinatorResult: 전체 코디네이션 결과
        """
        logger.info("[CoordinatorMode] 파이프라인 시작: %s", task[:100])

        # 1. 사용 가능한 에이전트 목록 조회
        available_agents = self._registry.list_available()

        # 2. 태스크 분해
        subtasks = await self.decompose_task(task, llm, available_agents)
        logger.info(
            "[CoordinatorMode] %d개 서브태스크로 분해됨",
            len(subtasks),
        )

        # 3. 병렬 실행
        result = await self.execute_parallel(subtasks, context)

        # 4. 결과 통합
        synthesized = await self.synthesize(task, result["worker_results"], llm)
        result["synthesized_response"] = synthesized

        logger.info(
            "[CoordinatorMode] 파이프라인 완료: %.2fs (효율성: %.2f)",
            result["total_duration_s"],
            result["parallel_efficiency"],
        )

        return result
