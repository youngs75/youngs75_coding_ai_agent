"""
A2A AgentExecutor 구현 - 2가지 버전

1. BaseAgentExecutor: 일반 async callable을 A2A로 래핑 (LangGraph 없이)
2. LGAgentExecutor: LangGraph CompiledStateGraph를 A2A로 래핑
"""

from typing import Any, AsyncIterator, Callable
import asyncio
import logging
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Part, TaskState, TextPart, DataPart, TaskNotFoundError
from a2a.utils import new_agent_text_message, new_task, get_data_parts
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. BaseAgentExecutor — 일반 async callable 용
# ---------------------------------------------------------------------------
# 스트리밍: agent_fn이 AsyncIterator[str]를 반환하면 스트리밍
# Polling:  agent_fn이 str을 반환하면 단일 응답 (polling)
# 취소:     asyncio.Task.cancel()로 즉시 취소 (CancelledError 전파)

AgentFn = Callable[[str, dict[str, Any]], Any]
# agent_fn(query: str, context: dict) -> str | AsyncIterator[str]


class BaseAgentExecutor(AgentExecutor):
    """LangGraph가 아닌 일반 에이전트를 A2A로 래핑하는 Executor.

    agent_fn의 반환 타입에 따라 자동으로 스트리밍/폴링을 구분:
    - str 반환 → Polling 방식 (한 번에 결과 전달)
    - AsyncIterator[str] 반환 → Streaming 방식 (청크 단위 전달)

    취소는 asyncio.Task.cancel()로 CancelledError를 전파하여 즉시 중단.
    """

    def __init__(
        self,
        agent_fn: AgentFn,
        *,
        execution_timeout: float | None = None,
    ):
        self.agent_fn = agent_fn
        self._execution_timeout = execution_timeout
        # 실행 중인 asyncio.Task를 task_id별로 추적 (취소용)
        self._running_tasks: dict[str, asyncio.Task] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input() or ""
        logger.info(f"BaseAgentExecutor 요청 수신: {query[:100]}")

        # 태스크 생성
        task = context.current_task
        if not task:
            if not context.message:
                raise ServerError(error=InternalError())
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.submitted)

        task_id = str(task.id)

        try:
            await updater.start_work()

            # agent_fn 호출을 asyncio.Task로 감싸서 취소 가능하게 만든다
            async def _run():
                return await self.agent_fn(query, {"task_id": task_id})

            run_task = asyncio.create_task(_run())
            self._running_tasks[task_id] = run_task

            try:
                result = await asyncio.wait_for(
                    run_task, timeout=self._execution_timeout
                )
            except asyncio.TimeoutError:
                run_task.cancel()
                self._running_tasks.pop(task_id, None)
                logger.warning(
                    f"BaseAgentExecutor 실행 타임아웃: {task_id} "
                    f"({self._execution_timeout}초)"
                )
                timeout_msg = new_agent_text_message(
                    f"실행 시간 초과 ({self._execution_timeout}초)",
                    task.context_id, task.id,
                )
                await updater.failed(message=timeout_msg)
                return
            finally:
                self._running_tasks.pop(task_id, None)

            # 반환 타입에 따라 스트리밍 vs 폴링 분기
            if hasattr(result, "__aiter__"):
                # --- Streaming 방식 ---
                accumulated = ""
                async for chunk in result:
                    if isinstance(chunk, str) and chunk:
                        accumulated += chunk
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message(chunk, task.context_id, task.id),
                        )
                final_text = accumulated or "결과를 생성하지 못했습니다."
            else:
                # --- Polling 방식 ---
                final_text = str(result) if result else "결과를 생성하지 못했습니다."

            await updater.add_artifact([Part(root=TextPart(text=final_text))])
            await updater.complete()

        except asyncio.CancelledError:
            # asyncio.Task.cancel()에 의한 즉시 취소
            logger.info(f"BaseAgentExecutor 작업 취소됨: {task_id}")
            message = new_agent_text_message(
                "작업이 취소되었습니다.", task.context_id, task.id
            )
            await updater.cancel(message)

        except Exception as e:
            logger.error(f"BaseAgentExecutor 실행 오류: {e}")
            error_message = new_agent_text_message(
                f"실행 중 오류: {e}", task.context_id, task.id
            )
            await updater.failed(message=error_message)
            raise ServerError(error=InternalError()) from e

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """asyncio.Task.cancel()로 실행 중인 작업을 즉시 취소."""
        task = context.current_task
        if not task:
            raise ServerError(error=TaskNotFoundError())

        task_id = str(task.id)
        running = self._running_tasks.get(task_id)
        if running and not running.done():
            running.cancel()  # CancelledError가 execute()로 전파됨
            logger.info(f"BaseAgentExecutor 취소 요청: {task_id}")
        else:
            # 이미 완료되었거나 찾을 수 없는 경우
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            message = new_agent_text_message(
                "작업이 취소되었습니다.", task.context_id, task.id
            )
            await updater.cancel(message)


# ---------------------------------------------------------------------------
# 2. LGAgentExecutor — LangGraph 전용
# ---------------------------------------------------------------------------
# 스트리밍: graph.astream()으로 노드 단위 스트리밍
# 취소:     스트림 루프 폴링 + asyncio.Task.cancel() 하이브리드

class LGAgentExecutor(AgentExecutor):
    """LangGraph CompiledStateGraph를 A2A로 래핑하는 Executor.

    스트리밍은 graph.astream()을 사용하며,
    취소는 두 가지 메커니즘을 하이브리드로 제공:
    1) 스트림 루프 폴링: 노드 사이에서 취소 플래그 확인 (협력적)
    2) asyncio.Task.cancel(): 노드 실행 중에도 즉시 취소 (강제적)
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        result_extractor: Callable[[dict[str, Any]], str] | None = None,
        *,
        execution_timeout: float | None = None,
    ):
        self.graph = graph
        self._extract_result = result_extractor or self._default_extract_text
        self._execution_timeout = execution_timeout
        self._cancelled_task_ids: set[str] = set()
        self._running_tasks: dict[str, asyncio.Task] = {}

    def _default_extract_text(self, result: dict[str, Any]) -> str:
        """LangGraph 스트림 청크에서 마지막 AI 메시지의 텍스트를 추출."""
        if not isinstance(result, dict):
            return ""

        # messages 리스트 탐색 (최상위 또는 1단계 중첩)
        messages = None
        if isinstance(result.get("messages"), list):
            messages = result["messages"]
        else:
            for value in result.values():
                if isinstance(value, dict) and isinstance(value.get("messages"), list):
                    messages = value["messages"]
                    break

        if not messages:
            return ""

        # 마지막 AIMessage의 content 추출
        try:
            filtered = filter_messages(messages, include_types=[AIMessage])
            if filtered:
                content = filtered[-1].content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
        except Exception:
            pass
        return ""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input() or ""
        logger.info(f"LGAgentExecutor 요청 수신: {query[:100]}")

        # 태스크 생성
        task = context.current_task
        if not task:
            if not context.message:
                raise ServerError(error=InternalError())
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.submitted)

        task_id = str(task.id)
        thread_id = task_id

        try:
            await updater.start_work()

            graph_input = {"messages": [HumanMessage(content=query)]}
            config = RunnableConfig(configurable={"thread_id": thread_id})

            # 스트리밍 실행을 asyncio.Task로 감싸서 강제 취소도 가능하게 함
            async def _stream():
                accumulated_text = ""
                last_result = None
                # 모든 청크에서 커스텀 추출 결과를 누적 (final_report 등)
                best_extracted = ""

                async for chunk in self.graph.astream(graph_input, config=config):
                    # --- 취소 체크 (협력적: 노드 사이에서 확인) ---
                    if task_id in self._cancelled_task_ids:
                        logger.info(f"LGAgentExecutor 폴링 취소 감지: {task_id}")
                        return None, "cancelled", ""

                    last_result = chunk

                    # 커스텀 추출기로 매 청크 시도 (가장 긴 결과를 보존)
                    try:
                        extracted = self._extract_result(chunk)
                        if extracted and len(extracted) > len(best_extracted):
                            best_extracted = extracted
                    except Exception:
                        pass

                    # AI 텍스트 증분 추출 및 스트리밍 전송
                    partial = self._default_extract_text(chunk)
                    if partial and partial != accumulated_text:
                        # 증분(delta)만 전송
                        if partial.startswith(accumulated_text):
                            delta = partial[len(accumulated_text):]
                        else:
                            delta = partial
                        accumulated_text = partial

                        if delta:
                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message(delta, task.context_id, task.id),
                            )

                return last_result, accumulated_text, best_extracted

            run_task = asyncio.create_task(_stream())
            self._running_tasks[task_id] = run_task

            try:
                result = await asyncio.wait_for(
                    run_task, timeout=self._execution_timeout
                )
            except asyncio.TimeoutError:
                run_task.cancel()
                self._running_tasks.pop(task_id, None)
                self._cancelled_task_ids.discard(task_id)
                logger.warning(
                    f"LGAgentExecutor 실행 타임아웃: {task_id} "
                    f"({self._execution_timeout}초)"
                )
                timeout_msg = new_agent_text_message(
                    f"실행 시간 초과 ({self._execution_timeout}초)",
                    task.context_id, task.id,
                )
                await updater.failed(message=timeout_msg)
                return
            finally:
                self._running_tasks.pop(task_id, None)
                self._cancelled_task_ids.discard(task_id)

            if result is None:
                # asyncio.Task.cancel()에 의한 취소 (CancelledError는 위에서 catch)
                return

            last_result, accumulated_text, best_extracted = result

            if accumulated_text == "cancelled":
                # 폴링 기반 취소
                message = new_agent_text_message(
                    "작업이 취소되었습니다.", task.context_id, task.id
                )
                await updater.cancel(message)
                return

            # 최종 결과: 커스텀 추출 > 마지막 청크 추출 > 스트림 누적 텍스트
            final_text = best_extracted
            if not final_text:
                final_text = self._extract_result(last_result or {})
            if not final_text:
                final_text = accumulated_text or "결과를 생성하지 못했습니다."

            await updater.add_artifact([Part(root=TextPart(text=final_text))])
            await updater.complete()

        except asyncio.CancelledError:
            # asyncio.Task.cancel()에 의한 강제 취소
            logger.info(f"LGAgentExecutor 강제 취소됨: {task_id}")
            message = new_agent_text_message(
                "작업이 취소되었습니다.", task.context_id, task.id
            )
            await updater.cancel(message)

        except Exception as e:
            logger.error(f"LGAgentExecutor 실행 오류: {e}")
            error_message = new_agent_text_message(
                f"실행 중 오류: {e}", task.context_id, task.id
            )
            await updater.failed(message=error_message)
            raise ServerError(error=InternalError()) from e

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """하이브리드 취소: 폴링 플래그 + asyncio.Task.cancel() 동시 적용."""
        task = context.current_task
        if not task:
            raise ServerError(error=TaskNotFoundError())

        task_id = str(task.id)

        # 1) 폴링 플래그 세팅 (노드 사이에서 감지)
        self._cancelled_task_ids.add(task_id)

        # 2) asyncio.Task 강제 취소 (노드 실행 중에도 즉시 중단)
        running = self._running_tasks.get(task_id)
        if running and not running.done():
            running.cancel()
            logger.info(f"LGAgentExecutor 취소 요청 (하이브리드): {task_id}")
        else:
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            message = new_agent_text_message(
                "작업이 취소되었습니다.", task.context_id, task.id
            )
            await updater.cancel(message)
