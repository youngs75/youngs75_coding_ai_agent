"""이벤트 기반 훅 시스템.

Claude Code의 훅 패턴을 참고하여 구현한 이벤트 기반 훅 매니저.
도구 실행, 노드 진입/완료, LLM 호출, 에러 발생 시점에 핸들러를 등록하여
관측성, 감사, 커스텀 로직을 삽입할 수 있다.

사용 예시:
    from youngs75_a2a.core.hooks import HookManager, HookEvent, HookContext

    manager = HookManager()
    manager.register(HookEvent.PRE_TOOL_CALL, my_handler, priority=10)
    ctx = await manager.emit(HookContext(event=HookEvent.PRE_TOOL_CALL, tool_name="read_file"))
"""

from __future__ import annotations

import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """훅 이벤트 타입."""

    PRE_TOOL_CALL = "pre_tool_call"  # 도구 실행 전
    POST_TOOL_CALL = "post_tool_call"  # 도구 실행 후
    PRE_NODE = "pre_node"  # 그래프 노드 진입 전
    POST_NODE = "post_node"  # 그래프 노드 완료 후
    ON_ERROR = "on_error"  # 에러 발생 시
    PRE_LLM_CALL = "pre_llm_call"  # LLM 호출 전
    POST_LLM_CALL = "post_llm_call"  # LLM 호출 후


@dataclass
class HookContext:
    """훅 핸들러에 전달되는 컨텍스트.

    핸들러는 이 객체를 수정하여 후속 핸들러와 실제 실행에 영향을 줄 수 있다.
    예: pre_tool_call 훅에서 tool_args를 수정하면 변경된 인자로 도구가 실행됨.

    Attributes:
        event: 이벤트 타입
        tool_name: 도구 이름 (도구 관련 이벤트)
        tool_args: 도구 인자 (도구 관련 이벤트)
        tool_result: 도구 실행 결과 (POST_TOOL_CALL)
        node_name: 노드 이름 (노드 관련 이벤트)
        state: 그래프 상태 딕셔너리
        error: 에러 객체 (ON_ERROR)
        metadata: 핸들러 간 데이터 공유용 메타데이터.
            - "cancel": True 설정 시 실행 스킵
    """

    event: HookEvent
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: Any | None = None
    node_name: str | None = None
    state: dict | None = None
    error: Exception | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class _RegisteredHandler:
    """등록된 핸들러 내부 표현."""

    handler_id: str
    event: HookEvent
    handler: Callable
    priority: int  # 낮은 숫자가 먼저 실행


class HookManager:
    """이벤트 기반 훅 매니저. 동기/비동기 핸들러 모두 지원.

    - 핸들러 체이닝: pre 훅에서 context를 수정하면 후속 핸들러와 실제 실행에 반영
    - 우선순위: 낮은 숫자가 먼저 실행
    - 에러 격리: 개별 핸들러 실패가 전체 파이프라인을 중단하지 않음 (로깅 후 계속)
    - 취소 지원: pre 훅에서 context.metadata["cancel"] = True 설정 시 실행 스킵
    """

    def __init__(self) -> None:
        # 이벤트별 핸들러 리스트
        self._handlers: dict[HookEvent, list[_RegisteredHandler]] = {
            event: [] for event in HookEvent
        }
        # handler_id로 빠른 조회
        self._handler_map: dict[str, _RegisteredHandler] = {}

    def register(self, event: HookEvent, handler: Callable, priority: int = 0) -> str:
        """훅 핸들러를 등록한다.

        Args:
            event: 구독할 이벤트 타입
            handler: HookContext를 받아 HookContext를 반환하는 callable.
                동기/비동기 모두 가능. 반환값이 None이면 원래 context를 유지.
            priority: 실행 우선순위. 낮은 숫자가 먼저 실행. 기본값 0.

        Returns:
            handler_id: 등록 해제에 사용할 고유 ID
        """
        handler_id = str(uuid.uuid4())
        registered = _RegisteredHandler(
            handler_id=handler_id,
            event=event,
            handler=handler,
            priority=priority,
        )
        self._handlers[event].append(registered)
        # 우선순위 기준으로 정렬 (안정 정렬 — 동일 우선순위면 등록 순서 유지)
        self._handlers[event].sort(key=lambda h: h.priority)
        self._handler_map[handler_id] = registered

        logger.debug(
            "훅 핸들러 등록: event=%s, handler=%s, priority=%d, id=%s",
            event.value,
            getattr(handler, "__name__", repr(handler)),
            priority,
            handler_id,
        )
        return handler_id

    def unregister(self, handler_id: str) -> bool:
        """핸들러를 제거한다.

        Args:
            handler_id: register()에서 반환된 ID

        Returns:
            True: 성공적으로 제거됨
            False: 해당 ID를 찾을 수 없음
        """
        registered = self._handler_map.pop(handler_id, None)
        if registered is None:
            return False

        handlers = self._handlers[registered.event]
        self._handlers[registered.event] = [
            h for h in handlers if h.handler_id != handler_id
        ]
        logger.debug(
            "훅 핸들러 제거: id=%s, event=%s", handler_id, registered.event.value
        )
        return True

    async def emit(self, context: HookContext) -> HookContext:
        """이벤트를 발행하고 등록된 핸들러를 순차 실행한다.

        핸들러 체이닝: 각 핸들러가 반환한 context가 다음 핸들러에 전달된다.
        에러 격리: 개별 핸들러 실패 시 로깅 후 다음 핸들러를 계속 실행한다.

        Args:
            context: 이벤트 컨텍스트

        Returns:
            핸들러 체이닝을 거친 최종 HookContext
        """
        handlers = self._handlers.get(context.event, [])
        if not handlers:
            return context

        for registered in handlers:
            try:
                result = registered.handler(context)
                # 비동기 핸들러 지원
                if inspect.isawaitable(result):
                    result = await result
                # 핸들러가 context를 반환하면 체이닝, None이면 유지
                if result is not None:
                    context = result
            except Exception:
                logger.exception(
                    "훅 핸들러 실행 실패 (격리됨): event=%s, handler=%s, id=%s",
                    context.event.value,
                    getattr(registered.handler, "__name__", repr(registered.handler)),
                    registered.handler_id,
                )
                # 에러 격리 — 계속 실행

        return context

    def load_from_config(self, config_path: Path) -> None:
        """YAML 또는 JSON 설정 파일에서 훅을 로드한다.

        설정 파일 형식 (.agent/hooks.yaml):
            hooks:
              - event: pre_tool_call
                handler: youngs75_a2a.core.builtin_hooks.logging_hook
                priority: 0
              - event: pre_tool_call
                handler: youngs75_a2a.core.builtin_hooks.timing_hook
                priority: 10

        Args:
            config_path: 설정 파일 경로 (.yaml, .yml, .json)
        """
        if not config_path.is_file():
            logger.debug("훅 설정 파일 없음: %s", config_path)
            return

        suffix = config_path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                data = self._load_yaml(config_path)
            elif suffix == ".json":
                data = self._load_json(config_path)
            else:
                logger.warning("지원하지 않는 훅 설정 파일 형식: %s", config_path)
                return
        except Exception:
            logger.exception("훅 설정 파일 로드 실패: %s", config_path)
            return

        hook_configs = data.get("hooks", [])
        if not isinstance(hook_configs, list):
            logger.warning("훅 설정이 리스트가 아님: %s", config_path)
            return

        for hook_cfg in hook_configs:
            if not isinstance(hook_cfg, dict):
                continue
            event_str = hook_cfg.get("event")
            handler_path = hook_cfg.get("handler")
            priority = hook_cfg.get("priority", 0)

            if not event_str or not handler_path:
                logger.warning("훅 설정 항목에 event 또는 handler 누락: %s", hook_cfg)
                continue

            try:
                event = HookEvent(event_str)
            except ValueError:
                logger.warning("잘못된 훅 이벤트: %s", event_str)
                continue

            handler = self._import_handler(handler_path)
            if handler is not None:
                self.register(event, handler, priority=int(priority))

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """YAML 파일을 로드한다."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML 미설치 — YAML 훅 설정 로드 불가: %s", path)
            return {}
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _load_json(path: Path) -> dict:
        """JSON 파일을 로드한다."""
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _import_handler(dotted_path: str) -> Callable | None:
        """점 표기법 경로에서 핸들러 함수를 임포트한다.

        예: "youngs75_a2a.core.builtin_hooks.logging_hook"
        """
        parts = dotted_path.rsplit(".", 1)
        if len(parts) != 2:
            logger.warning("잘못된 핸들러 경로: %s", dotted_path)
            return None

        module_path, func_name = parts
        try:
            import importlib

            module = importlib.import_module(module_path)
            handler = getattr(module, func_name, None)
            if handler is None:
                logger.warning(
                    "핸들러 함수를 찾을 수 없음: %s in %s", func_name, module_path
                )
                return None
            return handler
        except ImportError:
            logger.warning("핸들러 모듈 임포트 실패: %s", module_path)
            return None

    def get_handler_count(self, event: HookEvent | None = None) -> int:
        """등록된 핸들러 수를 반환한다.

        Args:
            event: 특정 이벤트의 핸들러 수. None이면 전체 합계.
        """
        if event is not None:
            return len(self._handlers[event])
        return sum(len(handlers) for handlers in self._handlers.values())

    def clear(self, event: HookEvent | None = None) -> None:
        """등록된 핸들러를 모두 제거한다.

        Args:
            event: 특정 이벤트만 제거. None이면 전체 제거.
        """
        if event is not None:
            for h in self._handlers[event]:
                self._handler_map.pop(h.handler_id, None)
            self._handlers[event] = []
        else:
            self._handlers = {e: [] for e in HookEvent}
            self._handler_map.clear()
