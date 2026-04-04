"""LangGraph 그래프 에이전트 기본 클래스.

모든 에이전트는 이 클래스를 상속하여 일관된 방식으로
노드/엣지를 정의하고 그래프를 빌드한다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from .config import BaseAgentConfig
from .context_manager import ContextManager
from .hooks import HookContext, HookEvent, HookManager
from .parallel_tool_executor import ParallelToolExecutor
from .tool_permissions import ToolPermissionManager

logger = logging.getLogger(__name__)


class BaseGraphAgent:
    """LangGraph 그래프 에이전트 기본 클래스.

    서브클래스는 init_nodes()와 init_edges()를 구현해야 한다.

    Phase 10 통합:
    - permission_manager: 도구 실행 전 권한 검사
    - tool_executor: 병렬 도구 실행기
    - project_context: 프로젝트 컨텍스트 문자열 (시스템 프롬프트에 주입)
    - context_manager: 컨텍스트 윈도우 관리 + 자동 컴팩션

    사용 패턴:
        # 동기 초기화 (MCP 등 비동기 작업 불필요 시)
        agent = MyAgent(model=llm)

        # 비동기 초기화 (MCP 도구 로딩 등 필요 시)
        agent = await MyAgent.create(model=llm)
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        *,
        config: BaseAgentConfig | None = None,
        model: BaseChatModel | None = None,
        state_schema: type | None = None,
        config_schema: type | None = None,
        input_state: type | None = None,
        output_state: type | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        store: BaseStore | None = None,
        max_retry_attempts: int = 2,
        agent_name: str | None = None,
        debug: bool = False,
        auto_build: bool = True,
        hook_manager: HookManager | None = None,
    ) -> None:
        self.agent_config = config
        self.model = model
        self.state_schema = state_schema
        self.config_schema = config_schema
        self.input_state = input_state
        self.output_state = output_state
        self.checkpointer = checkpointer
        self.store = store
        self.agent_name = agent_name or self.__class__.__name__
        self.debug = debug
        self.retry_policy = (
            RetryPolicy(max_attempts=max_retry_attempts)
            if max_retry_attempts > 0
            else None
        )
        self.graph: CompiledStateGraph | None = None
        self.hook_manager = hook_manager or HookManager()

        # Phase 10 통합 필드 — 옵셔널 (None이면 비활성)
        self.permission_manager: ToolPermissionManager | None = None
        self.tool_executor: ParallelToolExecutor | None = None
        self.project_context: str | None = None
        self.context_manager: ContextManager | None = None

        if auto_build:
            self.build_graph()

    # ── Phase 10: 시스템 프롬프트 빌더 ────────────────────────

    def _build_system_prompt(self, base_prompt: str) -> str:
        """기본 시스템 프롬프트에 프로젝트 컨텍스트를 추가한다.

        Args:
            base_prompt: 기본 시스템 프롬프트 문자열

        Returns:
            프로젝트 컨텍스트가 추가된 시스템 프롬프트
        """
        if self.project_context:
            return f"{base_prompt}\n\n{self.project_context}"
        return base_prompt

    # ── Phase 10: 컨텍스트 컴팩션 ─────────────────────────────

    async def _check_context_and_compact(
        self,
        messages: list[BaseMessage],
        llm: Any,
    ) -> list[BaseMessage]:
        """컨텍스트 윈도우 확인 후 필요시 자동 컴팩션을 수행한다.

        context_manager가 설정되어 있고 컴팩션이 필요한 경우에만 동작한다.
        설정되지 않은 경우 원본 메시지를 그대로 반환한다.

        Args:
            messages: 현재 메시지 리스트
            llm: 요약에 사용할 LLM 인스턴스

        Returns:
            (필요시 컴팩션된) 메시지 리스트
        """
        if self.context_manager and self.context_manager.should_compact(messages):
            logger.info(
                "[%s] 컨텍스트 컴팩션 수행 중...", self.agent_name
            )
            return await self.context_manager.compact(messages, llm)
        return messages

    @classmethod
    async def create(cls, **kwargs: Any) -> "BaseGraphAgent":
        """비동기 팩토리 메서드.

        MCP 도구 로딩 등 비동기 초기화가 필요한 에이전트에서 사용.
        서브클래스는 async_init()을 오버라이드한다.
        """
        kwargs.setdefault("auto_build", False)
        instance = cls(**kwargs)
        await instance.async_init()
        instance.build_graph()
        return instance

    async def async_init(self) -> None:
        """비동기 초기화 훅. 서브클래스에서 오버라이드."""

    def get_node_name(self, key: str) -> str:
        if key not in self.NODE_NAMES:
            raise ValueError(f"노드 이름 키 '{key}'가 정의되어 있지 않습니다.")
        return self.NODE_NAMES[key]

    def build_graph(self) -> None:
        """StateGraph를 빌드하고 컴파일한다."""
        builder = StateGraph(
            state_schema=self.state_schema,
            config_schema=self.config_schema,
            input=self.input_state,
            output=self.output_state,
        )
        self.init_nodes(builder)
        self.init_edges(builder)
        self.graph = builder.compile(
            checkpointer=self.checkpointer,
            store=self.store,
            name=self.agent_name,
            debug=self.debug,
        )

    async def _wrap_node(
        self,
        node_name: str,
        node_func: Callable[..., Any],
        state: Any,
        config: Any = None,
    ) -> Any:
        """노드 실행을 훅으로 감싸는 헬퍼.

        PRE_NODE, POST_NODE, ON_ERROR 훅을 자동으로 발행한다.
        PRE_NODE 훅에서 cancel=True 설정 시 노드 실행을 스킵한다.

        Args:
            node_name: 노드 이름
            node_func: 실행할 노드 함수
            state: 그래프 상태
            config: LangGraph 실행 설정 (선택)

        Returns:
            노드 함수의 반환값 (또는 취소 시 원래 상태)
        """
        # state를 dict로 변환 (가능한 경우)
        state_dict = dict(state) if isinstance(state, dict) else None

        # PRE_NODE 훅
        pre_ctx = HookContext(
            event=HookEvent.PRE_NODE,
            node_name=node_name,
            state=state_dict,
        )
        pre_ctx = await self.hook_manager.emit(pre_ctx)

        if pre_ctx.metadata.get("cancel"):
            return state

        try:
            result = node_func(state, config) if config else node_func(state)
            # 비동기 노드 함수 지원
            import inspect

            if inspect.isawaitable(result):
                result = await result

            # POST_NODE 훅
            post_ctx = HookContext(
                event=HookEvent.POST_NODE,
                node_name=node_name,
                state=dict(result) if isinstance(result, dict) else state_dict,
                metadata=dict(pre_ctx.metadata),
            )
            await self.hook_manager.emit(post_ctx)

            return result
        except Exception as e:
            # ON_ERROR 훅
            err_ctx = HookContext(
                event=HookEvent.ON_ERROR,
                node_name=node_name,
                state=state_dict,
                error=e,
            )
            await self.hook_manager.emit(err_ctx)
            raise

    def init_nodes(self, graph: StateGraph) -> None:
        raise NotImplementedError

    def init_edges(self, graph: StateGraph) -> None:
        raise NotImplementedError
