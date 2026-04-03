"""Coding Assistant 에이전트 — MCP 도구 연동 Harness.

논문 인사이트 기반 설계:
- P1 (Agent-as-a-Judge): parse → execute(ReAct) → verify
- P2 (RubricRewards): Generator/Verifier 모델 분리
- P5 (GAM): 도구를 통한 JIT 원본 참조

사용 예:
    agent = await CodingAssistantAgent.create(config=CodingConfig())
    result = await agent.graph.ainvoke({
        "messages": [HumanMessage("파이썬으로 피보나치 함수를 작성해줘")],
        "iteration": 0,
        "max_iterations": 3,
    })
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.mcp_loader import MCPToolLoader
from youngs75_a2a.core.tool_call_utils import tc_args, tc_id, tc_name

from .config import CodingConfig
from .prompts import EXECUTE_SYSTEM_PROMPT, PARSE_SYSTEM_PROMPT, VERIFY_SYSTEM_PROMPT
from .schemas import CodingState


async def _execute_tool_safely(tool: Any, args: dict) -> str:
    """도구를 안전하게 실행하고 결과를 문자열로 반환한다."""
    try:
        result = await tool.ainvoke(args)
        return str(result) if result else "실행 완료 (출력 없음)"
    except Exception as e:
        return f"도구 실행 오류: {e}"


def _get_tools_by_name(tools: list[Any]) -> dict[str, Any]:
    """도구 목록을 이름→도구 딕셔너리로 변환한다."""
    return {getattr(t, "name", None): t for t in tools if getattr(t, "name", None)}


class CodingAssistantAgent(BaseGraphAgent):
    """MCP 도구를 활용하는 Coding Assistant Harness.

    parse → execute(ReAct 루프: LLM + MCP 도구) → verify → retry/END
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "PARSE": "parse_request",
        "EXECUTE": "execute_code",
        "EXECUTE_TOOLS": "execute_tools",
        "VERIFY": "verify_result",
    }

    def __init__(
        self,
        *,
        config: CodingConfig | None = None,
        model: BaseChatModel | None = None,
        **kwargs: Any,
    ) -> None:
        self._coding_config = config or CodingConfig()
        self._mcp_loader = MCPToolLoader(self._coding_config.mcp_servers)
        self._tools: list[Any] = []

        self._explicit_model = model
        self._gen_model: BaseChatModel | None = None
        self._verify_model: BaseChatModel | None = None
        self._parse_model: BaseChatModel | None = None

        kwargs.pop("auto_build", None)
        super().__init__(
            config=self._coding_config,
            model=model,
            state_schema=CodingState,
            agent_name="CodingAssistantAgent",
            auto_build=False,
            **kwargs,
        )

    async def async_init(self) -> None:
        """MCP 도구를 비동기로 로드한다."""
        self._tools = await self._mcp_loader.load()

    # ── 모델 lazy init ──────────────────────────────────────

    def _get_parse_model(self) -> BaseChatModel:
        if self._parse_model is None:
            self._parse_model = self._explicit_model or self._coding_config.get_model("default")
        return self._parse_model

    def _get_gen_model(self) -> BaseChatModel:
        if self._gen_model is None:
            self._gen_model = self._coding_config.get_model("generation")
        return self._gen_model

    def _get_verify_model(self) -> BaseChatModel:
        if self._verify_model is None:
            self._verify_model = self._coding_config.get_model("verification")
        return self._verify_model

    # ── 노드 구현 ──────────────────────────────────────────

    async def _parse_request(self, state: CodingState) -> dict[str, Any]:
        """사용자 요청을 분석하여 작업 유형과 요구사항을 추출한다."""
        messages = [
            SystemMessage(content=PARSE_SYSTEM_PROMPT),
            *state["messages"],
        ]
        response = await self._get_parse_model().ainvoke(messages)

        try:
            parse_result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            parse_result = {
                "task_type": "generate",
                "language": "python",
                "description": response.content,
                "target_files": [],
                "requirements": [],
            }

        return {
            "parse_result": parse_result,
            "execution_log": [f"[parse] task_type={parse_result.get('task_type')}"],
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
        }

    async def _execute_code(self, state: CodingState) -> dict[str, Any]:
        """ReAct 루프: LLM이 MCP 도구로 컨텍스트를 수집하고 코드를 생성한다."""
        parse_result = state.get("parse_result", {})
        verify_result = state.get("verify_result")
        language = parse_result.get("language", "python")

        # 도구 설명 생성
        tool_descriptions = self._mcp_loader.get_tool_descriptions() if self._tools else "사용 가능한 도구 없음"

        # 시스템 프롬프트 구성
        system_prompt = EXECUTE_SYSTEM_PROMPT.format(
            language=language,
            tool_descriptions=tool_descriptions,
        )

        # Semantic Memory 주입 — 프로젝트 규칙/컨벤션
        semantic_context = state.get("semantic_context", [])
        if semantic_context:
            system_prompt += "\n\n## 프로젝트 컨텍스트 (Semantic Memory)\n"
            system_prompt += "\n".join(f"- {ctx}" for ctx in semantic_context)

        # Skills 컨텍스트 주입 — 활성 스킬 정보
        skill_context = state.get("skill_context", [])
        if skill_context:
            system_prompt += "\n\n## 사용 가능한 스킬\n"
            system_prompt += "\n".join(f"- {ctx}" for ctx in skill_context)

        # Episodic Memory 주입 — 이전 실행 이력 참조
        episodic_log = state.get("episodic_log", [])
        if episodic_log:
            system_prompt += "\n\n## 이전 실행 이력 (Episodic Memory)\n"
            system_prompt += "\n".join(f"- {entry}" for entry in episodic_log)

        # 컨텍스트 메시지 구성
        context_parts = [
            f"작업 유형: {parse_result.get('task_type', 'generate')}",
            f"설명: {parse_result.get('description', '')}",
        ]
        if parse_result.get("requirements"):
            context_parts.append(f"요구사항: {', '.join(parse_result['requirements'])}")
        if parse_result.get("target_files"):
            context_parts.append(f"대상 파일: {', '.join(parse_result['target_files'])}")

        # 검증 실패 재시도 시 피드백 반영
        if verify_result and not verify_result.get("passed"):
            issues = verify_result.get("issues", [])
            context_parts.append(f"\n이전 검증에서 발견된 문제:\n- " + "\n- ".join(issues))
            context_parts.append("위 문제를 수정하여 다시 코드를 작성하세요.")

        context_msg = HumanMessage(content="\n".join(context_parts))

        # LLM에 도구 바인딩
        llm_with_tools = self._get_gen_model().bind_tools(self._tools) if self._tools else self._get_gen_model()

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            context_msg,
        ]

        response = await llm_with_tools.ainvoke(messages)

        iteration = state.get("iteration", 0)
        log = state.get("execution_log", [])
        log.append(f"[execute] iteration={iteration}, tools_bound={len(self._tools)}")

        return {
            "generated_code": response.content or "",
            "execution_log": log,
            "messages": [response],
        }

    async def _execute_tools(self, state: CodingState) -> dict[str, Any]:
        """LLM이 요청한 도구를 실행하고 결과를 반환한다."""
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None) or []

        if not tool_calls:
            return {}

        tools_by_name = _get_tools_by_name(self._tools)
        tool_messages = []
        context_entries = list(state.get("project_context", []))

        for call in tool_calls:
            name = tc_name(call)
            args = tc_args(call)
            call_id = tc_id(call)

            if name in tools_by_name:
                result = await _execute_tool_safely(tools_by_name[name], args)
                # read_file 결과를 project_context에 축적
                if name == "read_file":
                    context_entries.append(f"[{args.get('path', '?')}]\n{result[:2000]}")
            else:
                result = f"알 수 없는 도구: {name}"

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=call_id, name=name)
            )

        return {
            "messages": tool_messages,
            "project_context": context_entries,
        }

    async def _verify_result(self, state: CodingState) -> dict[str, Any]:
        """생성된 코드를 검증한다 (검증자 특권 정보 포함)."""
        generated_code = state.get("generated_code", "")
        parse_result = state.get("parse_result", {})

        verify_prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=self._coding_config.max_delete_lines,
            allowed_extensions=", ".join(self._coding_config.allowed_extensions),
        )

        verify_context = (
            f"원래 요청: {parse_result.get('description', '')}\n\n"
            f"생성된 코드:\n{generated_code}"
        )
        messages = [
            SystemMessage(content=verify_prompt),
            HumanMessage(content=verify_context),
        ]
        response = await self._get_verify_model().ainvoke(messages)

        try:
            verify_result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            verify_result = {
                "passed": True,
                "issues": [],
                "suggestions": [],
            }

        log = state.get("execution_log", [])
        log.append(f"[verify] passed={verify_result.get('passed')}")

        return {
            "verify_result": verify_result,
            "execution_log": log,
            "iteration": state.get("iteration", 0) + 1,
        }

    # ── 라우팅 ──────────────────────────────────────────────

    def _should_use_tools(self, state: CodingState) -> str:
        """execute 후 도구 호출이 있으면 tools 노드로, 없으면 verify로."""
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None) or []

        if tool_calls:
            return self.get_node_name("EXECUTE_TOOLS")
        return self.get_node_name("VERIFY")

    def _should_retry(self, state: CodingState) -> str:
        """검증 실패 시 재시도 여부를 판단한다."""
        verify_result = state.get("verify_result", {})
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if verify_result.get("passed", True):
            return END
        if iteration >= max_iterations:
            return END
        return self.get_node_name("EXECUTE")

    # ── 그래프 구성 ─────────────────────────────────────────

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("PARSE"), self._parse_request)
        graph.add_node(self.get_node_name("EXECUTE"), self._execute_code)
        graph.add_node(self.get_node_name("EXECUTE_TOOLS"), self._execute_tools)
        graph.add_node(self.get_node_name("VERIFY"), self._verify_result)

    def init_edges(self, graph: StateGraph) -> None:
        # parse → execute
        graph.set_entry_point(self.get_node_name("PARSE"))
        graph.add_edge(
            self.get_node_name("PARSE"),
            self.get_node_name("EXECUTE"),
        )
        # execute → (tools or verify)
        graph.add_conditional_edges(
            self.get_node_name("EXECUTE"),
            self._should_use_tools,
        )
        # tools → execute (ReAct 루프)
        graph.add_edge(
            self.get_node_name("EXECUTE_TOOLS"),
            self.get_node_name("EXECUTE"),
        )
        # verify → (retry or END)
        graph.add_conditional_edges(
            self.get_node_name("VERIFY"),
            self._should_retry,
        )
