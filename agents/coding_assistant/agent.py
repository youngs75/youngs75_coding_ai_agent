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
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.context_manager import (
    ContextManager,
    invoke_with_max_tokens_recovery,
)
from youngs75_a2a.core.mcp_loader import MCPToolLoader
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.memory.store import MemoryStore
from youngs75_a2a.core.tool_call_utils import tc_args, tc_id, tc_name
from youngs75_a2a.core.tool_permissions import PermissionDecision

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
        "RETRIEVE_MEMORY": "retrieve_memory",
        "EXECUTE": "execute_code",
        "EXECUTE_TOOLS": "execute_tools",
        "VERIFY": "verify_result",
    }

    def __init__(
        self,
        *,
        config: CodingConfig | None = None,
        model: BaseChatModel | None = None,
        memory_store: MemoryStore | None = None,
        **kwargs: Any,
    ) -> None:
        self._coding_config = config or CodingConfig()
        self._mcp_loader = MCPToolLoader(self._coding_config.mcp_servers)
        self._tools: list[Any] = []
        self._memory_store = memory_store

        self._explicit_model = model
        self._gen_model: BaseChatModel | None = None
        self._verify_model: BaseChatModel | None = None
        self._parse_model: BaseChatModel | None = None
        self._context_manager = ContextManager(
            max_tokens=getattr(self._coding_config, "max_context_tokens", 128000),
            compact_threshold=getattr(self._coding_config, "compact_threshold", 0.8),
        )

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
            self._parse_model = self._explicit_model or self._coding_config.get_model(
                "default"
            )
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
        # 컨텍스트 컴팩션 + max_tokens 복구 적용
        if self._context_manager.should_compact(messages):
            messages = await self._context_manager.compact(
                messages, self._get_parse_model()
            )
        response = await invoke_with_max_tokens_recovery(
            self._get_parse_model(),
            messages,
            self._context_manager,
        )

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
            "tool_call_count": 0,
        }

    async def _retrieve_memory(self, state: CodingState) -> dict[str, Any]:
        """Episodic/Procedural Memory를 검색하여 상태에 주입한다.

        parse 후 execute 전에 실행되어 관련 과거 이력과
        학습된 스킬 패턴을 자동으로 컨텍스트에 포함시킨다.
        """
        if not self._memory_store:
            return {}

        parse_result = state.get("parse_result", {})
        description = parse_result.get("description", "")
        language = parse_result.get("language", "python")
        task_type = parse_result.get("task_type", "generate")

        # 검색 쿼리: 작업 설명 기반
        query = description or f"{task_type} {language}"

        result: dict[str, Any] = {}

        # ── Procedural Memory 검색: 학습된 스킬 패턴 ──
        try:
            skills = self._memory_store.retrieve_skills(
                query=query,
                tags=[language, task_type],
                limit=5,
            )
            if skills:
                result["procedural_skills"] = [
                    f"[{s.tags}] {s.metadata.get('description', s.content[:100])}"
                    for s in skills
                ]
        except Exception:
            pass  # 검색 실패 시 무시

        # ── Episodic Memory 검색: 이전 실행 이력 ──
        try:
            episodes = self._memory_store.search(
                query=query,
                memory_type=MemoryType.EPISODIC,
                limit=3,
            )
            if episodes:
                result["episodic_log"] = [e.content for e in episodes]
        except Exception:
            pass  # 검색 실패 시 무시

        return result

    async def _execute_code(self, state: CodingState) -> dict[str, Any]:
        """ReAct 루프: LLM이 MCP 도구로 컨텍스트를 수집하고 코드를 생성한다."""
        parse_result = state.get("parse_result", {})
        verify_result = state.get("verify_result")
        language = parse_result.get("language", "python")

        # 도구 설명 생성
        tool_descriptions = (
            self._mcp_loader.get_tool_descriptions()
            if self._tools
            else "사용 가능한 도구 없음"
        )

        # 시스템 프롬프트 구성 (프로젝트 컨텍스트 포함)
        system_prompt = self._build_system_prompt(
            EXECUTE_SYSTEM_PROMPT.format(
                language=language,
                tool_descriptions=tool_descriptions,
            )
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

        # Procedural Memory 주입 — 학습된 스킬 패턴 참조
        procedural_skills = state.get("procedural_skills", [])
        if procedural_skills:
            system_prompt += "\n\n## 학습된 코드 패턴 (Procedural Memory)\n"
            system_prompt += "\n".join(f"- {skill}" for skill in procedural_skills)

        # 컨텍스트 메시지 구성
        context_parts = [
            f"작업 유형: {parse_result.get('task_type', 'generate')}",
            f"설명: {parse_result.get('description', '')}",
        ]
        if parse_result.get("requirements"):
            context_parts.append(f"요구사항: {', '.join(parse_result['requirements'])}")
        if parse_result.get("target_files"):
            context_parts.append(
                f"대상 파일: {', '.join(parse_result['target_files'])}"
            )

        # 검증 실패 재시도 시 피드백 반영
        if verify_result and not verify_result.get("passed"):
            issues = verify_result.get("issues", [])
            issue_strs = [
                i if isinstance(i, str) else json.dumps(i, ensure_ascii=False)
                for i in issues
            ]
            context_parts.append(
                "\n이전 검증에서 발견된 문제:\n- " + "\n- ".join(issue_strs)
            )
            context_parts.append("위 문제를 수정하여 다시 코드를 작성하세요.")

        context_msg = HumanMessage(content="\n".join(context_parts))

        # LLM에 도구 바인딩
        llm_with_tools = (
            self._get_gen_model().bind_tools(self._tools)
            if self._tools
            else self._get_gen_model()
        )

        messages = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            context_msg,
        ]

        # 컨텍스트 컴팩션 + max_tokens 복구 적용
        if self._context_manager.should_compact(messages):
            messages = await self._context_manager.compact(
                messages, self._get_gen_model()
            )
        response = await invoke_with_max_tokens_recovery(
            llm_with_tools,
            messages,
            self._context_manager,
        )

        iteration = state.get("iteration", 0)
        log = state.get("execution_log", [])
        log.append(f"[execute] iteration={iteration}, tools_bound={len(self._tools)}")

        return {
            "generated_code": response.content or "",
            "execution_log": log,
            "messages": [response],
        }

    async def _execute_tools(self, state: CodingState) -> dict[str, Any]:
        """LLM이 요청한 도구를 실행하고 결과를 반환한다.

        Phase 10 통합:
        - ToolPermissionManager가 설정되어 있으면 실행 전 권한 검사
        - ParallelToolExecutor가 설정되어 있으면 병렬/순차 분류 실행
        - 두 기능이 없으면 기존 순차 실행 폴백

        Phase 12 강화:
        - max_tool_calls 한도 체크 (초과 시 도구 실행 스킵)
        - project_context 파일 경로 기준 중복 제거
        - tool_call_count 누적
        """
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None) or []

        if not tool_calls:
            return {}

        # ── max_tool_calls 한도 체크 ──
        current_count = state.get("tool_call_count", 0)
        max_calls = self._coding_config.max_tool_calls
        if current_count >= max_calls:
            skip_messages = []
            for call in tool_calls:
                call_id = tc_id(call) or f"call_{tc_name(call)}"
                skip_messages.append(
                    ToolMessage(
                        content=f"도구 호출 한도({max_calls}회) 초과. 수집된 정보로 코드를 생성하세요.",
                        tool_call_id=call_id,
                        name=tc_name(call) or "unknown",
                    )
                )
            return {"messages": skip_messages}

        tools_by_name = _get_tools_by_name(self._tools)
        context_entries = list(state.get("project_context", []))

        # project_context 중복 제거를 위한 기존 경로 추적
        seen_paths: set[str] = set()
        for entry in context_entries:
            if entry.startswith("[") and "]\n" in entry:
                seen_paths.add(entry[1 : entry.index("]\n")])

        # 병렬 실행기가 있으면 ParallelToolExecutor를 통한 실행
        if self.tool_executor:
            # 권한 검사를 포함하는 도구 실행 함수
            async def _checked_tool_executor(name: str, args: dict) -> str:
                """권한 검사 후 도구를 실행하는 래퍼."""
                # 권한 검사
                if self.permission_manager:
                    decision = self.permission_manager.check(name, args)
                    if decision == PermissionDecision.DENY:
                        return f"권한 거부: {name}"
                    # ASK인 경우 로그만 남기고 실행 허용 (CLI 레벨에서 처리)

                if name in tools_by_name:
                    result = await _execute_tool_safely(tools_by_name[name], args)
                    # read_file 결과를 project_context에 축적 (중복 경로 갱신)
                    if name == "read_file":
                        path = args.get("path", "?")
                        new_entry = f"[{path}]\n{result[:2000]}"
                        if path not in seen_paths:
                            context_entries.append(new_entry)
                            seen_paths.add(path)
                        else:
                            for i, e in enumerate(context_entries):
                                if e.startswith(f"[{path}]"):
                                    context_entries[i] = new_entry
                                    break
                    return result
                return f"알 수 없는 도구: {name}"

            tool_messages = await self.tool_executor.execute_batch(
                tool_calls, _checked_tool_executor
            )
        else:
            # 기존 순차 실행 폴백
            tool_messages = []
            for call in tool_calls:
                name = tc_name(call)
                args = tc_args(call)
                call_id = tc_id(call)

                # 권한 검사 (permission_manager가 있는 경우)
                if self.permission_manager and name:
                    decision = self.permission_manager.check(name, args)
                    if decision == PermissionDecision.DENY:
                        tool_messages.append(
                            ToolMessage(
                                content=f"권한 거부: {name}",
                                tool_call_id=call_id or f"call_{name}",
                                name=name,
                            )
                        )
                        continue

                if name and name in tools_by_name:
                    result = await _execute_tool_safely(tools_by_name[name], args)
                    # read_file 결과를 project_context에 축적 (중복 경로 갱신)
                    if name == "read_file":
                        path = args.get("path", "?")
                        new_entry = f"[{path}]\n{result[:2000]}"
                        if path not in seen_paths:
                            context_entries.append(new_entry)
                            seen_paths.add(path)
                        else:
                            for i, e in enumerate(context_entries):
                                if e.startswith(f"[{path}]"):
                                    context_entries[i] = new_entry
                                    break
                else:
                    result = f"알 수 없는 도구: {name}"

                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=call_id or f"call_{name}",
                        name=name or "unknown",
                    )
                )

        return {
            "messages": tool_messages,
            "project_context": context_entries,
            "tool_call_count": current_count + len(tool_calls),
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
        # max_tokens 복구 적용
        response = await invoke_with_max_tokens_recovery(
            self._get_verify_model(),
            messages,
            self._context_manager,
        )

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

        result: dict[str, Any] = {
            "verify_result": verify_result,
            "execution_log": log,
            "iteration": state.get("iteration", 0) + 1,
            "tool_call_count": 0,  # 재시도 시 도구 호출 카운터 리셋
        }

        # Procedural Memory 훅: 검증 통과 시 코드 패턴 자동 누적
        if verify_result.get("passed") and self._memory_store:
            self._accumulate_skill_from_execution(state, result)

        # Episodic Memory 훅: 실행 결과를 에피소딕 메모리에 기록
        if self._memory_store:
            self._record_episodic_memory(state, verify_result)

        return result

    # ── Procedural Memory 훅 ─────────────────────────────────

    def _accumulate_skill_from_execution(
        self, state: CodingState, result: dict[str, Any]
    ) -> None:
        """검증 통과 후 코드 패턴을 Procedural Memory에 누적한다.

        Voyager 패턴: 성공적인 코드 실행 결과에서 패턴을 추출하여
        novelty 필터링 후 저장한다.
        """
        generated_code = state.get("generated_code", "")
        if not generated_code or not generated_code.strip():
            return

        parse_result = state.get("parse_result", {})
        task_type = parse_result.get("task_type", "generate")
        language = parse_result.get("language", "python")
        description = parse_result.get("description", "")

        skill_description = f"[{task_type}] {language}: {description}"
        tags = [task_type, language]

        item = self._memory_store.accumulate_skill(
            code=generated_code,
            description=skill_description,
            tags=tags,
        )
        if item:
            result.setdefault("execution_log", []).append(
                f"[procedural] 스킬 저장됨: {skill_description[:80]}"
            )

    # ── Episodic Memory 훅 ─────────────────────────────────

    def _record_episodic_memory(
        self, state: CodingState, verify_result: dict[str, Any]
    ) -> None:
        """실행 결과를 Episodic Memory에 기록한다.

        성공/실패 여부, 작업 설명, 언어 등을 포함하여
        이후 유사한 작업 수행 시 참고 컨텍스트로 활용한다.
        """
        try:
            parse_result = state.get("parse_result", {})
            task_type = parse_result.get("task_type", "generate")
            language = parse_result.get("language", "python")
            description = parse_result.get("description", "")
            passed = verify_result.get("passed", False)

            status = "성공" if passed else "실패"
            issues = verify_result.get("issues", [])
            issue_summary = ""
            if issues:
                issue_strs = [
                    i if isinstance(i, str) else json.dumps(i, ensure_ascii=False)
                    for i in issues[:3]
                ]
                issue_summary = f" | 이슈: {', '.join(issue_strs)}"

            content = f"[{status}] {task_type}/{language}: {description}{issue_summary}"

            tags = [task_type, language, status]

            item = MemoryItem(
                type=MemoryType.EPISODIC,
                content=content,
                tags=tags,
                metadata={
                    "task_type": task_type,
                    "language": language,
                    "passed": passed,
                },
            )
            self._memory_store.put(item)
        except Exception:
            pass  # 에피소딕 기록 실패 시 무시

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
        graph.add_node(self.get_node_name("RETRIEVE_MEMORY"), self._retrieve_memory)
        graph.add_node(self.get_node_name("EXECUTE"), self._execute_code)
        graph.add_node(self.get_node_name("EXECUTE_TOOLS"), self._execute_tools)
        graph.add_node(self.get_node_name("VERIFY"), self._verify_result)

    def init_edges(self, graph: StateGraph) -> None:
        # parse → retrieve_memory → execute
        graph.set_entry_point(self.get_node_name("PARSE"))
        graph.add_edge(
            self.get_node_name("PARSE"),
            self.get_node_name("RETRIEVE_MEMORY"),
        )
        graph.add_edge(
            self.get_node_name("RETRIEVE_MEMORY"),
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
