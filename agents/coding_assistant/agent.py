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

import asyncio
import json
import logging
import os
import re
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.context_manager import (
    ContextManager,
    invoke_with_max_tokens_recovery,
)
from youngs75_a2a.core.mcp_loader import MCPToolLoader
from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.memory.store import MemoryStore
from youngs75_a2a.core.skills.registry import SkillRegistry
from youngs75_a2a.core.stall_detector import StallAction, StallDetector
from youngs75_a2a.core.turn_budget import BudgetVerdict, TurnBudgetTracker
from youngs75_a2a.core.tool_call_utils import (
    sanitize_messages_for_llm,
    tc_args,
    tc_id,
    tc_name,
)
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


# ── 코드 블록 파싱 ──

_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)

_FILEPATH_COMMENT_PATTERNS = [
    # # filepath: path/to/file.py  또는  // filepath: path/to/file.py
    re.compile(r"^(?:#|//)\s*(?:filepath:\s*)(\S*\.\S+)\s*$"),
    # <!-- filepath: path/to/file.html -->
    re.compile(r"^<!--\s*(?:filepath:\s*)(\S*\.\S+)\s*-->$"),
    # /* filepath: path/to/file.css */
    re.compile(r"^/\*\s*(?:filepath:\s*)(\S*\.\S+)\s*\*/$"),
    # 경로만 있는 주석: // app.py  또는  # app.py
    re.compile(r"^(?:#|//)\s*(\S*\.\S+)\s*$"),
    # <!-- templates/index.html -->
    re.compile(r"^<!--\s*(\S*\.\S+)\s*-->$"),
]

# 코드 블록 바로 위에 있는 filepath 마크다운 헤딩 패턴
# LLM이 `# filepath: app.py` 를 코드 블록 밖에 제목으로 생성하는 경우 대응
_HEADING_FILEPATH_RE = re.compile(r"#+\s*(?:filepath:\s*)?(\S*\.\S+)\s*$", re.MULTILINE)
# 굵은 텍스트 형태: **filepath: app.py**
_BOLD_FILEPATH_RE = re.compile(
    r"\*{1,2}(?:filepath:\s*)?([^\s*]+\.[^\s*]+)\*{1,2}\s*$", re.MULTILINE
)

_FORBIDDEN_PATHS = (".claude/", ".git/", "__pycache__/", "node_modules/")


def _find_filepath_before_fence(text: str, fence_start: int) -> str:
    """코드 블록 바로 위 텍스트에서 filepath를 찾는다."""
    # 코드 블록 앞 200자에서 검색
    preceding = text[max(0, fence_start - 200) : fence_start]
    # 빈 줄 기준으로 마지막 단락만 추출
    lines = preceding.rstrip().rsplit("\n", 3)

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # 마크다운 제목: # filepath: app.py  또는  ## app.py
        m = _HEADING_FILEPATH_RE.match(line)
        if m:
            return m.group(1).strip()
        # 굵은 텍스트: **filepath: app.py**
        m = _BOLD_FILEPATH_RE.match(line)
        if m:
            return m.group(1).strip()
        # 일반 텍스트 filepath: app.py
        for pattern in _FILEPATH_COMMENT_PATTERNS:
            m = pattern.match(line)
            if m:
                return m.group(1).strip()
        # 첫 비빈 줄만 확인
        break
    return ""


def _extract_code_blocks(text: str) -> list[dict[str, str]]:
    """마크다운에서 파일 경로가 포함된 코드 블록을 추출한다.

    filepath를 다음 위치에서 순서대로 탐색:
    1. 코드 블록 내부 첫 줄 (# filepath: app.py)
    2. 코드 블록 바로 위 텍스트 (마크다운 제목, 주석 등)
    """
    blocks: list[dict[str, str]] = []
    for match in _FENCE_RE.finditer(text):
        lang = match.group(1) or ""
        code = match.group(2)

        filepath = ""
        # 전략 1: 코드 블록 내부 첫 줄에서 filepath 추출
        if code:
            first_line = code.split("\n", 1)[0].strip()
            for pattern in _FILEPATH_COMMENT_PATTERNS:
                m = pattern.match(first_line)
                if m:
                    filepath = m.group(1).strip()
                    # 파일 경로 주석 줄 제거
                    code = code.split("\n", 1)[1] if "\n" in code else ""
                    break

        # 전략 2: 코드 블록 바로 위 텍스트에서 filepath 추출
        if not filepath:
            filepath = _find_filepath_before_fence(text, match.start())

        if filepath:
            blocks.append(
                {
                    "filepath": filepath,
                    "language": lang,
                    "code": code.rstrip("\n"),
                }
            )

    return blocks


class CodingAssistantAgent(BaseGraphAgent):
    """MCP 도구를 활용하는 Coding Assistant Harness.

    parse → execute(ReAct 루프: LLM + MCP 도구) → verify → apply_code → END
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "PARSE": "parse_request",
        "RETRIEVE_MEMORY": "retrieve_memory",
        "EXECUTE": "execute_code",
        "EXECUTE_TOOLS": "execute_tools",
        "GENERATE_FINAL": "generate_final",
        "VERIFY": "verify_result",
        "RUN_TESTS": "run_tests",
        "APPLY_CODE": "apply_code",
        "INJECT_TEST_FAILURE": "inject_test_failure",
    }

    def __init__(
        self,
        *,
        config: CodingConfig | None = None,
        model: BaseChatModel | None = None,
        memory_store: MemoryStore | None = None,
        skill_registry: SkillRegistry | None = None,
        **kwargs: Any,
    ) -> None:
        self._coding_config = config or CodingConfig()
        self._mcp_loader = MCPToolLoader(self._coding_config.mcp_servers)
        self._tools: list[Any] = []
        self._memory_store = memory_store
        self._skill_registry = skill_registry

        self._explicit_model = model
        self._gen_model: BaseChatModel | None = None
        self._tool_planning_model: BaseChatModel | None = None
        self._verify_model: BaseChatModel | None = None
        self._parse_model: BaseChatModel | None = None
        self._context_manager = ContextManager(
            max_tokens=getattr(self._coding_config, "max_context_tokens", 128000),
            compact_threshold=getattr(self._coding_config, "compact_threshold", 0.8),
        )

        # 다층 안전장치 (Claude Code OS 패턴)
        self._stall_detector = StallDetector(
            warn_threshold=self._coding_config.stall_warn_threshold,
            exit_threshold=self._coding_config.stall_exit_threshold,
        )
        self._turn_budget = TurnBudgetTracker(
            max_llm_calls=self._coding_config.max_llm_calls_per_turn,
            diminishing_streak_limit=self._coding_config.diminishing_streak_limit,
            min_delta_tokens=self._coding_config.min_delta_tokens,
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

    def _get_tool_planning_model(self) -> BaseChatModel:
        if self._tool_planning_model is None:
            self._tool_planning_model = self._coding_config.get_model("tool_planning")
        return self._tool_planning_model

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
        # 턴 시작 시 안전장치 초기화
        self._stall_detector.reset()
        self._turn_budget.reset()

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

        result: dict[str, Any] = {
            "parse_result": parse_result,
            "execution_log": [f"[parse] task_type={parse_result.get('task_type')}"],
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
            "tool_call_count": 0,
        }

        # task_type 기반 스킬 자동 활성화
        if self._skill_registry:
            task_type = parse_result.get("task_type", "generate")
            framework_hint = parse_result.get("framework", "")
            activated = self._skill_registry.auto_activate_for_task(
                task_type, framework_hint=framework_hint
            )
            if activated:
                # L2 본문이 로드된 스킬의 컨텍스트 주입
                skill_bodies = self._skill_registry.get_active_skill_bodies()
                if skill_bodies:
                    result["skill_context"] = skill_bodies
                result["execution_log"].append(
                    f"[skills] 자동 활성화: {', '.join(activated)}"
                )

        return result

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

    def _build_execute_system_prompt(self, state: CodingState) -> str:
        """execute/generate 노드 공통 시스템 프롬프트를 구성한다."""
        parse_result = state.get("parse_result", {})
        language = parse_result.get("language", "python")

        tool_descriptions = (
            self._mcp_loader.get_tool_descriptions()
            if self._tools
            else "사용 가능한 도구 없음"
        )

        system_prompt = self._build_system_prompt(
            EXECUTE_SYSTEM_PROMPT.format(
                language=language,
                tool_descriptions=tool_descriptions,
            )
        )

        # Semantic Memory 주입
        semantic_context = state.get("semantic_context", [])
        if semantic_context:
            system_prompt += "\n\n## 프로젝트 컨텍스트 (Semantic Memory)\n"
            system_prompt += "\n".join(f"- {ctx}" for ctx in semantic_context)

        # Skills 컨텍스트 주입 — 활성 스킬 L2 본문 포함
        skill_context = state.get("skill_context", [])
        if skill_context:
            system_prompt += "\n\n## 활성 스킬\n"
            system_prompt += "\n".join(f"- {ctx}" for ctx in skill_context)

        # Episodic Memory 주입
        episodic_log = state.get("episodic_log", [])
        if episodic_log:
            system_prompt += "\n\n## 이전 실행 이력 (Episodic Memory)\n"
            system_prompt += "\n".join(f"- {entry}" for entry in episodic_log)

        # Procedural Memory 주입
        procedural_skills = state.get("procedural_skills", [])
        if procedural_skills:
            system_prompt += "\n\n## 학습된 코드 패턴 (Procedural Memory)\n"
            system_prompt += "\n".join(f"- {skill}" for skill in procedural_skills)

        return system_prompt

    def _build_context_message(self, state: CodingState) -> HumanMessage:
        """execute/generate 노드 공통 컨텍스트 메시지를 구성한다."""
        parse_result = state.get("parse_result", {})
        verify_result = state.get("verify_result")

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

        return HumanMessage(content="\n".join(context_parts))

    async def _execute_code(self, state: CodingState) -> dict[str, Any]:
        """ReAct 루프 1단계: 도구 호출을 판단하고 컨텍스트를 수집한다.

        generate 작업은 LLM 호출 없이 패스스루 — generate_final(STRONG)이 직접 생성.
        fix/refactor/analyze만 FAST 모델로 도구 호출을 판단한다.
        """
        parse_result = state.get("parse_result", {})
        task_type = parse_result.get("task_type", "generate")
        iteration = state.get("iteration", 0)
        log = state.get("execution_log", [])

        # generate/scaffold 작업 → LLM 호출 없이 패스스루 (이중 생성 방지)
        # generate_final(STRONG)이 직접 코드를 생성한다.
        # scaffold도 FAST 모델이 도구를 호출하지 않고 저품질 코드를 생성하므로 STRONG 직행.
        if task_type in ("generate", "scaffold") and iteration == 0:
            log.append(f"[execute] {task_type} 태스크 — FAST 스킵, STRONG으로 직행")
            return {"execution_log": log}

        system_prompt = self._build_execute_system_prompt(state)
        context_msg = self._build_context_message(state)

        # fix/refactor/analyze만 도구가 필요 (기존 파일 읽기/수정)
        needs_tools = task_type in ("fix", "refactor", "analyze")

        llm_with_tools = self._get_tool_planning_model()
        if needs_tools and self._tools:
            llm_with_tools = llm_with_tools.bind_tools(self._tools)

        messages = [
            SystemMessage(content=system_prompt),
            *sanitize_messages_for_llm(state["messages"]),
            context_msg,
        ]

        if self._context_manager.should_compact(messages):
            messages = await self._context_manager.compact(
                messages, self._get_tool_planning_model()
            )
        try:
            response = await invoke_with_max_tokens_recovery(
                llm_with_tools,
                messages,
                self._context_manager,
            )
        except Exception as e:
            # DashScope 400 (invalid JSON arguments) 등 LLM 호출 실패 방어
            error_msg = str(e)
            logging.getLogger(__name__).warning(
                "[execute] LLM 호출 실패 — GENERATE_FINAL로 전환: %s", error_msg[:200]
            )
            log.append(f"[execute] LLM 오류 — GENERATE_FINAL로 전환: {error_msg[:100]}")
            return {
                "generated_code": "",
                "execution_log": log,
                "exit_reason": "llm_error",
            }

        # 토큰 예산 추적 (감소수익 감지)
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            output_tokens = getattr(response.usage_metadata, "output_tokens", 0)
        elif hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)
        # 폴백: 콘텐츠 길이 기반 추정
        if output_tokens == 0 and response.content:
            output_tokens = len(response.content) // 4

        budget_verdict = self._turn_budget.record_llm_call(output_tokens)

        log.append(
            f"[execute] iteration={iteration}, tools_bound={needs_tools}, model=FAST"
        )

        result: dict[str, Any] = {
            "generated_code": response.content or "",
            "execution_log": log,
            "messages": [response],
        }

        if budget_verdict == BudgetVerdict.STOP:
            result["exit_reason"] = "budget_exceeded"
            log.append(f"[budget] {self._turn_budget.get_summary()}")

        return result

    async def _generate_final(self, state: CodingState) -> dict[str, Any]:
        """2단계: STRONG 모델이 수집된 컨텍스트를 바탕으로 최종 코드를 생성한다.

        도구 호출 없이 코드 생성에만 집중한다. ReAct 루프에서 수집된
        project_context와 도구 결과가 메시지에 포함된 상태에서 호출된다.
        """
        system_prompt = self._build_execute_system_prompt(state)

        # 프로젝트 컨텍스트 축적분 주입
        project_context = state.get("project_context", [])
        if project_context:
            system_prompt += "\n\n## 수집된 프로젝트 파일 컨텍스트\n"
            system_prompt += "\n".join(project_context[:10])  # 최대 10개 파일

        context_msg = self._build_context_message(state)

        # STRONG 모델 — 도구 없이 최종 코드 생성
        gen_model = self._get_gen_model()

        messages = [
            SystemMessage(content=system_prompt),
            *sanitize_messages_for_llm(state["messages"]),
            context_msg,
        ]

        if self._context_manager.should_compact(messages):
            messages = await self._context_manager.compact(messages, gen_model)

        try:
            response = await invoke_with_max_tokens_recovery(
                gen_model,
                messages,
                self._context_manager,
            )
        except Exception as e:
            error_msg = str(e)
            logging.getLogger(__name__).warning(
                "[generate_final] LLM 호출 실패: %s", error_msg[:200]
            )
            log = state.get("execution_log", [])
            log.append(f"[generate_final] LLM 오류: {error_msg[:100]}")
            return {
                "generated_code": state.get("generated_code", ""),
                "execution_log": log,
            }

        log = state.get("execution_log", [])
        log.append("[generate_final] model=STRONG, 최종 코드 생성 완료")

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
            return {"messages": skip_messages, "exit_reason": "turn_limit"}

        # ── 반복 도구 호출 감지 (StallDetector) ──
        for call in tool_calls:
            action = self._stall_detector.record_and_check(tc_name(call), tc_args(call))
            if action == StallAction.FORCE_EXIT:
                summary = self._stall_detector.get_stall_summary()
                stall_messages = []
                for c in tool_calls:
                    c_id = tc_id(c) or f"call_{tc_name(c)}"
                    stall_messages.append(
                        ToolMessage(
                            content=f"[루프 감지] {summary}",
                            tool_call_id=c_id,
                            name=tc_name(c) or "unknown",
                        )
                    )
                return {"messages": stall_messages, "exit_reason": "stall_detected"}

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

    # 검증 타임아웃 (초) — 이 시간 초과 시 검증 스킵하고 진행
    _VERIFY_TIMEOUT_S = 30
    # 검증에 보낼 코드 최대 길이 (토큰 폭발 방지)
    _VERIFY_MAX_CODE_CHARS = 4000

    async def _verify_result(self, state: CodingState) -> dict[str, Any]:
        """생성된 코드를 검증한다.

        안전장치:
        - simple 태스크는 검증 스킵 (불필요한 LLM 호출 방지)
        - 긴 코드는 truncation하여 검증 (토큰 폭발 방지)
        - 타임아웃 보호 (30초 초과 시 graceful skip)
        - LLM 호출 실패 시 passed=True로 폴백 (파이프라인 중단 방지)
        """
        generated_code = state.get("generated_code", "")
        parse_result = state.get("parse_result", {})
        log = state.get("execution_log", [])

        # simple 태스크는 검증 스킵
        task_type = parse_result.get("task_type", "generate")
        if task_type == "generate" and len(generated_code) < 500:
            log.append("[verify] simple 태스크 — 검증 스킵")
            return {
                "verify_result": {"passed": True, "issues": [], "suggestions": []},
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
                "tool_call_count": 0,
            }

        # 코드 truncation — 검증에 전문 대신 요약만 전달
        code_for_verify = generated_code
        if len(code_for_verify) > self._VERIFY_MAX_CODE_CHARS:
            code_for_verify = (
                code_for_verify[: self._VERIFY_MAX_CODE_CHARS]
                + f"\n\n... (총 {len(generated_code)}자 중 {self._VERIFY_MAX_CODE_CHARS}자만 검증)"
            )

        verify_prompt = VERIFY_SYSTEM_PROMPT.format(
            max_delete_lines=self._coding_config.max_delete_lines,
            allowed_extensions=", ".join(self._coding_config.allowed_extensions),
        )
        verify_context = (
            f"원래 요청: {parse_result.get('description', '')}\n\n"
            f"생성된 코드:\n{code_for_verify}"
        )
        messages = [
            SystemMessage(content=verify_prompt),
            HumanMessage(content=verify_context),
        ]

        # 타임아웃 + 에러 보호
        verify_result: dict[str, Any]
        try:
            async with asyncio.timeout(self._VERIFY_TIMEOUT_S):
                response = await invoke_with_max_tokens_recovery(
                    self._get_verify_model(),
                    messages,
                    self._context_manager,
                )
            verify_result = json.loads(response.content)
        except TimeoutError:
            logging.getLogger(__name__).warning(
                "verify_result 타임아웃 (%ds) — 검증 스킵", self._VERIFY_TIMEOUT_S
            )
            log.append(f"[verify] 타임아웃 ({self._VERIFY_TIMEOUT_S}s) — 스킵")
            verify_result = {
                "passed": True,
                "issues": [],
                "suggestions": ["검증 타임아웃으로 스킵됨"],
            }
        except (json.JSONDecodeError, TypeError):
            verify_result = {"passed": True, "issues": [], "suggestions": []}
        except Exception as e:
            logging.getLogger(__name__).warning("verify_result 실패: %s", e)
            log.append(f"[verify] 오류 — 스킵: {e}")
            verify_result = {
                "passed": True,
                "issues": [],
                "suggestions": [f"검증 오류: {e}"],
            }

        log.append(f"[verify] passed={verify_result.get('passed')}")

        result: dict[str, Any] = {
            "verify_result": verify_result,
            "execution_log": log,
            "iteration": state.get("iteration", 0) + 1,
            "tool_call_count": 0,
        }

        # Procedural Memory 훅: 검증 통과 시 코드 패턴 자동 누적
        if verify_result.get("passed") and self._memory_store:
            self._accumulate_skill_from_execution(state, result)

        # Episodic Memory 훅
        if self._memory_store:
            self._record_episodic_memory(state, verify_result)

        return result

    # ── 실행 기반 검증 ─────────────────────────────────────

    _RUN_TESTS_TIMEOUT_S = 60
    _INSTALL_TIMEOUT_S = 120
    _VENV_TIMEOUT_S = 60

    # ── 런타임 감지 ──

    # 언어별 런타임 확인 명령어
    _RUNTIME_CHECKS: dict[str, list[tuple[str, str]]] = {
        "python": [("python3", "--version"), ("python", "--version")],
        "node": [("node", "--version")],
        "java": [("java", "--version"), ("javac", "--version")],
        "go": [("go", "version")],
        "rust": [("cargo", "--version"), ("rustc", "--version")],
    }

    @staticmethod
    async def _check_command_exists(cmd: str) -> bool:
        """명령어가 시스템에 설치되어 있는지 확인한다."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "which", cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    async def _detect_runtimes(self, languages: list[str], log: list[str]) -> dict[str, bool]:
        """필요한 런타임이 설치되어 있는지 확인한다."""
        results: dict[str, bool] = {}
        for lang in languages:
            checks = self._RUNTIME_CHECKS.get(lang, [])
            found = False
            for cmd, arg in checks:
                if await self._check_command_exists(cmd):
                    found = True
                    break
            results[lang] = found
            status = "✓" if found else "✗ 미설치"
            log.append(f"[runtime] {lang}: {status}")
        return results

    # ── 프로젝트 환경 설정 (venv 생성 + 의존성 설치) ──

    async def _get_venv_python(self, workspace: str) -> str | None:
        """workspace의 venv python 경로를 반환한다. venv가 없으면 None."""
        venv_python = os.path.join(workspace, ".venv", "bin", "python")
        if os.path.exists(venv_python):
            return venv_python
        return None

    async def _setup_project_env(self, workspace: str, log: list[str]) -> str | None:
        """workspace에 격리된 가상환경을 생성하고 의존성을 설치한다.

        Returns:
            venv의 python 경로, 또는 실패/비해당 시 None.
        """
        # 이미 venv가 있으면 재사용
        existing = await self._get_venv_python(workspace)
        if existing:
            log.append(f"[env] 기존 venv 재사용: {existing}")
            return existing

        # requirements.txt 또는 pyproject.toml이 있을 때만 Python venv 생성
        has_python_deps = any(
            os.path.exists(os.path.join(workspace, f))
            for f in ("requirements.txt", "pyproject.toml")
        )
        if not has_python_deps:
            return None

        # uv가 있으면 uv venv, 없으면 python -m venv
        uv_available = await self._check_command_exists("uv")
        venv_path = os.path.join(workspace, ".venv")

        if uv_available:
            create_cmd = ["uv", "venv", venv_path]
        else:
            # python3 우선, 없으면 python
            py_cmd = "python3" if await self._check_command_exists("python3") else "python"
            create_cmd = [py_cmd, "-m", "venv", venv_path]

        log.append(f"[env] venv 생성: {' '.join(create_cmd)}")
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *create_cmd,
                    cwd=workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self._VENV_TIMEOUT_S,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.append(f"[env] ⚠ venv 생성 실패: {stderr.decode()[:300]}")
                return None
        except (TimeoutError, Exception) as e:
            log.append(f"[env] ⚠ venv 생성 오류: {e}")
            return None

        venv_python = os.path.join(venv_path, "bin", "python")
        if not os.path.exists(venv_python):
            log.append("[env] ⚠ venv python 바이너리를 찾을 수 없음")
            return None

        log.append(f"[env] ✓ venv 생성 완료: {venv_python}")
        return venv_python

    async def _install_dependencies(
        self, workspace: str, venv_python: str | None, log: list[str]
    ) -> None:
        """workspace의 의존성 파일을 감지하여 격리된 환경에 설치한다."""

        dep_files: list[tuple[str, list[str]]] = []

        # Python 의존성
        req_path = os.path.join(workspace, "requirements.txt")
        pyproject_path = os.path.join(workspace, "pyproject.toml")
        if venv_python and os.path.exists(req_path):
            uv_available = await self._check_command_exists("uv")
            if uv_available:
                dep_files.append(("requirements.txt", [
                    "uv", "pip", "install",
                    "--python", venv_python,
                    "-r", "requirements.txt",
                ]))
                # pytest도 함께 설치
                dep_files.append(("pytest", [
                    "uv", "pip", "install",
                    "--python", venv_python,
                    "pytest",
                ]))
            else:
                dep_files.append(("requirements.txt", [
                    venv_python, "-m", "pip", "install", "-r", "requirements.txt",
                ]))
                dep_files.append(("pytest", [
                    venv_python, "-m", "pip", "install", "pytest",
                ]))
        elif venv_python and os.path.exists(pyproject_path):
            dep_files.append(("pyproject.toml", [venv_python, "-m", "pip", "install", "-e", "."]))

        # Node.js 의존성
        pkg_path = os.path.join(workspace, "package.json")
        if os.path.exists(pkg_path) and await self._check_command_exists("npm"):
            dep_files.append(("package.json", ["npm", "install"]))

        # Go 의존성
        gomod_path = os.path.join(workspace, "go.mod")
        if os.path.exists(gomod_path) and await self._check_command_exists("go"):
            dep_files.append(("go.mod", ["go", "mod", "tidy"]))

        # Rust 의존성
        cargo_path = os.path.join(workspace, "Cargo.toml")
        if os.path.exists(cargo_path) and await self._check_command_exists("cargo"):
            dep_files.append(("Cargo.toml", ["cargo", "build"]))

        for label, cmd in dep_files:
            log.append(f"[install_deps] {' '.join(cmd)}")
            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        cwd=workspace,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    ),
                    timeout=self._INSTALL_TIMEOUT_S,
                )
                _, stderr = await proc.communicate()
                if proc.returncode == 0:
                    log.append(f"[install_deps] ✓ {label} 완료")
                else:
                    log.append(f"[install_deps] ⚠ {label} 실패: {stderr.decode()[:300]}")
            except TimeoutError:
                log.append(f"[install_deps] ⚠ {label} 타임아웃")
            except Exception as e:
                log.append(f"[install_deps] ⚠ {label} 오류: {e}")

    # ── 실행 기반 검증 ──

    async def _run_tests(self, state: CodingState) -> dict[str, Any]:
        """디스크에 저장된 코드를 격리된 프로젝트 환경에서 실제로 실행하여 검증한다.

        0단계: 런타임 감지 + venv 생성 + 의존성 설치
        1단계: syntax check (py_compile)
        1.5단계: JS/TS/Vue 문법 검증 (node --check, tsc --noEmit)
        2단계: 프론트엔드 테스트 실행 (jest / vitest)
        3단계: pytest 실행 (workspace venv의 python 사용)

        실패 시 test_passed=False + 에러 메시지를 test_output에 저장.
        """
        written_files = state.get("written_files", [])
        log = list(state.get("execution_log", []))

        if not written_files:
            log.append("[run_tests] 저장된 파일 없음 — 스킵")
            return {
                "test_passed": True,
                "test_output": "",
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

        workspace = os.environ.get("CODE_TOOLS_WORKSPACE", os.getcwd())

        # 파일 경로에서 실제 경로 추출 ("app.py (+24 lines)" → "app.py")
        real_files = []
        for f in written_files:
            path = f.split(" (")[0].strip()
            real_files.append(path)

        py_files = [f for f in real_files if f.endswith(".py")]
        test_files = [f for f in real_files if "test" in f.lower() and f.endswith(".py")]
        # 프론트엔드 테스트 파일 감지 (.test.js, .spec.ts 등)
        _FE_TEST_PATTERN = re.compile(
            r"\.(test|spec)\.(js|jsx|ts|tsx|mjs|cjs)$", re.IGNORECASE,
        )
        fe_test_files = [f for f in real_files if _FE_TEST_PATTERN.search(f)]

        # 0단계: 런타임 감지 + venv 생성 + 의존성 설치
        needed_langs = []
        if py_files:
            needed_langs.append("python")
        js_files = [f for f in real_files if f.endswith((".js", ".ts", ".jsx", ".tsx", ".vue"))]
        if js_files:
            needed_langs.append("node")

        runtimes = await self._detect_runtimes(needed_langs, log)
        missing = [lang for lang, found in runtimes.items() if not found]
        if missing:
            log.append(f"[run_tests] 런타임 미설치: {missing} — 테스트 스킵")
            return {
                "test_passed": True,
                "test_output": f"런타임 미설치로 스킵: {missing}",
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

        # 0.5단계: 환경 설정 승인 요청 (HITL)
        # 이미 venv가 있으면 승인 스킵 (재사용)
        existing_venv = await self._get_venv_python(workspace)
        if not existing_venv and not state.get("env_approved"):
            # 의존성 파일 내용 미리 수집
            dep_summary = []
            req_path = os.path.join(workspace, "requirements.txt")
            if os.path.exists(req_path):
                with open(req_path) as f:
                    dep_summary.append(f"requirements.txt:\n{f.read()[:500]}")
            pkg_path = os.path.join(workspace, "package.json")
            if os.path.exists(pkg_path):
                dep_summary.append("package.json 존재")

            env_info = {
                "type": "env_approval",
                "venv_path": os.path.join(workspace, ".venv"),
                "workspace": workspace,
                "runtimes": {k: v for k, v in runtimes.items() if v},
                "dependencies": "\n".join(dep_summary) if dep_summary else "(의존성 파일 없음)",
            }
            log.append("[run_tests] 환경 설정 승인 대기 (HITL interrupt)")

            try:
                response = interrupt(env_info)
            except RuntimeError:
                # 그래프 컨텍스트 밖 (테스트 등) — 자동 승인
                response = True

            if not response:
                # 거부: 실행 기반 검증 스킵, LLM 검증만으로 진행
                log.append("[run_tests] 환경 설정 거부 — 실행 검증 스킵")
                return {
                    "test_passed": True,
                    "test_output": "환경 설정 거부 — LLM 검증만 통과",
                    "execution_log": log,
                    "iteration": state.get("iteration", 0) + 1,
                }
            log.append("[run_tests] 환경 설정 승인됨")

        # Python venv 생성 + 의존성 설치
        # 재시도 시: 의존성 파일(requirements.txt 등)이 변경된 경우에만 재설치
        iteration = state.get("iteration", 0)
        venv_python = None
        should_install_deps = iteration == 0
        if iteration > 0:
            written = state.get("written_files", [])
            dep_files = {"requirements.txt", "package.json", "go.mod", "Cargo.toml",
                         "pyproject.toml", "setup.py", "setup.cfg"}
            if any(f.split(" (")[0].strip() in dep_files for f in written):
                should_install_deps = True
                log.append("[run_tests] 의존성 파일 변경 감지 — 재설치")

        if py_files:
            venv_python = await self._setup_project_env(workspace, log)
            if should_install_deps:
                await self._install_dependencies(workspace, venv_python, log)

        # Node.js / Go / Rust 의존성 설치 (venv 불필요)
        if not py_files and should_install_deps:
            await self._install_dependencies(workspace, None, log)

        # 실제 사용할 python 경로 결정
        project_python = venv_python or "python3"

        errors: list[str] = []

        # 1단계: syntax check
        for filepath in py_files:
            full_path = os.path.join(workspace, filepath)
            if not os.path.exists(full_path):
                continue
            try:
                import py_compile
                py_compile.compile(full_path, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"[syntax] {filepath}: {e}")

        if errors:
            output = "\n".join(errors)
            log.append(f"[run_tests] syntax 오류 {len(errors)}건")
            return {
                "test_passed": False,
                "test_output": output,
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

        # 1.5단계: JS/TS/Vue 문법 검증 (node --check)
        js_check_files = [f for f in real_files if f.endswith((".js", ".mjs", ".cjs"))]
        ts_files = [f for f in real_files if f.endswith((".ts", ".tsx"))]
        vue_files = [f for f in real_files if f.endswith(".vue")]

        if js_check_files and runtimes.get("node"):
            for filepath in js_check_files:
                full_path = os.path.join(workspace, filepath)
                if not os.path.exists(full_path):
                    continue
                try:
                    proc = await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            "node", "--check", full_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        ),
                        timeout=15,
                    )
                    _, stderr_out = await proc.communicate()
                    if proc.returncode != 0:
                        errors.append(f"[js-syntax] {filepath}: {stderr_out.decode()[:500]}")
                except (TimeoutError, Exception) as e:
                    log.append(f"[run_tests] node --check 실패 ({filepath}): {e}")

        if ts_files and runtimes.get("node"):
            # npx tsc --noEmit 사용 (tsconfig가 있을 때만)
            tsconfig_path = os.path.join(workspace, "tsconfig.json")
            if os.path.exists(tsconfig_path):
                try:
                    proc = await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            "npx", "tsc", "--noEmit",
                            cwd=workspace,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        ),
                        timeout=30,
                    )
                    stdout_out, stderr_out = await proc.communicate()
                    if proc.returncode != 0:
                        output = stdout_out.decode()[:500] + stderr_out.decode()[:500]
                        errors.append(f"[ts-check] TypeScript 오류:\n{output}")
                except (TimeoutError, Exception) as e:
                    log.append(f"[run_tests] tsc --noEmit 실패: {e}")

        if errors:
            output = "\n".join(errors)
            log.append(f"[run_tests] JS/TS 문법 오류 {len(errors)}건")
            return {
                "test_passed": False,
                "test_output": output,
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

        # 2단계: 프론트엔드 테스트 실행 (jest / vitest)
        frontend_only = not py_files and (js_check_files or ts_files or vue_files)
        fe_test_result = None
        if fe_test_files and runtimes.get("node"):
            fe_test_result = await self._run_frontend_tests(
                workspace, fe_test_files, log,
            )

        # 3단계: pytest 실행 (격리된 venv의 pytest 사용)
        if not test_files:
            # 프론트엔드 테스트 결과가 있으면 그 결과를 반환
            if fe_test_result is not None:
                return {
                    **fe_test_result,
                    "iteration": state.get("iteration", 0) + 1,
                }
            label = "프론트엔드 문법 검증 통과" if frontend_only else "syntax 검증 통과"
            log.append(f"[run_tests] 테스트 파일 없음 — {label}")
            return {
                "test_passed": True,
                "test_output": f"{label} (테스트 파일 없음)",
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

        test_paths = [os.path.join(workspace, f) for f in test_files]
        try:
            # 격리된 venv의 python으로 pytest 실행
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    project_python, "-m", "pytest",
                    "--tb=short", "-q", "--no-header",
                    "-p", "no:cacheprovider",
                    *test_paths,
                    cwd=workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={
                        **os.environ,
                        "PYTHONPATH": workspace,
                        "VIRTUAL_ENV": os.path.join(workspace, ".venv"),
                    },
                ),
                timeout=self._RUN_TESTS_TIMEOUT_S,
            )
            stdout, stderr = await proc.communicate()
            output = self._smart_truncate_test_output(stdout.decode(), limit=3000)
            if stderr:
                output += "\n" + stderr.decode()[:1000]

            passed = proc.returncode == 0
            log.append(f"[run_tests] pytest {'통과' if passed else '실패'} (venv={project_python}): {test_files}")

            # 프론트엔드 테스트 결과 병합
            if fe_test_result is not None:
                fe_passed = fe_test_result["test_passed"]
                passed = passed and fe_passed
                output = output + "\n\n--- Frontend Tests ---\n" + fe_test_result["test_output"]

            # 테스트 통과 + 이전에 실패가 있었으면 "에러→수정" 패턴을 Procedural Memory에 저장
            if passed and state.get("iteration", 0) > 0:
                self._save_fix_pattern(state)

            return {
                "test_passed": passed,
                "test_output": output,
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }
        except TimeoutError:
            log.append(f"[run_tests] 타임아웃 ({self._RUN_TESTS_TIMEOUT_S}s) — 통과 처리")
            return {
                "test_passed": True,
                "test_output": "테스트 타임아웃 — 스킵",
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }
        except Exception as e:
            log.append(f"[run_tests] 실행 오류: {e}")
            return {
                "test_passed": True,
                "test_output": f"테스트 실행 실패: {e}",
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

    # ── 프론트엔드 테스트 실행 ─────────────────────────────────

    _FE_TEST_TIMEOUT_S: int = 60

    async def _run_frontend_tests(
        self,
        workspace: str,
        fe_test_files: list[str],
        log: list[str],
    ) -> dict[str, Any]:
        """jest 또는 vitest로 프론트엔드 테스트를 실행한다.

        탐지 우선순위:
        1. package.json의 scripts.test가 있으면 `npm test` 사용
        2. vitest.config / vite.config 존재 → `npx vitest run`
        3. jest.config 존재 → `npx jest`
        4. 폴백: `npx jest` (가장 보편적)
        """
        runner_cmd: list[str] = []
        runner_label = ""

        pkg_path = os.path.join(workspace, "package.json")
        has_test_script = False
        if os.path.exists(pkg_path):
            try:
                import json

                with open(pkg_path) as f:
                    pkg = json.load(f)
                scripts = pkg.get("scripts", {})
                test_script = scripts.get("test", "")
                # "echo \"Error: no test specified\" && exit 1" 같은 기본값 제외
                if test_script and "no test specified" not in test_script:
                    has_test_script = True
            except Exception:
                pass

        if has_test_script:
            runner_cmd = ["npm", "test", "--"]
            runner_label = "npm test"
        elif any(
            os.path.exists(os.path.join(workspace, cfg))
            for cfg in ("vitest.config.ts", "vitest.config.js", "vitest.config.mts")
        ):
            runner_cmd = ["npx", "vitest", "run"]
            runner_label = "vitest"
        elif any(
            os.path.exists(os.path.join(workspace, cfg))
            for cfg in (
                "vite.config.ts", "vite.config.js", "vite.config.mts",
            )
        ):
            # vite 프로젝트는 vitest 사용 가능성 높음
            runner_cmd = ["npx", "vitest", "run"]
            runner_label = "vitest (vite project)"
        elif any(
            os.path.exists(os.path.join(workspace, cfg))
            for cfg in ("jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.cjs", "jest.config.json")
        ):
            runner_cmd = ["npx", "jest"]
            runner_label = "jest"
        else:
            # 폴백: npx jest (가장 보편적)
            runner_cmd = ["npx", "jest"]
            runner_label = "jest (fallback)"

        # 테스트 파일 경로 추가
        test_paths = [os.path.join(workspace, f) for f in fe_test_files]
        cmd = [*runner_cmd, *test_paths]

        log.append(f"[run_tests] 프론트엔드 테스트 실행: {runner_label} ({len(fe_test_files)}개 파일)")

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "CI": "true"},
                ),
                timeout=self._FE_TEST_TIMEOUT_S,
            )
            stdout, stderr = await proc.communicate()
            output = self._smart_truncate_test_output(stdout.decode(), limit=3000)
            if stderr:
                output += "\n" + stderr.decode()[:1000]

            passed = proc.returncode == 0
            log.append(
                f"[run_tests] {runner_label} {'통과' if passed else '실패'}: {fe_test_files}"
            )
            return {
                "test_passed": passed,
                "test_output": output,
                "execution_log": log,
            }
        except TimeoutError:
            log.append(f"[run_tests] {runner_label} 타임아웃 ({self._FE_TEST_TIMEOUT_S}s) — 스킵")
            return {
                "test_passed": True,
                "test_output": f"{runner_label} 타임아웃 — 스킵",
                "execution_log": log,
            }
        except FileNotFoundError:
            log.append(f"[run_tests] {runner_label} 실행 불가 (npx/npm 없음) — 스킵")
            return {
                "test_passed": True,
                "test_output": f"{runner_label} 실행 불가 — 스킵",
                "execution_log": log,
            }
        except Exception as e:
            log.append(f"[run_tests] {runner_label} 오류: {e}")
            return {
                "test_passed": True,
                "test_output": f"{runner_label} 실행 실패: {e}",
                "execution_log": log,
            }

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

    # ── 코드 적용 (파일 저장) ──────────────────────────────

    async def _apply_code(self, state: CodingState) -> dict[str, Any]:
        """생성된 코드에서 파일 경로를 추출하여 디스크에 저장한다."""
        generated_code = state.get("generated_code", "")
        if not generated_code:
            return {"written_files": []}

        blocks = _extract_code_blocks(generated_code)
        if not blocks:
            return {"written_files": []}

        workspace = os.environ.get("CODE_TOOLS_WORKSPACE", os.getcwd())
        workspace_resolved = os.path.realpath(workspace)
        written: list[str] = []
        log = list(state.get("execution_log", []))

        for block in blocks:
            filepath = block["filepath"]
            code = block["code"]

            # 절대 경로 계산
            if os.path.isabs(filepath):
                full_path = filepath
            else:
                full_path = os.path.join(workspace, filepath)

            resolved = os.path.realpath(full_path)

            # 보안: workspace 밖 쓰기 금지
            if (
                not resolved.startswith(workspace_resolved + os.sep)
                and resolved != workspace_resolved
            ):
                log.append(f"[apply] workspace 외부 경로 스킵: {filepath}")
                continue

            # 보안: 금지 경로 체크
            if any(forbidden in resolved for forbidden in _FORBIDDEN_PATHS):
                log.append(f"[apply] 금지 경로 스킵: {filepath}")
                continue

            try:
                dir_path = os.path.dirname(resolved)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                with open(resolved, "w", encoding="utf-8") as f:
                    f.write(code)
                    if not code.endswith("\n"):
                        f.write("\n")
                line_count = code.count("\n") + 1
                written.append(f"{filepath} (+{line_count} lines)")
                log.append(f"[apply] ✓ {filepath} (+{line_count} lines)")
            except OSError as e:
                log.append(f"[apply] ✗ {filepath}: {e}")

        return {
            "written_files": written,
            "execution_log": log,
        }

    # ── 라우팅 ──────────────────────────────────────────────

    def _should_use_tools(self, state: CodingState) -> str:
        """execute 후 라우팅 판단.

        우선순위:
        1. exit_reason이 설정됨 → GENERATE_FINAL (안전장치 발동)
        2. 도구 호출 한도 도달 → GENERATE_FINAL (루프 강제 탈출)
        3. 도구 호출이 있으면 → EXECUTE_TOOLS (ReAct 루프 계속)
        4. 도구를 한 번이라도 사용했으면 → GENERATE_FINAL (STRONG으로 최종 생성)
        5. 도구를 전혀 사용하지 않았으면 → VERIFY (FAST 출력 그대로 검증)
        """
        # 안전장치 발동 시 즉시 최종 생성으로 전환
        exit_reason = state.get("exit_reason", "")
        if exit_reason:
            return self.get_node_name("GENERATE_FINAL")

        # generate/scaffold 작업 → ReAct 루프 불필요, STRONG 모델로 직접 생성
        parse_result = state.get("parse_result", {})
        task_type = parse_result.get("task_type", "generate")
        if task_type in ("generate", "scaffold"):
            return self.get_node_name("GENERATE_FINAL")

        tool_call_count = state.get("tool_call_count", 0)
        max_calls = self._coding_config.max_tool_calls

        # 도구 호출 한도 도달 시 강제 탈출 (recursion_limit 방어)
        if tool_call_count >= max_calls:
            return self.get_node_name("GENERATE_FINAL")

        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", None) or []

        if tool_calls:
            return self.get_node_name("EXECUTE_TOOLS")

        # 도구를 한 번이라도 사용했으면 STRONG 모델로 최종 생성
        if tool_call_count > 0:
            return self.get_node_name("GENERATE_FINAL")

        # 도구 미사용 → FAST 출력 그대로 검증 (불필요한 STRONG 호출 생략)
        return self.get_node_name("VERIFY")

    def _should_retry(self, state: CodingState) -> str:
        """LLM 검증 실패 시 재시도 여부를 판단한다."""
        verify_result = state.get("verify_result", {})
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if verify_result.get("passed", True):
            return self.get_node_name("APPLY_CODE")
        if iteration >= max_iterations:
            return self.get_node_name("APPLY_CODE")
        return self.get_node_name("EXECUTE")

    def _should_retry_tests(self, state: CodingState) -> str:
        """실행 기반 테스트 실패 시 수정 루프를 결정한다.

        테스트 통과 → END
        테스트 실패 + iteration 남음 → GENERATE_FINAL (에러 메시지가 messages에 주입됨)
        테스트 실패 + iteration 소진 → END (최선의 결과로 종료)
        """
        test_passed = state.get("test_passed", True)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if test_passed:
            return END

        if iteration >= max_iterations:
            logging.getLogger(__name__).warning("[run_tests] 최대 반복 도달 — 테스트 실패 상태로 종료")
            return END

        # 실패 시: INJECT_TEST_FAILURE 노드에서 에러를 messages에 주입 → GENERATE_FINAL
        return self.get_node_name("INJECT_TEST_FAILURE")

    # ── 학습 패턴 저장 (Procedural Memory) ──────────────────────

    def _save_fix_pattern(self, state: CodingState) -> None:
        """테스트 실패→성공 사이클에서 학습된 수정 패턴을 Procedural Memory에 저장한다.

        이전 테스트 에러와 현재 성공한 코드를 비교하여
        "에러 유형 → 해결 방법" 패턴을 추출한다.
        """
        if not self._memory_store:
            return

        prev_test_output = state.get("_prev_test_output", "")
        if not prev_test_output:
            return

        # 이전 에러 분류
        error_type, _ = self._classify_test_error(prev_test_output)
        if error_type == "Unknown":
            # 알 수 없는 에러는 저장하지 않음 (노이즈 방지)
            return

        parse_result = state.get("parse_result", {})
        task_type = parse_result.get("task_type", "generate")
        language = parse_result.get("language", "python")

        # 수정된 코드에서 핵심 변경 사항 추출
        generated_code = state.get("generated_code", "")
        code_snippet = generated_code[:500] if generated_code else ""

        description = (
            f"[{error_type}] {language}/{task_type} 프로젝트에서 발생. "
            f"에러: {prev_test_output[:200]}... "
            f"해결: 코드 재생성으로 수정 성공."
        )

        try:
            result = self._memory_store.accumulate_skill(
                code=code_snippet,
                description=description,
                tags=[error_type.lower(), language, task_type, "test_fix"],
            )
            if result:
                logging.getLogger(__name__).info(
                    "[memory] 수정 패턴 저장: %s (%s)", error_type, result.id[:8]
                )
        except Exception as e:
            logging.getLogger(__name__).debug("[memory] 패턴 저장 실패 (무시): %s", e)

    # ── 테스트 출력 처리 ───────────────────────────────────────

    @staticmethod
    def _smart_truncate_test_output(raw: str, limit: int = 3000) -> str:
        """pytest 출력에서 에러 섹션을 우선 추출하여 절단한다.

        단순 [:limit] 절단 시 진행 바(FFFF...)만 남고 실제 에러가 잘리는 문제를 해결.
        FAILURES/ERRORS 섹션을 우선 포함하고, 남은 예산으로 요약 줄을 추가한다.
        """
        if len(raw) <= limit:
            return raw

        lines = raw.split("\n")
        # FAILURES 또는 ERRORS 섹션 시작점 찾기
        failure_start = -1
        for i, line in enumerate(lines):
            if line.startswith("=") and ("FAILURES" in line or "ERRORS" in line):
                failure_start = i
                break

        if failure_start >= 0:
            # 에러 섹션부터 끝까지 추출
            error_section = "\n".join(lines[failure_start:])
            # 첫 줄(진행 표시)과 마지막 요약 줄도 포함
            summary_line = lines[0] if lines else ""
            short_summary = lines[-1] if len(lines) > 1 else ""
            result = f"{summary_line}\n...\n{error_section}\n{short_summary}"
            return result[:limit]

        # FAILURES 섹션이 없으면 마지막 부분 우선 (에러는 보통 뒤에 출력)
        tail = raw[-limit:]
        if not tail.startswith("\n"):
            # 줄 중간 절단 방지
            first_newline = tail.find("\n")
            if first_newline > 0:
                tail = tail[first_newline + 1:]
        return "...\n" + tail

    # ── 에러 유형 분석 ─────────────────────────────────────────

    _ERROR_PATTERNS: ClassVar[list[tuple[str, str, str]]] = [
        # (정규식 패턴, 에러 유형, 수정 지시)
        (
            r"circular import|ImportError.*cannot import name.*most likely due to a circular",
            "CircularImport",
            "순환 import 감지. 해결: 1) 공유 객체를 extensions.py로 분리 "
            "2) Factory 패턴(create_app) 사용 3) import를 함수 내부로 이동",
        ),
        (
            r"ModuleNotFoundError|No module named",
            "ModuleNotFoundError",
            "모듈 미설치 또는 경로 오류. 1) 프로젝트 내 모듈이면 해당 .py 파일을 함께 생성하세요 "
            "(예: backend/config.py, backend/extensions.py 등). "
            "2) 외부 패키지면 requirements.txt에 추가하세요. "
            "3) import 경로가 프로젝트 디렉토리 구조와 일치하는지 확인하세요.",
        ),
        (
            r"ImportError.*cannot import name",
            "ImportError",
            "import 대상이 존재하지 않음. 해당 함수/클래스가 정의된 파일이 실제로 존재하는지 확인하세요. "
            "없으면 해당 파일을 생성하세요. "
            "예: from backend import create_app이 실패하면 backend/__init__.py에 create_app 함수를 정의하세요.",
        ),
        (
            r"ImportError",
            "ImportError",
            "import 실패. __init__.py 존재 여부, 패키지 경로, 절대/상대 import 일관성을 확인하세요. "
            "참조하는 모듈 파일이 존재하지 않으면 함께 생성하세요.",
        ),
        (
            r"SyntaxError",
            "SyntaxError",
            "문법 오류. 괄호 짝, 들여쓰기, 콜론 누락 등을 확인하세요.",
        ),
        (
            r"NameError",
            "NameError",
            "정의되지 않은 변수/함수. 오타, import 누락, 스코프 문제를 확인하세요.",
        ),
        (
            r"AttributeError",
            "AttributeError",
            "존재하지 않는 속성 접근. 클래스/모듈의 실제 인터페이스를 확인하세요.",
        ),
        (
            r"TypeError",
            "TypeError",
            "타입 불일치 또는 인자 개수 오류. 함수 시그니처와 호출부를 대조하세요.",
        ),
        (
            r"OperationalError|no such table|ProgrammingError.*relation.*does not exist|"
            r"no such column|table.*already exists",
            "DatabaseError",
            "DB 테이블/컬럼 미생성 또는 스키마 불일치. "
            "테스트 fixture에서 app_context 내 db.create_all()을 호출하세요. "
            "conftest.py에 app/db fixture를 정의하고, teardown에서 db.drop_all()을 실행하세요.",
        ),
        (
            r"IntegrityError|UNIQUE constraint|duplicate key|NOT NULL constraint",
            "IntegrityError",
            "DB 무결성 제약 위반. 테스트 간 데이터 격리를 확인하세요. "
            "fixture에서 매 테스트마다 트랜잭션 롤백 또는 db.drop_all() + db.create_all()을 수행하세요.",
        ),
        (
            r"ConnectionRefusedError|could not connect to server|Connection refused",
            "ConnectionError",
            "외부 서비스 연결 실패. 테스트에서는 인메모리 DB(sqlite:///:memory:)를 사용하고, "
            "외부 API는 mock 처리하세요.",
        ),
    ]

    def _classify_test_error(self, test_output: str) -> tuple[str, str]:
        """테스트 출력에서 에러 유형을 분류하고 수정 지시를 반환한다."""
        for pattern, error_type, guidance in self._ERROR_PATTERNS:
            if re.search(pattern, test_output, re.IGNORECASE):
                return error_type, guidance
        return "Unknown", "에러 메시지를 분석하여 원인을 파악하고 수정하세요."

    async def _inject_test_failure(self, state: CodingState) -> dict[str, Any]:
        """테스트 실패 에러를 messages에 주입하여 GENERATE_FINAL이 수정 코드를 생성하게 한다."""
        test_output = state.get("test_output", "")
        log = list(state.get("execution_log", []))
        iteration = state.get("iteration", 0)

        # 반복 감지 1: 동일 코드 반복 (유사도 비교)
        generated_code = state.get("generated_code", "")
        prev_code = state.get("_prev_generated_code", "")
        if prev_code and generated_code:
            norm_cur = re.sub(r"\s+", " ", generated_code).strip()
            norm_prev = re.sub(r"\s+", " ", prev_code).strip()
            if norm_cur == norm_prev or (
                len(norm_cur) > 100
                and abs(len(norm_cur) - len(norm_prev)) < len(norm_cur) * 0.05
                and norm_cur[:500] == norm_prev[:500]
            ):
                log.append(f"[inject_test_failure] 동일 코드 반복 감지 — 재시도 중단")
                return {
                    "test_passed": True,
                    "test_output": f"동일 코드 반복으로 재시도 중단 (시도 {iteration}회)\n{test_output[:500]}",
                    "execution_log": log,
                }

        # 반복 감지 2: 동일 테스트 에러 반복 (연속 3회 같은 에러면 중단)
        prev_test_output = state.get("_prev_test_output", "")
        if prev_test_output and test_output and iteration >= 3:
            # FAILURES 섹션만 추출하여 비교 (타임스탬프 등 무시)
            # pytest: FAILED/ERROR/assert, Go: --- FAIL/panic, Rust: failures:/panicked, Jest: FAIL/●
            _FAIL_KEYWORDS = ("FAILED", "ERROR", "assert", "--- FAIL", "panic",
                              "failures:", "panicked", "FAIL ", "●", "Expected")

            def _extract_failures(s: str) -> str:
                lines = [l for l in s.split("\n") if any(kw in l for kw in _FAIL_KEYWORDS)]
                return "\n".join(lines[:20])
            cur_failures = _extract_failures(test_output)
            prev_failures = _extract_failures(prev_test_output)
            if cur_failures and cur_failures == prev_failures:
                log.append(f"[inject_test_failure] 동일 테스트 에러 3회 이상 반복 — 재시도 중단")
                return {
                    "test_passed": True,
                    "test_output": f"동일 에러 반복으로 재시도 중단 (시도 {iteration}회)\n{test_output[:500]}",
                    "execution_log": log,
                }

        error_type, guidance = self._classify_test_error(test_output)
        log.append(f"[inject_test_failure] 에러 유형: {error_type}")

        # 이전 phase 파일 목록 수집 (MCP read_file로 읽을 수 있도록 안내)
        workspace = os.environ.get("CODE_TOOLS_WORKSPACE", os.getcwd())
        written_files = state.get("written_files", [])
        existing_hint = ""
        if written_files:
            all_files: list[str] = []
            for root, _dirs, files in os.walk(workspace):
                for fname in files:
                    rel = os.path.relpath(os.path.join(root, fname), workspace)
                    if not rel.startswith((".venv", "__pycache__", ".git", "node_modules")):
                        all_files.append(rel)
            if all_files:
                existing_hint = (
                    "\n\n### 프로젝트 내 기존 파일 (수정 가능)\n"
                    f"```\n{chr(10).join(sorted(all_files)[:30])}\n```\n"
                    "**위 파일 중 에러의 원인이 되는 파일이 있다면 해당 파일도 함께 수정하세요.**\n"
                    "예: 초기화 파일에 라우트/모듈 등록이 누락되었거나, "
                    "설정 파일에 의존성이 빠져있다면 해당 파일도 수정 코드에 포함하세요.\n"
                )

        error_msg = HumanMessage(content=(
            f"## 테스트 실행 실패 (시도 {iteration}회)\n\n"
            f"### 에러 유형: {error_type}\n"
            f"**수정 방향**: {guidance}\n\n"
            f"### 에러 출력:\n"
            f"```\n{test_output[:2000]}\n```\n"
            f"{existing_hint}\n"
            f"위 수정 방향을 반드시 따라 코드를 수정하세요. "
            f"수정된 전체 파일을 코드 블록으로 제공하세요 (filepath 주석 포함)."
        ))
        log.append(f"[inject_test_failure] 에러 주입, 수정 루프 시작 (iteration={iteration})")

        return {
            "messages": [error_msg],
            "execution_log": log,
            "_prev_generated_code": generated_code,
            "_prev_test_output": test_output,
        }

    # ── 그래프 구성 ─────────────────────────────────────────

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("PARSE"), self._parse_request)
        graph.add_node(self.get_node_name("RETRIEVE_MEMORY"), self._retrieve_memory)
        graph.add_node(self.get_node_name("EXECUTE"), self._execute_code)
        graph.add_node(self.get_node_name("EXECUTE_TOOLS"), self._execute_tools)
        graph.add_node(self.get_node_name("GENERATE_FINAL"), self._generate_final)
        graph.add_node(self.get_node_name("VERIFY"), self._verify_result)
        graph.add_node(self.get_node_name("APPLY_CODE"), self._apply_code)
        graph.add_node(self.get_node_name("RUN_TESTS"), self._run_tests)
        graph.add_node(self.get_node_name("INJECT_TEST_FAILURE"), self._inject_test_failure)

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
        # execute(FAST) → (tools or generate_final)
        graph.add_conditional_edges(
            self.get_node_name("EXECUTE"),
            self._should_use_tools,
        )
        # tools → execute (ReAct 루프 — FAST 모델)
        graph.add_edge(
            self.get_node_name("EXECUTE_TOOLS"),
            self.get_node_name("EXECUTE"),
        )
        # generate_final(STRONG) → verify(LLM)
        graph.add_edge(
            self.get_node_name("GENERATE_FINAL"),
            self.get_node_name("VERIFY"),
        )
        # verify(LLM) → (retry or apply_code)
        graph.add_conditional_edges(
            self.get_node_name("VERIFY"),
            self._should_retry,
        )
        # apply_code → run_tests (파일 저장 후 실제 실행)
        graph.add_edge(
            self.get_node_name("APPLY_CODE"),
            self.get_node_name("RUN_TESTS"),
        )
        # run_tests → (END or inject_test_failure for fix)
        graph.add_conditional_edges(
            self.get_node_name("RUN_TESTS"),
            self._should_retry_tests,
        )
        # inject_test_failure → generate_final (에러 메시지 주입 후 재생성)
        graph.add_edge(
            self.get_node_name("INJECT_TEST_FAILURE"),
            self.get_node_name("GENERATE_FINAL"),
        )
