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

from coding_agent.core.abort_controller import AbortController
from coding_agent.core.base_agent import BaseGraphAgent
from coding_agent.core.context_manager import ContextManager
from coding_agent.core.mcp_loader import MCPToolLoader
from coding_agent.core.memory.schemas import MemoryItem, MemoryType
from coding_agent.core.memory.store import MemoryStore
from coding_agent.core.skills.registry import SkillRegistry
from coding_agent.core.stall_detector import StallAction, StallDetector
from coding_agent.core.turn_budget import TurnBudgetTracker
from coding_agent.core.tool_call_utils import (
    sanitize_messages_for_llm,
    tc_args,
    tc_id,
    tc_name,
)
from coding_agent.core.tool_permissions import PermissionDecision

from .config import CodingConfig
from .prompts import (
    EXECUTE_SYSTEM_PROMPT,
    GENERATE_FINAL_SYSTEM_PROMPT,
)
from .schemas import CodingState

logger = logging.getLogger(__name__)


def _normalize_file_path(path: str, workspace: str | None = None) -> str:
    """파일 경로를 workspace 기준 상대 경로로 정규화한다.

    - 절대 경로가 workspace 하위이면 상대 경로로 변환
    - "./" 접두사, ".." 등을 제거
    - "app.py (+24 lines)" 같은 메타 접미사도 제거
    """
    # 메타 접미사 제거: "app.py (+24 lines)" → "app.py"
    raw = path.split(" (")[0].strip()
    if not raw:
        return raw

    # 절대 경로 → workspace 기준 상대 경로
    if workspace and os.path.isabs(raw):
        ws = os.path.normpath(workspace)
        norm = os.path.normpath(raw)
        # workspace 하위인 경우만 상대 경로로 변환
        if norm.startswith(ws + os.sep) or norm == ws:
            raw = os.path.relpath(norm, ws)

    # "./" 제거, ".." 정규화
    return os.path.normpath(raw)


async def _execute_tool_safely(tool: Any, args: dict) -> str:
    """도구를 안전하게 실행하고 결과를 문자열로 반환한다."""
    try:
        result = await tool.ainvoke(args)
        if not result:
            return "실행 완료 (출력 없음)"
        # MCP 도구는 [{'type': 'text', 'text': '...'}] 형태로 반환
        if isinstance(result, list):
            texts = [
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in result
            ]
            return "\n".join(texts) if texts else "실행 완료 (출력 없음)"
        return str(result)
    except Exception as e:
        return f"도구 실행 오류: {e}"


def _get_tools_by_name(tools: list[Any]) -> dict[str, Any]:
    """도구 목록을 이름→도구 딕셔너리로 변환한다."""
    return {getattr(t, "name", None): t for t in tools if getattr(t, "name", None)}


_FORBIDDEN_PATHS = (".claude/", ".git/", "__pycache__/", "node_modules/")


class CodingAssistantAgent(BaseGraphAgent):
    """MCP 도구를 활용하는 Coding Assistant Harness.

    parse → execute(ReAct 루프: LLM + MCP 도구) → verify → apply_code → END
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "RETRIEVE_MEMORY": "retrieve_memory",
        "GENERATE": "generate_code",
        "RUN_TESTS": "run_tests",
        "INJECT_ERROR": "inject_error",
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
        self._context_manager = ContextManager(
            max_tokens=getattr(self._coding_config, "max_context_tokens", 128000),
            compact_threshold=getattr(self._coding_config, "compact_threshold", 0.8),
        )

        # AbortController — 턴 시작 시 reset
        self._abort_controller = AbortController()

        # 미들웨어 체인 — LLM 호출 전후 메시지/컨텍스트 자동 관리
        # 양파 순서: Resilience(가장 바깥) → Window → Summarization → Memory(가장 안쪽)
        from coding_agent.core.middleware import (
            MemoryMiddleware,
            MessageWindowMiddleware,
            MiddlewareChain,
            ResilienceMiddleware,
            SummarizationMiddleware,
        )
        # DEFAULT 모델: 요약용 (코드 구조 이해가 필요하므로 FAST보다 상위 티어)
        # 요약은 100K 초과 시에만 드물게 발생, 비용 부담 적음
        summarize_model = self._coding_config.get_model("verification") if hasattr(
            self._coding_config, "get_model"
        ) else None

        # 양파 순서 (바깥→안쪽):
        # 1. Resilience: 재시도/타임아웃 (가장 바깥)
        # 2. Summarization(110K): LLM 요약으로 중복/반복 압축 (DEFAULT 모델)
        # 3. MessageWindow(100K): 토큰 기반 다단계 컴팩션 (규칙 기반 안전망)
        # 4. Memory: 메모리 컨텍스트 주입 (가장 안쪽)
        # ※ SLM 컨텍스트 128K 기준, 출력 토큰 ~16K 확보 → 입력 ~110K 사용 가능
        self._middleware_chain = MiddlewareChain([
            ResilienceMiddleware(abort_controller=self._abort_controller),
            SummarizationMiddleware(
                token_threshold=110_000,
                keep_recent_messages=8,
                max_tool_arg_chars=3000,
                summarize_model=summarize_model,  # DEFAULT 모델로 LLM 요약
            ),
            MessageWindowMiddleware(
                max_context_tokens=100_000,    # 128K 모델 기준 안전 마진
                tool_result_max_tokens=8_000,  # 개별 도구 결과 상한
                keep_recent=8,                 # 최근 보존 메시지 수
            ),
            MemoryMiddleware(
                memory_store=self._memory_store,
                slm_invoker=self._make_slm_invoker(),
            ),
        ])

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

    def _make_slm_invoker(self):
        """메모리 자동 축적용 SLM(FAST) 호출 함수를 생성한다."""
        try:
            slm = self._coding_config.get_model("parsing")  # FAST 티어
        except Exception:
            return None

        async def _invoke(messages):
            resp = await slm.ainvoke(messages)
            return resp.content if resp else ""

        return _invoke

    async def async_init(self) -> None:
        """MCP 도구를 비동기로 로드한다."""
        self._tools = await self._mcp_loader.load()
        if not self._tools:
            logging.getLogger(__name__).warning(
                "MCP 도구가 0개 로드됨 — write_file 없이는 코드 생성 불가. "
                "MCP 서버 연결을 확인하세요: %s",
                self._coding_config.mcp_servers,
            )

    # ── 모델 lazy init ──────────────────────────────────────

    def _get_gen_model(self) -> BaseChatModel:
        if self._gen_model is None:
            self._gen_model = self._coding_config.get_model("generation")
        return self._gen_model

    # ── 노드 구현 ──────────────────────────────────────────

    async def _retrieve_memory(self, state: CodingState) -> dict[str, Any]:
        """진입점: 안전장치 초기화 + Memory 검색.

        단순화된 그래프(v2)의 첫 번째 노드.
        기존 PARSE에서 수행하던 안전장치 초기화를 흡수하고,
        Episodic/Procedural Memory를 검색하여 상태에 주입한다.
        """
        # 안전장치 초기화 (기존 PARSE에서 이관)
        self._abort_controller.reset()
        self._stall_detector.reset()
        self._turn_budget.reset()

        # iteration/max_iterations 초기화 (Orchestrator가 설정하지 않은 경우)
        base_result: dict[str, Any] = {
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
            "tool_call_count": 0,
            "exit_reason": "",
        }

        parse_result = state.get("parse_result", {})
        description = parse_result.get("description", "")
        language = parse_result.get("language", "python")
        task_type = parse_result.get("task_type", "generate")

        # 검색 쿼리: 작업 설명 기반
        query = description or f"{task_type} {language}"

        result: dict[str, Any] = {}

        # ── Memory 검색 (memory_store가 있을 때만) ──
        if self._memory_store:
            # Procedural Memory 검색: 학습된 스킬 패턴
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
                pass

            # Episodic Memory 검색: 이전 실행 이력
            try:
                episodes = self._memory_store.search(
                    query=query,
                    memory_type=MemoryType.EPISODIC,
                    limit=3,
                )
                if episodes:
                    result["episodic_log"] = [e.content for e in episodes]
            except Exception:
                pass

            # User Profile 검색: 사용자 선호도/습관
            try:
                profiles = self._memory_store.search(
                    query=query,
                    memory_type=MemoryType.USER_PROFILE,
                    limit=5,
                )
                if profiles:
                    result["user_profile_context"] = [p.content for p in profiles]
            except Exception:
                pass

            # Domain Knowledge 검색: 도메인 지식
            try:
                domain = self._memory_store.search(
                    query=query,
                    memory_type=MemoryType.DOMAIN_KNOWLEDGE,
                    limit=5,
                )
                if domain:
                    result["domain_knowledge_context"] = [d.content for d in domain]
            except Exception:
                pass

        # ── Skill Registry: 활성 스킬 본문을 컨텍스트에 주입 ──
        if self._skill_registry:
            # planned_files가 있으면 scaffold로 판단 (Orchestrator에서 Phase 실행 시)
            planned_files = state.get("planned_files", [])
            skill_task_type = "scaffold" if planned_files else task_type

            # planned_files 확장자에서 framework 힌트 추출
            _fw_ext_map = {
                ".tsx": "fastapi_react", ".jsx": "react_express",
                ".vue": "flask_vue",
            }
            framework_hint = ""
            for f in planned_files:
                ext = "." + f.rsplit(".", 1)[-1] if "." in f else ""
                if ext in _fw_ext_map:
                    framework_hint = _fw_ext_map[ext]
                    break

            activated = self._skill_registry.auto_activate_for_task(
                skill_task_type, framework_hint=framework_hint
            )
            skill_bodies = self._skill_registry.get_active_skill_bodies()
            if skill_bodies:
                result["skill_context"] = skill_bodies
                logger.info(
                    "[스킬] CodingAssistant 활성화: %s (%d개 본문 주입)",
                    activated, len(skill_bodies),
                )

        return {**base_result, **result}

    @staticmethod
    def _detect_language_from_files(state: CodingState) -> str:
        """planned_files/written_files에서 주요 언어/프레임워크를 감지한다."""
        files = state.get("planned_files", []) or state.get("written_files", [])
        if not files:
            return state.get("parse_result", {}).get("language", "python")

        ext_map = {
            ".tsx": "TypeScript/React", ".ts": "TypeScript", ".jsx": "React/JavaScript",
            ".js": "JavaScript", ".py": "Python", ".go": "Go", ".rs": "Rust",
            ".java": "Java", ".vue": "Vue.js", ".svelte": "Svelte",
        }
        ext_count: dict[str, int] = {}
        for f in files:
            for ext, lang in ext_map.items():
                if f.endswith(ext):
                    ext_count[lang] = ext_count.get(lang, 0) + 1
                    break

        if not ext_count:
            return state.get("parse_result", {}).get("language", "python")

        # 가장 많은 확장자의 언어를 반환, 여러 개면 결합
        sorted_langs = sorted(ext_count.items(), key=lambda x: -x[1])
        top_langs = [lang for lang, _ in sorted_langs[:3]]
        return ", ".join(top_langs)

    def _build_execute_system_prompt(
        self, state: CodingState, *, purpose: str = "execute"
    ) -> str:
        """execute/generate 노드 공통 시스템 프롬프트를 구성한다.

        Args:
            purpose: "execute" — ReAct 도구 루프용, "generate" — 최종 코드 생성용
        """
        # Planner가 결정한 tech_stack을 파일 확장자에서 동적 감지
        language = self._detect_language_from_files(state)

        if purpose == "generate":
            # GENERATE_FINAL: write_file 도구 사용 지시 프롬프트
            system_prompt = self._build_system_prompt(
                GENERATE_FINAL_SYSTEM_PROMPT.format(language=language)
            )
        else:
            # EXECUTE: 기존 MCP 도구 루프 프롬프트
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

        # User Profile 주입
        user_profile_context = state.get("user_profile_context", [])
        if user_profile_context:
            system_prompt += "\n\n## 사용자 선호도 (User Profile)\n"
            system_prompt += "\n".join(f"- {ctx}" for ctx in user_profile_context)

        # Domain Knowledge 주입
        domain_knowledge_context = state.get("domain_knowledge_context", [])
        if domain_knowledge_context:
            system_prompt += "\n\n## 도메인 지식 (Domain Knowledge)\n"
            system_prompt += "\n".join(f"- {ctx}" for ctx in domain_knowledge_context)

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
            # 이전 생성 코드를 포함하여 LLM이 read_file 없이도 수정 가능하게 함
            prev_code = state.get("generated_code", "")
            if prev_code:
                # 토큰 폭발 방지: 최대 8000자
                truncated = prev_code[:8000]
                if len(prev_code) > 8000:
                    truncated += f"\n\n... (총 {len(prev_code)}자 중 8000자만 표시)"
                context_parts.append(
                    f"\n이전에 생성한 코드 (위 문제를 수정하세요):\n```\n{truncated}\n```"
                )
            context_parts.append("위 문제를 수정하여 다시 코드를 작성하세요.")

        return HumanMessage(content="\n".join(context_parts))

    @staticmethod
    def _format_manifest(manifest: dict) -> str:
        """FileManifest dict를 프롬프트 주입용 텍스트로 변환한다."""
        lines = ["### 파일 목록"]
        for f in manifest.get("files", []):
            lines.append(f"- `{f}`")

        contracts = manifest.get("contracts", [])
        if contracts:
            lines.append("\n### 함수 계약 (반드시 준수)")
            for c in contracts:
                params = ", ".join(c.get("params", []))
                callers = ", ".join(c.get("called_from", []))
                lines.append(
                    f"- `{c['name']}({params})` — 정의: `{c['defined_in']}`, "
                    f"호출: {callers or '(미정)'}"
                )

        shared = manifest.get("shared_objects", [])
        if shared:
            lines.append("\n### 공유 객체 (단일 출처 원칙)")
            for s in shared:
                importers = ", ".join(s.get("imported_by", []))
                lines.append(
                    f"- `{s['name']}` — 정의: `{s['defined_in']}`, "
                    f"import: {importers or '(미정)'}"
                )

        return "\n".join(lines)

    # write_file 도구 루프 최대 반복 횟수 — 단일 턴 완결을 위해 충분히 확보
    # 파일 수 제한 없이 LLM이 자율적으로 모든 파일을 한 턴에서 생성하도록 허용
    _GENERATE_MAX_TOOL_LOOPS = 30

    async def _generate_code(self, state: CodingState) -> dict[str, Any]:
        """단순화된 그래프(v2)의 코드 생성 노드.

        기존 GENERATE_FINAL + 정적 검증 + 마크다운 폴백을 통합한다.
        STRONG 모델이 write_file 도구로 파일을 직접 저장한다.
        """
        # 핵심: _generate_final 호출 (LLM이 코드 생성)
        result = await self._generate_final(state)

        # 반복 감지: LLM 호출 이후 이전 생성과 비교하여 동일하면 재시도 중단
        prev_code = state.get("_prev_generated_code", "")
        new_code = result.get("generated_code", "")
        if prev_code and new_code:
            norm_cur = re.sub(r"\s+", " ", new_code).strip()
            norm_prev = re.sub(r"\s+", " ", prev_code).strip()
            if norm_cur == norm_prev or (
                len(norm_cur) > 100
                and abs(len(norm_cur) - len(norm_prev)) < len(norm_cur) * 0.05
                and norm_cur[:500] == norm_prev[:500]
            ):
                iteration = state.get("iteration", 0)
                log = list(result.get("execution_log", []))
                log.append("[generate] 동일 코드 반복 감지 — 재시도 중단")
                return {
                    **result,
                    "test_passed": True,
                    "test_output": f"동일 코드 반복으로 재시도 중단 (시도 {iteration}회)",
                    "execution_log": log,
                }

        # 마크다운 폴백/정적 검증 제거 — LLM이 write_file 도구로 직접 저장
        # validate_consistency MCP 도구가 제공되므로 LLM이 필요 시 호출
        result["verify_result"] = {"passed": True, "issues": [], "suggestions": []}
        return result

    async def _generate_final(self, state: CodingState) -> dict[str, Any]:
        """2단계: STRONG 모델이 write_file 도구로 파일을 직접 저장한다.

        DeepAgents 패턴: LLM이 write_file 도구를 호출하면 MCP 서버가
        즉시 디스크에 저장. 마크다운 파싱 불필요.
        """
        # execute 단계의 stall 카운트를 리셋 — generate는 별개 단계
        self._stall_detector.reset()

        system_prompt = self._build_execute_system_prompt(
            state, purpose="generate"
        )

        # 프로젝트 컨텍스트 축적분 주입
        project_context = state.get("project_context", [])
        if project_context:
            system_prompt += "\n\n## 수집된 프로젝트 파일 컨텍스트\n"
            system_prompt += "\n".join(project_context[:10])

        # Planner가 지정한 생성 예정 파일 목록 — 루프 내부에서 동적 체크리스트로 주입
        # (시스템 프롬프트에 고정하면 진척도가 반영되지 않음)
        planned_files = state.get("planned_files", [])

        # FileManifest 주입 — 파일 간 인터페이스 계약
        file_manifest = state.get("file_manifest")
        if file_manifest:
            system_prompt += (
                "\n\n## 파일 매니페스트 (반드시 준수)\n"
                + self._format_manifest(file_manifest)
            )

        context_msg = self._build_context_message(state)

        # 모델 선택: 문서 위주 Phase면 DEFAULT(비용 절감), 코드 Phase면 STRONG
        planned = state.get("planned_files", [])
        doc_exts = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
        if planned:
            doc_count = sum(1 for f in planned if any(f.endswith(ext) for ext in doc_exts))
            is_docs_phase = doc_count > len(planned) * 0.5
        else:
            is_docs_phase = False

        # 모델 선택: 문서 Phase → DEFAULT(비용 절감), 코드 Phase → STRONG
        if is_docs_phase:
            gen_model = self._coding_config.get_model("verification")  # DEFAULT 티어
        else:
            gen_model = self._get_gen_model()  # STRONG 티어

        # MCP 도구 바인딩: 파일 I/O + 셸 실행 (LLM이 직접 테스트/설치 수행)
        language = self._detect_language_from_files(state)
        _GENERATE_TOOL_NAMES = {
            "write_file", "read_file",
            "run_shell", "run_python", "list_directory",
        }
        # Python 프로젝트에서만 validate_consistency 제공 (TS/JS에서는 무의미)
        if "python" in language.lower():
            _GENERATE_TOOL_NAMES.add("validate_consistency")
        write_tools = [
            t for t in self._tools
            if getattr(t, "name", "") in _GENERATE_TOOL_NAMES
        ]
        if write_tools:
            gen_model = gen_model.bind_tools(write_tools)
        else:
            logging.getLogger(__name__).error(
                "[generate_final] write_file 도구 없음 — MCP 도구 %d개 중 매칭 0개. "
                "코드 생성이 텍스트 전용으로 실행됩니다.",
                len(self._tools),
            )

        from coding_agent.core.middleware import ModelRequest as MWRequest
        from langchain_core.messages import ToolMessage

        # 메시지 정리는 MessageWindowMiddleware(토큰 기반 다단계 컴팩션)에 위임.
        # 에러 메시지 우선 보존 + 오래된 도구 결과 자동 정리가 미들웨어에서 처리됨.
        sanitized = sanitize_messages_for_llm(state["messages"])
        sanitized.append(context_msg)

        log = list(state.get("execution_log", []))
        written_files: list[str] = []
        total_tool_calls = 0
        tools_by_name = _get_tools_by_name(write_tools)
        loop_messages = list(sanitized)
        response = None
        responded_ids: set[str] = set()

        for loop_idx in range(self._GENERATE_MAX_TOOL_LOOPS):
            mw_request = MWRequest(
                system_message=system_prompt,
                messages=loop_messages,
                state=dict(state),
                model_name="generation",
                metadata={
                    "purpose": "generation",
                    "request_timeout": self._coding_config.get_request_timeout("generation"),
                },
            )

            try:
                mw_response = await self._middleware_chain.invoke(
                    mw_request, gen_model
                )
                response = mw_response.message
            except Exception as e:
                error_msg = str(e)
                logging.getLogger(__name__).warning(
                    "[generate_final] LLM 호출 실패 (루프 %d): %s",
                    loop_idx, error_msg[:200],
                )
                log.append(f"[generate_final] LLM 오류: {error_msg[:100]}")
                break

            # 도구 호출이 없으면 루프 종료
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            # 도구 호출 실행
            # DashScope 호환: AIMessage의 tool_calls args를 JSON 문자열로 정규화
            # LangChain 내부 형식(args: dict)이 DashScope 재전송 시
            # arguments(JSON string) 변환에 실패하는 문제 방지
            sanitized_response = self._sanitize_ai_tool_calls(response)
            # write_file의 content 인자를 제거 — 파일은 디스크에 저장되었으므로
            # 대화 히스토리에 전체 내용을 유지할 필요 없음.
            # LLM이 필요하면 read_file로 읽을 수 있음.
            sanitized_response = self._strip_write_file_content(sanitized_response)
            loop_messages.append(sanitized_response)
            for call in tool_calls:
                call_id = tc_id(call) or f"call_{tc_name(call)}_{loop_idx}"
                call_name = tc_name(call) or "unknown"
                call_args = tc_args(call)

                if call_name in tools_by_name:
                    result = await _execute_tool_safely(
                        tools_by_name[call_name], call_args
                    )
                    # write_file 성공 결과에서 파일 경로 수집 (정규화)
                    if call_name == "write_file" and result.startswith("OK:"):
                        filepath = result.split("OK:")[1].split("(")[0].strip()
                        workspace = os.environ.get(
                            "CODE_TOOLS_WORKSPACE", os.getcwd()
                        )
                        filepath = _normalize_file_path(filepath, workspace)
                        if filepath and filepath not in written_files:
                            written_files.append(filepath)
                else:
                    result = f"알 수 없는 도구: {call_name}"

                loop_messages.append(
                    ToolMessage(
                        content=result, tool_call_id=call_id, name=call_name
                    )
                )
                responded_ids.add(call_id)

            total_tool_calls += len(tool_calls)

            # 진척도 체크리스트 동적 주입 — 매 iteration마다 업데이트
            # 3종 에이전트 패턴: 대화 히스토리로 진척도를 파악하되,
            # 명시적 체크리스트로 남은 작업을 안내
            if planned_files and loop_idx > 0:
                written_set = set(written_files)
                done = [f for f in planned_files if f in written_set]
                remaining = [f for f in planned_files if f not in written_set]
                if remaining:
                    checklist_parts = [
                        f"[시스템] 진척 현황: {len(done)}/{len(planned_files)}개 파일 완료."
                    ]
                    if done:
                        checklist_parts.append(f"완료: {', '.join(done)}")
                    checklist_parts.append(
                        f"남은 파일: {', '.join(remaining)}. "
                        f"이 파일들만 write_file로 생성하세요."
                    )
                    loop_messages.append(
                        SystemMessage(content=" ".join(checklist_parts))
                    )

            # StallDetector: write_file 루프에서도 반복 패턴 체크
            stall_break = False
            for call in tool_calls:
                stall_action = self._stall_detector.record_and_check(
                    tc_name(call), tc_args(call)
                )
                if stall_action == StallAction.WARN:
                    log.append(
                        "[stall] 진전 없음 감지 — 다른 접근을 시도하세요"
                    )
                    loop_messages.append(
                        SystemMessage(
                            content="[시스템] 진전 없음이 감지되었습니다. "
                            "다른 접근 방식을 시도하세요."
                        )
                    )
                elif stall_action == StallAction.FORCE_EXIT:
                    summary = self._stall_detector.get_stall_summary()
                    log.append(f"[stall] FORCE_EXIT: {summary}")
                    # LLM 요약은 _execute_tools 레벨에서 처리되므로 여기서는 플래그만 설정
                    stall_break = True
                    break
            if stall_break:
                break

            log.append(
                f"[generate_final] 루프 {loop_idx}: "
                f"도구 {len(tool_calls)}회 호출, "
                f"파일 {len(written_files)}개 저장"
            )

            # 이미 생성된 파일 목록을 LLM에 알려서 중복 방지
            if written_files and loop_idx > 0:
                files_summary = ", ".join(written_files)
                loop_messages.append(
                    SystemMessage(
                        content=f"[시스템] 이미 저장된 파일({len(written_files)}개): {files_summary}. "
                        f"이미 저장된 파일은 다시 write_file하지 마세요. "
                        f"아직 생성하지 않은 파일만 작성하세요."
                    )
                )

        # 미완성 도구 호출 처리 (PatchToolCalls 패턴)
        if response and hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                cid = tc_id(call)
                if cid and cid not in responded_ids:
                    loop_messages.append(
                        ToolMessage(
                            content="도구 호출이 루프 한도 초과로 취소되었습니다.",
                            tool_call_id=cid,
                            name=tc_name(call) or "unknown",
                        )
                    )

        content = (response.content or "") if response else ""
        log.append(
            f"[generate_final] model=STRONG, "
            f"write_file {len(written_files)}개 파일 저장 완료"
        )

        return {
            "generated_code": content,
            "written_files": written_files,
            "execution_log": log,
            "tool_call_count": state.get("tool_call_count", 0) + total_tool_calls,
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
        # 도구 실행 전 abort 체크
        self._abort_controller.check_or_raise()

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
                # LLM에게 상황 요약 요청 — 다음 Phase에 전달
                stall_context = await self._summarize_stall(state, summary)
                return {
                    "messages": stall_messages,
                    "exit_reason": "stall_detected",
                    "stall_context": stall_context,
                }

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

        # 도구 실행 후 abort 체크
        self._abort_controller.check_or_raise()

        return {
            "messages": tool_messages,
            "project_context": context_entries,
            "tool_call_count": current_count + len(tool_calls),
        }

    # ── 실행 기반 검증 ─────────────────────────────────────

    # ── 메시지 직렬화 호환성 ──────────────────────────────────

    @staticmethod
    def _sanitize_ai_tool_calls(ai_msg: Any) -> Any:
        """AIMessage의 tool_calls를 LLM 공급자 호환 형식으로 정규화한다.

        LangChain AIMessage는 tool_calls.args를 dict로 저장하지만,
        DashScope/vLLM 등 일부 공급자는 재전송 시 arguments가
        JSON 문자열이어야 한다. 이 메서드는 dict args를 JSON 문자열로
        변환한 additional_kwargs를 보장한다.
        """
        from langchain_core.messages import AIMessage

        if not isinstance(ai_msg, AIMessage):
            return ai_msg

        tool_calls = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            return ai_msg

        # additional_kwargs.tool_calls가 있으면 arguments를 JSON 문자열로 보장
        ak = getattr(ai_msg, "additional_kwargs", {}) or {}
        ak_tool_calls = ak.get("tool_calls", [])
        if ak_tool_calls:
            fixed = False
            for tc in ak_tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args, ensure_ascii=False)
                    fixed = True
            if fixed:
                new_ak = dict(ak)
                new_ak["tool_calls"] = ak_tool_calls
                return AIMessage(
                    content=ai_msg.content or "",
                    tool_calls=tool_calls,
                    additional_kwargs=new_ak,
                    id=getattr(ai_msg, "id", None),
                )

        return ai_msg

    @staticmethod
    def _strip_write_file_content(ai_msg: Any) -> Any:
        """AIMessage의 write_file tool_call에서 content 인자를 제거한다.

        파일은 이미 디스크에 저장되었으므로 대화 히스토리에 전체 내용을
        유지할 필요가 없다. LLM이 내용을 다시 봐야 하면 read_file을 사용.
        이로써 토큰 사용량을 대폭 절감하여 MessageWindowMiddleware가
        대화 히스토리를 잘라내는 문제를 방지한다.
        """
        from langchain_core.messages import AIMessage

        if not isinstance(ai_msg, AIMessage):
            return ai_msg

        tool_calls = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            return ai_msg

        modified = False
        new_tool_calls = []
        for tc in tool_calls:
            name = tc.get("name", "")
            if name == "write_file" and "args" in tc:
                args = tc["args"]
                if isinstance(args, dict) and "content" in args:
                    content = args["content"]
                    line_count = content.count("\n") + 1 if content else 0
                    new_args = dict(args)
                    new_args["content"] = f"(디스크에 저장됨, {line_count}줄 — 필요 시 read_file로 확인)"
                    new_tool_calls.append({**tc, "args": new_args})
                    modified = True
                    continue
            new_tool_calls.append(tc)

        if not modified:
            return ai_msg

        # additional_kwargs의 tool_calls도 동기화
        ak = dict(getattr(ai_msg, "additional_kwargs", {}) or {})
        ak_tool_calls = ak.get("tool_calls", [])
        if ak_tool_calls:
            new_ak_tcs = []
            for tc in ak_tool_calls:
                fn = tc.get("function", {})
                if fn.get("name") == "write_file":
                    args_raw = fn.get("arguments", "")
                    try:
                        args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        if isinstance(args_dict, dict) and "content" in args_dict:
                            content = args_dict["content"]
                            line_count = content.count("\n") + 1 if content else 0
                            args_dict["content"] = f"(디스크에 저장됨, {line_count}줄 — 필요 시 read_file로 확인)"
                            fn["arguments"] = json.dumps(args_dict, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        pass
                new_ak_tcs.append(tc)
            ak["tool_calls"] = new_ak_tcs

        return AIMessage(
            content=ai_msg.content or "",
            tool_calls=new_tool_calls,
            additional_kwargs=ak,
            id=getattr(ai_msg, "id", None),
        )

    # ── Python 캐시 정리 ──────────────────────────────────────

    async def _clean_python_cache(self, workspace: str, log: list[str]) -> None:
        """workspace 내 __pycache__, .pyc, .pytest_cache를 삭제한다.

        수정된 .py 파일이 캐시된 .pyc로 인해 반영되지 않는 문제를 방지한다.
        재시도 시에도 매번 호출되어 이전 빌드 아티팩트를 제거한다.
        """
        removed = 0
        for root, dirs, files in os.walk(workspace, topdown=True):
            # .venv, node_modules, .git 내부는 건너뜀
            dirs[:] = [
                d for d in dirs
                if d not in (".venv", "node_modules", ".git", ".tox")
            ]
            # __pycache__ 디렉토리 삭제
            for d in list(dirs):
                if d in ("__pycache__", ".pytest_cache"):
                    target = os.path.join(root, d)
                    try:
                        import shutil
                        shutil.rmtree(target)
                        removed += 1
                        dirs.remove(d)
                    except OSError:
                        pass
            # 개별 .pyc 파일 삭제
            for f in files:
                if f.endswith(".pyc"):
                    try:
                        os.remove(os.path.join(root, f))
                        removed += 1
                    except OSError:
                        pass
        if removed:
            log.append(f"[cache-clean] Python 캐시 {removed}건 삭제")

    # (환경 설정 메서드 제거됨 — LLM이 run_shell 도구로 직접 수행)

    # ── 실행 기반 검증 ──

    async def _run_tests(self, state: CodingState) -> dict[str, Any]:
        """디스크에 저장된 코드의 기계적 검증만 수행한다.

        원칙: Harness = 기계적 도구 제공자, LLM = 판단자.
        - Harness가 하는 것: syntax check (py_compile), planned 파일 존재 확인
        - LLM이 하는 것: 환경 설정, 의존성 설치, 테스트 실행 (run_shell 도구 사용)

        환경 설정(venv, pip, npm), 런타임 감지, 테스트 실행(pytest, jest)은
        LLM이 _generate_final에서 run_shell 도구로 직접 수행한다.
        """
        written_files = state.get("written_files", [])
        log = list(state.get("execution_log", []))
        workspace = os.environ.get("CODE_TOOLS_WORKSPACE", os.getcwd())

        # ── planned vs written 비교 검증 (디스크 존재 확인) ──
        planned_files = state.get("planned_files", [])
        if planned_files:
            missing = []
            for f in planned_files:
                norm = _normalize_file_path(f, workspace)
                full_path = os.path.join(workspace, norm)
                if not os.path.exists(full_path):
                    missing.append(f)
            if missing:
                missing_str = ", ".join(missing)
                log.append(
                    f"[run_tests] 계획 대비 누락 파일 {len(missing)}개: {missing_str}"
                )
                return {
                    "test_passed": False,
                    "test_output": (
                        f"계획된 파일 중 {len(missing)}개가 디스크에 존재하지 않습니다: {missing_str}\n"
                        "누락된 파일을 write_file로 생성하세요."
                    ),
                    "execution_log": log,
                    "iteration": state.get("iteration", 0) + 1,
                }
            log.append(
                f"[run_tests] 계획 파일 전수 확인 완료 ({len(planned_files)}개)"
            )

        if not written_files:
            log.append("[run_tests] 저장된 파일 없음 — 스킵")
            return {
                "test_passed": True,
                "test_output": "",
                "execution_log": log,
                "iteration": state.get("iteration", 0) + 1,
            }

        # 파일 경로 정규화
        real_files = [_normalize_file_path(f, workspace) for f in written_files]
        py_files = [f for f in real_files if f.endswith(".py")]

        # Python 캐시 정리 (기계적 작업)
        if py_files:
            await self._clean_python_cache(workspace, log)

        errors: list[str] = []

        # syntax check (py_compile) — 기계적 검증
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

        # ── 프론트엔드 필수 파일 검증 (기계적 체크) ──
        js_ts_files = [f for f in real_files if f.endswith((".js", ".jsx", ".ts", ".tsx"))]
        if js_ts_files:
            fe_errors: list[str] = []
            # package.json이 있는 디렉토리 탐지
            pkg_dirs: set[str] = set()
            for f in real_files:
                if os.path.basename(f) == "package.json":
                    pkg_dirs.add(os.path.dirname(f) or ".")
            for pkg_dir in pkg_dirs:
                pkg_path = os.path.join(workspace, pkg_dir, "package.json")
                if not os.path.exists(pkg_path):
                    continue
                # tsconfig.json 존재 확인 (TS 파일이 있을 때)
                ts_in_dir = [f for f in js_ts_files if f.endswith((".ts", ".tsx")) and f.startswith(pkg_dir)]
                if ts_in_dir:
                    tsconfig = os.path.join(workspace, pkg_dir, "tsconfig.json")
                    if not os.path.exists(tsconfig):
                        fe_errors.append(f"[frontend] {pkg_dir}/tsconfig.json 누락 — TypeScript 프로젝트에 필수")
                # public/index.html 존재 확인 (react-scripts 사용 시)
                try:
                    with open(pkg_path, "r") as pf:
                        pkg_content = pf.read()
                    if "react-scripts" in pkg_content:
                        index_html = os.path.join(workspace, pkg_dir, "public", "index.html")
                        if not os.path.exists(index_html):
                            fe_errors.append(f"[frontend] {pkg_dir}/public/index.html 누락 — CRA(react-scripts) 빌드에 필수")
                        # 엔트리포인트 확인
                        index_tsx = os.path.join(workspace, pkg_dir, "src", "index.tsx")
                        index_jsx = os.path.join(workspace, pkg_dir, "src", "index.jsx")
                        if not os.path.exists(index_tsx) and not os.path.exists(index_jsx):
                            fe_errors.append(f"[frontend] {pkg_dir}/src/index.tsx 누락 — CRA 엔트리포인트 필수 (main.tsx가 아닌 index.tsx)")
                except (OSError, ValueError):
                    pass
            if fe_errors:
                output = "\n".join(fe_errors)
                log.append(f"[run_tests] 프론트엔드 필수 파일 누락 {len(fe_errors)}건")
                return {
                    "test_passed": False,
                    "test_output": output,
                    "execution_log": log,
                    "iteration": state.get("iteration", 0) + 1,
                }

        # 테스트 통과 + 이전에 실패가 있었으면 패턴 저장
        if state.get("iteration", 0) > 0:
            self._save_fix_pattern(state)

        log.append(f"[run_tests] syntax 검증 통과 ({len(py_files)} py, {len(real_files) - len(py_files)} other)")
        return {
            "test_passed": True,
            "test_output": "syntax 검증 통과",
            "execution_log": log,
            "iteration": state.get("iteration", 0) + 1,
        }

    # (HTML/프론트엔드 검증 메서드 제거됨 — LLM이 run_shell 도구로 직접 수행)

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

    # ── 라우팅 ──────────────────────────────────────────────

    def _should_retry_tests(self, state: CodingState) -> str:
        """실행 기반 테스트 실패 시 수정 루프를 결정한다.

        테스트 통과 → END
        테스트 실패 + exit_reason 설정됨 → END (안전장치 발동, 재시도 무의미)
        테스트 실패 + iteration 남음 → GENERATE_FINAL (에러 메시지가 messages에 주입됨)
        테스트 실패 + iteration 소진 → END (최선의 결과로 종료)
        """
        test_passed = state.get("test_passed", True)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if test_passed:
            return END

        # StallDetector FORCE_EXIT 등 안전장치 발동 시 재시도하지 않고 종료
        exit_reason = state.get("exit_reason", "")
        if exit_reason:
            logging.getLogger(__name__).warning(
                "[run_tests] exit_reason=%s — 재시도 없이 종료", exit_reason
            )
            return END

        if iteration >= max_iterations:
            logging.getLogger(__name__).warning("[run_tests] 최대 반복 도달 — 테스트 실패 상태로 종료")
            self._save_failure_pattern(state)
            return END

        # 실패 시: INJECT_ERROR 노드에서 에러 원문을 messages에 주입 → GENERATE
        return self.get_node_name("INJECT_ERROR")

    # ── 학습 패턴 저장 (Procedural Memory) ──────────────────────

    def _save_fix_pattern(self, state: CodingState) -> None:
        """테스트 실패→성공 사이클에서 학습된 수정 패턴을 Procedural Memory에 저장한다."""
        if not self._memory_store:
            return

        prev_test_output = state.get("_prev_test_output", "")
        if not prev_test_output:
            return

        parse_result = state.get("parse_result", {})
        task_type = parse_result.get("task_type", "generate")
        language = parse_result.get("language", "python")
        iteration = state.get("iteration", 0)

        # 에러 원문 요약 (regex 분류 대신)
        error_summary = prev_test_output[:200].strip()
        generated_code = state.get("generated_code", "")
        code_snippet = generated_code[:500] if generated_code else ""

        description = (
            f"[FIX] {language}/{task_type} — "
            f"에러: {error_summary} | "
            f"시도: {iteration}회 | "
            f"수정 코드 패턴: {code_snippet[:200]}"
        )

        try:
            result = self._memory_store.accumulate_skill(
                code=code_snippet,
                description=description,
                tags=[language, task_type, "test_fix"],
            )
            if result:
                logging.getLogger(__name__).info(
                    "[memory] 수정 패턴 저장: %s", result.id[:8]
                )
        except Exception as e:
            logging.getLogger(__name__).debug("[memory] 패턴 저장 실패 (무시): %s", e)

    def _save_failure_pattern(self, state: CodingState) -> None:
        """최대 반복 도달 시 실패 패턴을 Procedural Memory에 저장한다."""
        if not self._memory_store:
            return

        test_output = state.get("test_output", "")
        if not test_output:
            return

        parse_result = state.get("parse_result", {})
        task_type = parse_result.get("task_type", "generate")
        language = parse_result.get("language", "python")
        iteration = state.get("iteration", 0)

        description = (
            f"[FAIL] {language}/{task_type} — "
            f"최대 반복({iteration}회) 도달 후 미해결. "
            f"에러: {test_output[:200].strip()} | "
            f"교훈: 코드 재생성만으로 해결 불가. "
            f"근본적 접근 변경 필요."
        )

        try:
            self._memory_store.accumulate_skill(
                code="",
                description=description,
                tags=[language, task_type, "test_failure", "anti_pattern"],
            )
            logging.getLogger(__name__).info(
                "[memory] 실패 패턴 저장 (iteration=%d)", iteration
            )
        except Exception as e:
            logging.getLogger(__name__).debug("[memory] 실패 패턴 저장 실패 (무시): %s", e)

    # (에러 분류/정적 검증/테스트 출력 절단 메서드 제거됨
    #  — LLM이 에러 원문을 직접 분석, validate_consistency MCP 도구로 일관성 검증 가능)

    async def _summarize_stall(self, state: CodingState, stall_summary: str) -> str:
        """StallDetector 강제 종료 시 LLM에게 상황 요약을 요청한다.

        요약 결과는 다음 Phase에 전달되어 동일 문제 반복을 방지한다.
        LLM 호출 실패 시 기계적 요약으로 폴백한다.
        """
        test_output = state.get("test_output", "")
        written_files = list(state.get("written_files", []))
        iteration = state.get("iteration", 0)

        try:
            model = self._coding_config.get_model("default")
            from langchain_core.messages import HumanMessage as HM, SystemMessage as SM

            response = await model.ainvoke([
                SM(content=(
                    "당신은 코딩 에이전트의 디버깅 상태를 분석하는 전문가입니다. "
                    "에이전트가 반복 루프에 빠져 강제 종료되었습니다. "
                    "다음 Phase에 전달할 상황 요약을 작성하세요.\n\n"
                    "## 요약 포함 사항\n"
                    "1. 어떤 작업을 시도하고 있었는가\n"
                    "2. 왜 반복 루프에 빠졌는가 (근본 원인)\n"
                    "3. 어떤 에러가 해결되지 않았는가\n"
                    "4. 다음 Phase에서 주의해야 할 사항\n\n"
                    "200자 이내로 간결하게 작성하세요."
                )),
                HM(content=(
                    f"## StallDetector 요약\n{stall_summary}\n\n"
                    f"## 반복 횟수\n{iteration}회\n\n"
                    f"## 생성된 파일\n{', '.join(written_files[:20]) or '없음'}\n\n"
                    f"## 마지막 테스트 출력\n```\n{test_output[:1500]}\n```"
                )),
            ])
            return response.content[:500]
        except Exception as e:
            logger.warning("StallDetector LLM 요약 실패 (폴백): %s", e)
            return (
                f"강제 종료: {stall_summary}. "
                f"반복 {iteration}회, 파일 {len(written_files)}개 생성. "
                f"마지막 에러: {test_output[:200]}"
            )

    async def _inject_error(self, state: CodingState) -> dict[str, Any]:
        """단순화된 에러 주입: 에러 원문만 LLM에게 전달한다.

        핵심 원칙: Harness = 기계적 도구 제공자, LLM = 판단자.
        에러 분류/힌트/가이던스를 제거하고, LLM이 직접 에러를 분석하게 한다.
        """
        test_output = state.get("test_output", "")
        log = list(state.get("execution_log", []))
        iteration = state.get("iteration", 0)
        workspace = os.environ.get("CODE_TOOLS_WORKSPACE", os.getcwd())

        # Harness가 하는 것: 기계적 작업만
        # 1. Python 캐시 정리
        await self._clean_python_cache(workspace, log)

        # 2. traceback에서 관련 파일 추출 → 실제 내용 수집 (기계적 작업)
        error_files: dict[str, str] = {}
        for tb_match in re.finditer(r'File "([^"]+)"', test_output):
            fpath = tb_match.group(1)
            if workspace in fpath and ".venv" not in fpath and "site-packages" not in fpath:
                rel = os.path.relpath(fpath, workspace)
                abs_path = os.path.join(workspace, rel)
                if os.path.exists(abs_path) and rel not in error_files:
                    try:
                        content = open(abs_path, encoding="utf-8").read()
                        if len(content) > 3000:
                            content = content[:3000] + "\n... (truncated)"
                        error_files[rel] = content
                    except Exception:
                        pass

        # ImportError/NameError에서 참조되는 모듈 파일도 수집
        for imp_match in re.finditer(
            r"(?:cannot import name ['\"](\w+)['\"] from ['\"]([^'\"]+)['\"]"
            r"|No module named ['\"]([^'\"]+)['\"])",
            test_output,
        ):
            module_path = imp_match.group(2) or imp_match.group(3) or ""
            if module_path:
                # 모듈 경로를 파일 경로로 변환 (backend.routes → backend/routes.py)
                file_path = module_path.replace(".", "/") + ".py"
                abs_path = os.path.join(workspace, file_path)
                if os.path.exists(abs_path) and file_path not in error_files:
                    try:
                        content = open(abs_path, encoding="utf-8").read()
                        if len(content) > 3000:
                            content = content[:3000] + "\n... (truncated)"
                        error_files[file_path] = content
                    except Exception:
                        pass

        # 3. 에러 원문 + 관련 파일 내용 전달 — LLM이 직접 판단
        file_context = ""
        if error_files:
            file_context = "\n\n## 에러에 관련된 파일 내용\n"
            file_context += "아래 파일에서 에러가 발생했습니다. **불일치 부분만 수정**하세요.\n"
            for rel_path, content in list(error_files.items())[:5]:
                file_context += f"\n### {rel_path}\n```python\n{content}\n```\n"

        error_msg = HumanMessage(content=(
            f"테스트가 실패했습니다 (시도 {iteration}회). "
            f"에러를 분석하고 **해당 파일만** 수정하세요.\n\n"
            f"### 에러 출력\n"
            f"```\n{test_output[:3000]}\n```\n"
            f"{file_context}\n"
            f"**전체 파일을 새로 만들지 말고, 에러가 발생한 파일만 read_file로 읽고 "
            f"문제가 되는 부분을 수정하여 write_file로 저장하세요.**"
        ))

        log.append(f"[inject_error] 에러 원문 주입 (iteration={iteration})")

        return {
            "messages": [error_msg],
            "execution_log": log,
            "written_files": [],
            "_prev_generated_code": state.get("generated_code", ""),
            "generated_code": "",  # 리셋: _generate_code 반복 감지가 LLM 호출 전에 발동하지 않도록
            "_prev_test_output": test_output,
        }

    # ── 그래프 구성 ─────────────────────────────────────────

    def init_nodes(self, graph: StateGraph) -> None:
        # 단순화된 4-노드 그래프 (v2)
        # Harness = 기계적 도구 제공자, LLM = 판단자
        graph.add_node(self.get_node_name("RETRIEVE_MEMORY"), self._retrieve_memory)
        graph.add_node(self.get_node_name("GENERATE"), self._generate_code)
        graph.add_node(self.get_node_name("RUN_TESTS"), self._run_tests)
        graph.add_node(self.get_node_name("INJECT_ERROR"), self._inject_error)

    def init_edges(self, graph: StateGraph) -> None:
        # 진입점 → 메모리 검색 → 코드 생성 → 테스트 → (종료 or 에러 주입 → 재생성)
        graph.set_entry_point(self.get_node_name("RETRIEVE_MEMORY"))
        graph.add_edge(
            self.get_node_name("RETRIEVE_MEMORY"),
            self.get_node_name("GENERATE"),
        )
        graph.add_edge(
            self.get_node_name("GENERATE"),
            self.get_node_name("RUN_TESTS"),
        )
        graph.add_conditional_edges(
            self.get_node_name("RUN_TESTS"),
            self._should_retry_tests,
        )
        graph.add_edge(
            self.get_node_name("INJECT_ERROR"),
            self.get_node_name("GENERATE"),
        )
