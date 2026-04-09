"""Orchestrator 에이전트.

사용자 입력을 분석하여 적합한 하위 에이전트로 라우팅하고,
A2A 프로토콜로 위임한 결과를 반환한다.

흐름:
    START → [CLASSIFY] ─→ [DELEGATE]    → [RESPOND] → END
                        └→ [COORDINATE] ↗

    복합 작업 감지 시 coordinator 모드로 병렬 워커 오케스트레이션.

사용 예:
    config = OrchestratorConfig(agent_endpoints=[...])
    agent = OrchestratorAgent(config=config)
    result = await agent.graph.ainvoke(
        {"messages": [HumanMessage("오늘 AI 뉴스 알려줘")]},
        config={"configurable": config.to_langgraph_configurable()},
    )
"""

from __future__ import annotations

import logging
import re as _re
import uuid
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt

from coding_agent.core.abort_controller import AbortController
from coding_agent.core.base_agent import BaseGraphAgent
from coding_agent.core.context_manager import ContextManager
from coding_agent.core.exceptions import SubAgentError
from coding_agent.core.middleware import (
    MemoryMiddleware,
    MiddlewareChain,
    ModelRequest as MWRequest,
    ResilienceMiddleware,
)
from coding_agent.core.subagents.registry import SubAgentRegistry
from .config import OrchestratorConfig
from .coordinator import CoordinatorMode
from .schemas import OrchestratorState

if TYPE_CHECKING:
    from coding_agent.core.subagents.process_manager import SubAgentProcessManager

# 오케스트레이터용 컨텍스트 매니저 (서브에이전트 호출 시 히스토리 필터링)
_orchestrator_context_manager = ContextManager()

# 모듈 레벨 미들웨어/중단 제어기 (OrchestratorAgent.__init__에서 초기화)
_abort_controller: AbortController | None = None
_middleware_chain: MiddlewareChain | None = None

logger = logging.getLogger(__name__)

CLASSIFY_SYSTEM_PROMPT = """\
당신은 사용자의 요청을 분석하여 가장 적합한 에이전트를 선택하는 라우터입니다.

사용 가능한 에이전트:
{agent_descriptions}

규칙:
1. 사용자의 요청 의도를 파악하고, 위 목록에서 가장 적합한 에이전트 이름을 정확히 하나만 선택하세요.
2. 에이전트 이름만 출력하세요. 다른 텍스트는 포함하지 마세요.
3. 어떤 에이전트에도 맞지 않으면 "none"을 출력하세요.
4. 코드 생성/개발 요청은 항상 coding_assistant를 선택하세요.
   프론트엔드+백엔드, 여러 파일, 풀스택 등 하나의 프로젝트를 구현하는 요청은 모두 코딩 작업입니다.
5. "coordinate"는 서로 다른 종류의 에이전트가 순차적으로 필요한 경우에만 사용하세요.
   예시: "기술을 조사한 뒤 그 결과로 코드를 작성해줘" (research → coding 순차 협업)
"""

# 복잡 요청 감지 키워드 패턴 (multi-file, fullstack, multi-step 등)
_COMPLEX_PATTERNS = _re.compile(
    r"(?:풀스택|fullstack|full[\-\s]stack|프론트엔드.*백엔드|백엔드.*프론트엔드"
    r"|여러\s*파일|multi[\-\s]*file|multi[\-\s]*step|multi[\-\s]*phase"
    r"|프로젝트\s*(?:생성|만들|구축|개발)|(?:앱|애플리케이션|서비스)\s*(?:만들|개발|구축)"
    r"|(?:CRUD|REST\s*API|GraphQL)\s*(?:서버|백엔드|API)"
    r"|(?:로그인|인증|회원가입).*(?:페이지|화면|UI)"
    r"|칸반|대시보드|관리\s*페이지|어드민"
    r"|(?:3|4|5|6|7|8|9|10)\s*개\s*(?:이상의?\s*)?(?:파일|페이지|컴포넌트|모듈))",
    _re.IGNORECASE,
)


def _detect_complexity(user_message: str) -> bool:
    """사용자 메시지에서 복잡한 코딩 요청인지 판단한다.

    복잡 요청 기준:
    - multi-file, fullstack, multi-step 키워드
    - 프론트엔드+백엔드 조합 요청
    - 프로젝트 수준의 생성/구축 요청
    - 메시지 길이가 충분히 긴 경우 (상세 요구사항)

    Args:
        user_message: 사용자의 원본 메시지

    Returns:
        True면 Planner를 경유해야 하는 복잡 요청
    """
    if _COMPLEX_PATTERNS.search(user_message):
        return True

    # 긴 메시지(500자 이상)는 상세한 요구사항일 가능성이 높음
    if len(user_message) > 500:
        return True

    return False


async def classify(state: OrchestratorState, config: RunnableConfig) -> dict:
    """사용자 입력을 분석하여 적합한 에이전트를 선택한다.

    SLM(FAST) 티어로 분류하여 비용/지연 최적화.
    ResilienceMiddleware로 재시도 + abort 체크포인트 보호.
    """
    # 턴 시작: abort 상태 리셋
    if _abort_controller:
        _abort_controller.reset()

    oc = OrchestratorConfig.from_runnable_config(config)
    llm = oc.get_model("parsing")  # FAST 티어 — 분류는 SLM으로 충분

    agent_descriptions = oc.get_agent_descriptions()
    system_prompt = CLASSIFY_SYSTEM_PROMPT.format(agent_descriptions=agent_descriptions)

    # 마지막 사용자 메시지 추출
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 미들웨어 체인으로 LLM 호출 (abort 체크 + 재시도 + 메모리 주입)
    if _middleware_chain:
        mw_request = MWRequest(
            system_message=system_prompt,
            messages=[HumanMessage(content=user_message)],
            metadata={
                "purpose": "classification",
                "request_timeout": oc.get_request_timeout("parsing"),
            },
        )
        mw_response = await _middleware_chain.invoke(mw_request, llm)
        response = mw_response.message
    else:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )

    selected = response.content.strip().lower()

    # 코디네이터 모드 감지
    if selected == "coordinate" or selected == "__coordinator__":
        logger.info(f"라우팅 결정: '{user_message[:50]}...' → __coordinator__")
        return {"selected_agent": "__coordinator__"}

    # 등록된 에이전트 이름과 매칭
    agent_names = [ep.name.lower() for ep in oc.agent_endpoints]
    if selected not in agent_names:
        # 부분 매칭 시도
        for name in agent_names:
            if name in selected or selected in name:
                selected = name
                break
        else:
            selected = "none"

    # 복잡도 판단: multi-file, fullstack, multi-step 키워드 감지
    is_complex = _detect_complexity(user_message)

    logger.info(
        "라우팅 결정: '%s...' → %s (complex=%s)",
        user_message[:50], selected, is_complex,
    )
    return {"selected_agent": selected, "is_complex": is_complex}


async def _invoke_planner(user_message: str) -> tuple[dict | None, str | None]:
    """Planner Agent를 호출하여 구현 계획을 생성한다.

    Returns:
        (task_plan_structured, plan_text): 구조화된 TaskPlan dict와 마크다운 텍스트

    Raises:
        AbortError: 중단 신호가 발생한 경우.
    """
    try:
        # SubAgent 호출 전 abort 체크
        if _abort_controller:
            _abort_controller.check_or_raise()

        from coding_agent.agents.planner.agent import PlannerAgent
        from coding_agent.agents.planner.config import PlannerConfig

        planner = await PlannerAgent.create(config=PlannerConfig())
        result = await planner.graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_message)],
                "user_request": user_message,
            }
        )

        # SubAgent 호출 후 abort 체크
        if _abort_controller:
            _abort_controller.check_or_raise()

        return result.get("task_plan"), result.get("plan_text", "")
    except SubAgentError:
        raise
    except Exception as e:
        logger.warning("Planner 호출 실패, 계획 없이 진행: %s", e)
        return None, None


def _get_process_manager() -> "SubAgentProcessManager":
    """싱글턴 SubAgentProcessManager를 반환한다."""
    global _process_manager
    if _process_manager is None:
        from coding_agent.core.subagents.process_manager import SubAgentProcessManager
        _process_manager = SubAgentProcessManager(
            registry=_local_registry,
            timeout_s=300.0,
        )
    return _process_manager


_process_manager: "SubAgentProcessManager | None" = None
_local_registry = SubAgentRegistry()


async def _invoke_local_agent(
    agent_name: str,
    user_message: str,
    *,
    task_plan: str | None = None,
) -> str | None:
    """로컬 에이전트를 별도 프로세스로 spawn하여 호출한다.

    Claude Code 스타일 동적 SubAgent: 프로세스 생성 → 작업 → 소멸.
    """
    # 에이전트 타입 정규화
    agent_type = agent_name
    if agent_name in ("coder",):
        agent_type = "coding_assistant"
    elif agent_name in ("researcher",):
        agent_type = "deep_research"
    elif agent_name in ("react",):
        agent_type = "simple_react"

    try:
        # SubAgent 호출 전 abort 체크
        if _abort_controller:
            _abort_controller.check_or_raise()

        manager = _get_process_manager()
        result = await manager.spawn_and_wait(
            agent_type=agent_type,
            task_message=user_message,
            task_plan=task_plan,
            timeout_s=300.0,
        )

        # SubAgent 호출 후 abort 체크
        if _abort_controller:
            _abort_controller.check_or_raise()

        if result.success and result.result:
            output = result.result
            if result.written_files:
                output += "\n\n📁 저장된 파일:\n" + "\n".join(
                    f"  • {f}" for f in result.written_files
                )
            return output

        if result.error:
            raise SubAgentError(
                agent_id=agent_type,
                message=result.error[:200],
            )
        return result.result  # partial 결과라도 반환

    except (SubAgentError, Exception) as e:
        if isinstance(e, SubAgentError):
            logger.warning("SubAgent 실패 (%s): %s", agent_type, e)
        else:
            logger.warning("SubAgent 프로세스 spawn 실패 '%s': %s", agent_name, e)
    return None


async def delegate(state: OrchestratorState, config: RunnableConfig) -> dict:
    """선택된 에이전트에 요청을 위임한다.

    우선순위:
    1. A2A 프로토콜 (HTTP 엔드포인트가 접근 가능한 경우)
    2. 로컬 에이전트 직접 호출 (폴백)
    """
    oc = OrchestratorConfig.from_runnable_config(config)
    selected = state.get("selected_agent", "none")

    if selected == "none":
        return {
            "agent_response": "죄송합니다. 현재 등록된 에이전트 중 적합한 것을 찾지 못했습니다. 질문을 다시 한번 구체적으로 말씀해 주세요."
        }

    # 서브에이전트용 히스토리 필터링 후 마지막 사용자 메시지 추출
    truncated_messages = _orchestrator_context_manager.truncate_for_subagent(
        state["messages"], last_n_turns=3
    )
    user_message = ""
    for msg in reversed(truncated_messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # A2A 프로토콜 시도
    url = oc.get_endpoint_url(selected)
    if url:
        try:
            from a2a.client import A2AClient
            from a2a.client.helpers import create_text_message_object
            from a2a.types import MessageSendParams, SendMessageRequest

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as hc:
                client = A2AClient(httpx_client=hc, url=url)
                msg = create_text_message_object(content=user_message)
                request = SendMessageRequest(
                    id=str(uuid.uuid4()),
                    params=MessageSendParams(message=msg),
                )
                response = await client.send_message(request)

            result = response.root
            if hasattr(result, "result"):
                obj = result.result
                if hasattr(obj, "artifacts") and obj.artifacts:
                    for artifact in obj.artifacts:
                        for part in artifact.parts or []:
                            root = getattr(part, "root", part)
                            if (
                                hasattr(root, "text")
                                and root.text
                                and len(root.text) > 5
                            ):
                                return {"agent_response": root.text}
                if hasattr(obj, "status"):
                    return {"agent_response": f"[에이전트 작업 상태: {obj.status}]"}

        except Exception as e:
            logger.warning(f"A2A 호출 실패 ({selected}), 로컬 폴백: {e}")

    # 로컬 에이전트 폴백 — 계획이 있으면 함께 전달
    task_plan = state.get("task_plan")
    task_plan_structured = state.get("task_plan_structured")

    # 멀티phase 순차 실행: 코딩 에이전트 + 구조화된 계획 + phase 2개 이상
    if selected in ("coding_assistant", "coder") and task_plan_structured:
        phases = task_plan_structured.get("phases", [])
        if len(phases) > 1:
            result = await _execute_phases_sequentially(
                user_message=user_message,
                task_plan=task_plan_structured,
                phases=phases,
            )
            return result

    # 단일 phase 또는 비코딩 에이전트: 기존 동작
    local_result = await _invoke_local_agent(
        selected, user_message, task_plan=task_plan
    )
    if local_result:
        return {"agent_response": local_result}

    return {
        "agent_response": f"에이전트 '{selected}'를 호출할 수 없습니다 (A2A 엔드포인트 미설정, 로컬 에이전트 미지원)."
    }


def _generate_prd(workspace: str, user_message: str, task_plan: dict) -> str | None:
    """task_plan 기반으로 PRD 마크다운 문서를 생성하여 저장한다."""
    import os

    try:
        parts = ["# PRD (Product Requirements Document)\n"]

        parts.append("## 프로젝트 개요\n")
        parts.append(user_message)
        parts.append("")

        if task_plan.get("summary"):
            parts.append("## 요약\n")
            parts.append(task_plan["summary"])
            parts.append("")

        if task_plan.get("architecture"):
            parts.append("## 아키텍처\n")
            parts.append(task_plan["architecture"])
            parts.append("")

        if task_plan.get("tech_stack"):
            parts.append("## 기술 스택\n")
            tech = task_plan["tech_stack"]
            if isinstance(tech, list):
                for item in tech:
                    parts.append(f"- {item}")
            elif isinstance(tech, dict):
                for k, v in tech.items():
                    parts.append(f"- **{k}**: {v}")
            else:
                parts.append(str(tech))
            parts.append("")

        if task_plan.get("constraints"):
            parts.append("## 제약 조건\n")
            constraints = task_plan["constraints"]
            if isinstance(constraints, list):
                for item in constraints:
                    parts.append(f"- {item}")
            else:
                parts.append(str(constraints))
            parts.append("")

        if task_plan.get("file_structure"):
            parts.append("## 파일 구조\n")
            fs = task_plan["file_structure"]
            if isinstance(fs, list):
                for item in fs:
                    parts.append(f"- {item}")
            elif isinstance(fs, dict):
                for k, v in fs.items():
                    parts.append(f"- **{k}**: {v}")
            else:
                parts.append(str(fs))
            parts.append("")

        doc_dir = os.path.join(workspace, "docs")
        os.makedirs(doc_dir, exist_ok=True)
        prd_path = os.path.join(doc_dir, "PRD.md")
        with open(prd_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        logger.info("PRD 문서 생성: %s", prd_path)
        return prd_path
    except Exception as e:
        logger.warning("PRD 문서 생성 실패: %s", e)
        return None


def _generate_sdd(workspace: str, task_plan: dict) -> str | None:
    """task_plan 기반으로 SDD(Software Design Document) 명세서를 생성하여 저장한다."""
    import os

    try:
        parts = ["# SDD (Software Design Document)\n"]

        if task_plan.get("summary"):
            parts.append("## 개요\n")
            parts.append(task_plan["summary"])
            parts.append("")

        if task_plan.get("tech_stack"):
            parts.append("## 기술 스택\n")
            tech = task_plan["tech_stack"]
            if isinstance(tech, list):
                for item in tech:
                    parts.append(f"- {item}")
            elif isinstance(tech, dict):
                for k, v in tech.items():
                    parts.append(f"- **{k}**: {v}")
            else:
                parts.append(str(tech))
            parts.append("")

        phases = task_plan.get("phases", [])
        if phases:
            parts.append("## Phase별 상세 명세\n")
            for phase in phases:
                phase_id = phase.get("id", "N/A")
                title = phase.get("title", "")
                parts.append(f"### {phase_id}: {title}\n")

                if phase.get("description"):
                    parts.append(f"**설명**: {phase['description']}\n")

                if phase.get("files"):
                    parts.append("**대상 파일**:")
                    for f in phase["files"]:
                        parts.append(f"- `{f}`")
                    parts.append("")

                if phase.get("depends_on"):
                    deps = phase["depends_on"]
                    parts.append(f"**의존성**: {', '.join(deps)}\n")

                if phase.get("instructions"):
                    parts.append("**구현 지시사항**:")
                    parts.append(phase["instructions"])
                    parts.append("")

        if task_plan.get("file_structure"):
            parts.append("## 전체 파일 구조\n")
            fs = task_plan["file_structure"]
            if isinstance(fs, list):
                for item in fs:
                    parts.append(f"- {item}")
            elif isinstance(fs, dict):
                for k, v in fs.items():
                    parts.append(f"- **{k}**: {v}")
            else:
                parts.append(str(fs))
            parts.append("")

        doc_dir = os.path.join(workspace, "docs")
        os.makedirs(doc_dir, exist_ok=True)
        sdd_path = os.path.join(doc_dir, "SDD.md")
        with open(sdd_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        logger.info("SDD 문서 생성: %s", sdd_path)
        return sdd_path
    except Exception as e:
        logger.warning("SDD 문서 생성 실패: %s", e)
        return None


async def _execute_phases_sequentially(
    user_message: str,
    task_plan: dict,
    phases: list[dict],
) -> dict:
    """계획의 phase를 순차 실행한다. 각 phase마다 CodingAssistant를 독립 호출한다.

    이전 phase 결과물은 파일 시스템에 저장되어 있고,
    다음 phase에는 파일 경로 목록만 전달하여 필요 시 MCP로 읽도록 안내한다.
    """
    import os as _os
    from pathlib import Path as _Path

    from coding_agent.agents.coding_assistant.agent import CodingAssistantAgent
    from coding_agent.agents.coding_assistant.config import CodingConfig
    from coding_agent.core.skills.loader import SkillLoader
    from coding_agent.core.skills.registry import SkillRegistry
    from coding_agent.core.subagent_context import SubagentContextFilter

    # 스킬 레지스트리 초기화 (_invoke_local_agent와 동일)
    skill_registry = SkillRegistry()
    skills_dir = _os.getenv(
        "SKILLS_DIR",
        str(_Path(__file__).resolve().parent.parent.parent / "data" / "skills"),
    )
    if _Path(skills_dir).is_dir():
        loader = SkillLoader(skills_dir)
        skill_registry = SkillRegistry(loader=loader)
        skill_registry.discover()

    plan_summary = task_plan.get("summary", "")
    architecture = task_plan.get("architecture", "")
    all_written_files: list[str] = []
    phase_results: list[dict] = []
    failed_ids: set[str] = set()

    # PRD/SDD 문서 자동 생성
    workspace = _os.getenv("CODE_TOOLS_WORKSPACE", _os.getcwd())
    prd_path = _generate_prd(workspace, user_message, task_plan)
    sdd_path = _generate_sdd(workspace, task_plan)
    if prd_path:
        all_written_files.append(f"{prd_path} (PRD)")
    if sdd_path:
        all_written_files.append(f"{sdd_path} (SDD)")

    for i, phase in enumerate(phases):
        phase_id = phase.get("id", f"phase_{i + 1}")
        title = phase.get("title", "")

        # depends_on 체크: 선행 phase가 실패 또는 스킵이면 skip
        deps = set(phase.get("depends_on", []))
        if deps & failed_ids:
            logger.warning("Phase %s 건너뜀: 선행 phase 실패/스킵 (%s)", phase_id, deps & failed_ids)
            phase_results.append({
                "phase_id": phase_id,
                "title": title,
                "status": "skipped",
                "written_files": [],
                "error": f"선행 phase 실패: {deps & failed_ids}",
            })
            failed_ids.add(phase_id)  # 스킵도 전파하여 후속 phase가 실행되지 않도록
            continue

        logger.info("Phase %d/%d 시작: %s — %s", i + 1, len(phases), phase_id, title)

        phase_message = SubagentContextFilter.build_phase_task_message(
            user_message=user_message,
            plan_summary=plan_summary,
            architecture=architecture,
            phase=phase,
            phase_index=i,
            total_phases=len(phases),
            prior_written_files=all_written_files,
        )

        try:
            # 마지막 phase가 통합 성격(파일 수 적음, instructions 짧음)이면 budget 완화
            is_final = i == len(phases) - 1 and len(phases) > 1
            phase_files = phase.get("files", [])
            phase_instructions_len = len(phase.get("instructions", ""))
            is_integration = is_final and len(phase_files) <= 2 and phase_instructions_len < 500
            config = CodingConfig(
                max_llm_calls_per_turn=20 if is_integration else 15,
                diminishing_streak_limit=5 if is_integration else 3,
                min_delta_tokens=300 if is_integration else 500,
            )
            agent = await CodingAssistantAgent.create(
                config=config,
                skill_registry=skill_registry,
            )
            _init_state = SubagentContextFilter.build_init_state(
                task_message=phase_message,
                max_iterations=11,
                env_approved=True,
                extra_state={"planned_files": phase.get("files", [])},
            )

            result = await agent.graph.ainvoke(
                _init_state
            )

            compacted = SubagentContextFilter.compact_result(result)
            written = compacted["written_files"]
            test_passed = compacted["test_passed"]
            phase_status = "success" if test_passed else "failed"

            # 파일 경로 기준 중복 제거 후 누적 (정규화 적용)
            from coding_agent.agents.coding_assistant.agent import (
                _normalize_file_path,
            )

            _ws = workspace
            existing_paths = {
                _normalize_file_path(
                    f.split(" (")[0].strip() if isinstance(f, str) else f.get("path", ""),
                    _ws,
                )
                for f in all_written_files
            }
            for f in written:
                fpath = f.get("path", "") if isinstance(f, dict) else str(f)
                norm = _normalize_file_path(fpath, _ws)
                if norm and norm not in existing_paths:
                    all_written_files.append(norm)
                    existing_paths.add(norm)

            unique_written = written

            phase_results.append({
                "phase_id": phase_id,
                "title": title,
                "status": phase_status,
                "written_files": unique_written,
            })

            if not test_passed:
                failed_ids.add(phase_id)
                logger.warning("Phase %s 테스트 실패 상태로 완료: %d개 파일", phase_id, len(written))
            else:
                logger.info("Phase %s 완료: %d개 파일 생성", phase_id, len(written))

        except Exception as e:
            logger.error("Phase %s 실패: %s", phase_id, e)
            failed_ids.add(phase_id)
            phase_results.append({
                "phase_id": phase_id,
                "title": title,
                "status": "failed",
                "written_files": [],
                "error": str(e),
            })

    # 전체 phase 완료 후 VerificationAgent 실행
    verification_result = None
    if all_written_files:
        try:
            from coding_agent.agents.verifier.agent import VerificationAgent
            from coding_agent.agents.verifier.config import VerifierConfig

            # 생성된 파일 확장자에서 주요 언어 감지
            detected_lang = _detect_language(all_written_files)
            verifier = await VerificationAgent.create(config=VerifierConfig())
            v_result = await verifier.graph.ainvoke({
                "code": "",  # 파일 기반 검증이므로 코드 본문 불필요
                "written_files": all_written_files,
                "language": detected_lang,
                "requirements": user_message,
            })
            verification_result = v_result.get("verification_result")
            logger.info("검증 완료: %s", verification_result.get("summary", ""))
        except Exception as e:
            logger.warning("VerificationAgent 실행 실패 (결과에 영향 없음): %s", e)

    # 집계 응답 생성
    response = _build_phase_summary_response(phase_results, all_written_files)
    return {
        "agent_response": response,
        "phase_results": phase_results,
        "verification_result": verification_result,
    }


def _detect_language(files: list[str]) -> str:
    """파일 확장자에서 주요 프로그래밍 언어를 감지한다."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript", ".vue": "javascript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".rb": "ruby",
        ".swift": "swift", ".kt": "kotlin", ".cs": "csharp",
    }
    counts: dict[str, int] = {}
    for f in files:
        for ext, lang in ext_map.items():
            if f.endswith(ext):
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return "python"
    return max(counts, key=counts.get)


def _build_phase_summary_response(
    phase_results: list[dict],
    all_written_files: list[str],
) -> str:
    """phase별 실행 결과를 집계하여 응답 텍스트를 생성한다."""
    parts = ["## Phase별 실행 결과\n"]
    for pr in phase_results:
        icon = {"success": "✓", "failed": "✗", "skipped": "⊘"}.get(pr["status"], "?")
        file_count = len(pr.get("written_files", []))
        line = f"- {icon} **{pr['phase_id']}**: {pr['title']}"
        if file_count:
            line += f" — {file_count}개 파일"
        if pr.get("error"):
            line += f" (오류: {pr['error'][:50]})"
        parts.append(line)

    success_count = sum(1 for pr in phase_results if pr["status"] == "success")
    parts.append(f"\n**총 {success_count}/{len(phase_results)} phase 완료**, {len(all_written_files)}개 파일 생성")

    if all_written_files:
        parts.append("\n**생성된 파일:**")
        for f in all_written_files:
            parts.append(f"  - {f}")

    return "\n".join(parts)


async def coordinate(state: OrchestratorState, config: RunnableConfig) -> dict:
    """복합 작업을 서브태스크로 분해하고 병렬 워커에 위임한다."""
    oc = OrchestratorConfig.from_runnable_config(config)
    llm = oc.get_model("default")

    # 사용자 메시지 추출
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # SubAgentRegistry 구성 — 등록된 에이전트 엔드포인트를 SubAgentSpec으로 변환
    registry = SubAgentRegistry()
    from coding_agent.core.subagents.schemas import SubAgentSpec

    for ep in oc.agent_endpoints:
        registry.register(
            SubAgentSpec(
                name=ep.name,
                description=ep.description,
                capabilities=[ep.name],
                endpoint=ep.url,
            )
        )

    coordinator = CoordinatorMode(
        registry=registry,
        context_manager=_orchestrator_context_manager,
    )

    try:
        # SubAgent 호출 전 abort 체크
        if _abort_controller:
            _abort_controller.check_or_raise()

        result = await coordinator.run(
            task=user_message,
            context=state["messages"],
            llm=llm,
        )

        # SubAgent 호출 후 abort 체크
        if _abort_controller:
            _abort_controller.check_or_raise()

        return {"agent_response": result["synthesized_response"]}
    except Exception as e:
        logger.error("코디네이터 실행 실패: %s", e)
        return {"agent_response": f"복합 작업 처리 중 오류가 발생했습니다: {e}"}


async def respond(state: OrchestratorState, config: RunnableConfig) -> dict:
    """에이전트 응답을 최종 메시지로 포맷팅한다."""
    selected = state.get("selected_agent", "unknown")
    response = state.get("agent_response", "")

    if selected and selected != "none":
        content = f"[{selected}] {response}"
    else:
        content = response

    return {"messages": [AIMessage(content=content)]}


async def plan(state: OrchestratorState, config: RunnableConfig) -> dict:
    """코딩 태스크에 대해 Planner Agent로 구현 계획을 수립한다.

    coding_assistant 위임 전에 실행되어 구조화된 계획을 생성한다.
    계획 수립 후 interrupt()로 사용자 승인을 대기한다 (Human-in-the-loop).
    비코딩 태스크는 패스스루 (계획 없이 바로 delegate).
    """
    selected = state.get("selected_agent", "none")

    # 코딩 에이전트 대상만 계획 수립
    if selected not in ("coding_assistant", "coder"):
        return {"task_plan": None}

    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    logger.info("Planner Agent 호출: '%s...'", user_message[:50])
    task_plan_structured, plan_text = await _invoke_planner(user_message)

    if plan_text:
        logger.info("계획 수립 완료 (%d chars)", len(plan_text))
        # Human-in-the-loop: 계획을 사용자에게 보여주고 승인 대기
        max_replan = 3  # 최대 재계획 횟수
        for attempt in range(max_replan + 1):
            response = interrupt(plan_text)

            # 승인: True 또는 truthy 단순값
            if response is True or response == True:  # noqa: E712
                logger.info("사용자가 계획을 승인함")
                return {"task_plan": plan_text, "task_plan_structured": task_plan_structured}

            # 거부 + 피드백: {"approved": False, "feedback": "..."}
            if isinstance(response, dict) and not response.get("approved", True):
                feedback = response.get("feedback", "")
                if feedback and attempt < max_replan:
                    logger.info("재계획 요청 (attempt %d): %s", attempt + 1, feedback)
                    # 피드백을 반영하여 재계획
                    replan_message = (
                        f"{user_message}\n\n"
                        f"## 이전 계획에 대한 사용자 피드백\n"
                        f"사용자가 다음 사항을 수정 요청했습니다:\n{feedback}\n\n"
                        f"위 피드백을 반영하여 계획을 다시 수립하세요."
                    )
                    task_plan_structured, plan_text = await _invoke_planner(replan_message)
                    if not plan_text:
                        break
                    continue  # 새 계획으로 다시 interrupt

            # 단순 거부 (피드백 없음) 또는 재계획 한도 초과
            logger.info("사용자가 계획을 거부함")
            return {"task_plan": None, "task_plan_structured": None}

    logger.info("계획 수립 스킵 (planner 미응답)")
    return {"task_plan": None, "task_plan_structured": None}


def _route_after_classify(state: OrchestratorState) -> str:
    """classify 노드 이후 라우팅 결정.

    복합 작업(coordinate)이면 coordinate 노드로,
    복잡한 코딩 작업(is_complex)이면 plan 노드로 (계획 수립 후 delegate),
    단순 코딩 작업은 delegate로 직행 (Planner 건너뜀).
    """
    selected = state.get("selected_agent", "")
    if selected == "__coordinator__":
        return "coordinate"
    if selected in ("coding_assistant", "coder") and state.get("is_complex"):
        return "plan"
    return "delegate"


class OrchestratorAgent(BaseGraphAgent):
    """사용자 요청을 분석하고 적합한 에이전트로 라우팅하는 오케스트레이터.

    단일 에이전트 위임(delegate)과 복합 작업 병렬 오케스트레이션(coordinate)을 지원.
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "CLASSIFY": "classify",
        "PLAN": "plan",
        "DELEGATE": "delegate",
        "COORDINATE": "coordinate",
        "RESPOND": "respond",
    }

    def __init__(
        self,
        *,
        config: OrchestratorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        global _abort_controller, _middleware_chain

        self._orch_config = config or OrchestratorConfig()

        # AbortController — 턴 시작 시 reset
        _abort_controller = AbortController()

        # 미들웨어 체인: ResilienceMiddleware(가장 바깥) + MemoryMiddleware
        _middleware_chain = MiddlewareChain([
            ResilienceMiddleware(abort_controller=_abort_controller),
            MemoryMiddleware(),
        ])

        super().__init__(
            config=self._orch_config,
            state_schema=OrchestratorState,
            agent_name="OrchestratorAgent",
            **kwargs,
        )

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("CLASSIFY"), classify)
        graph.add_node(self.get_node_name("PLAN"), plan)
        graph.add_node(self.get_node_name("DELEGATE"), delegate)
        graph.add_node(self.get_node_name("COORDINATE"), coordinate)
        graph.add_node(self.get_node_name("RESPOND"), respond)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("CLASSIFY"))
        # classify 이후 조건부 라우팅: plan(코딩) / delegate(기타) / coordinate(복합)
        graph.add_conditional_edges(
            self.get_node_name("CLASSIFY"),
            _route_after_classify,
            {
                "plan": self.get_node_name("PLAN"),
                "delegate": self.get_node_name("DELEGATE"),
                "coordinate": self.get_node_name("COORDINATE"),
            },
        )
        # plan → delegate (계획 수립 후 코딩 에이전트에 위임)
        graph.add_edge(self.get_node_name("PLAN"), self.get_node_name("DELEGATE"))
        graph.add_edge(self.get_node_name("DELEGATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("COORDINATE"), self.get_node_name("RESPOND"))
        graph.add_edge(self.get_node_name("RESPOND"), END)
