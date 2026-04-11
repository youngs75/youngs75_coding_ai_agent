"""Subagent 컨텍스트 격리 유틸리티.

DeepAgents의 _EXCLUDED_STATE_KEYS 패턴을 적용하여,
서브에이전트에 필요한 최소 컨텍스트만 전달한다.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

# 서브에이전트에 전달하지 않는 부모 state 키
# 서브에이전트는 자체적으로 스킬 활성화, 메모리 검색, 프로젝트 파일 읽기를 수행한다.
_EXCLUDED_STATE_KEYS: set[str] = {
    "messages",                  # 서브에이전트는 task description만 받음
    "skill_context",             # 서브에이전트가 자체 판단으로 스킬 활성화
    "semantic_context",          # 서브에이전트가 자체 메모리 검색
    "episodic_log",              # 서브에이전트 내부 로그
    "procedural_skills",         # 서브에이전트 내부
    "user_profile_context",      # 서브에이전트 내부
    "domain_knowledge_context",  # 서브에이전트 내부
    "project_context",           # 서브에이전트가 MCP read_file로 JIT 로드
    "parse_result",              # 서브에이전트 내부
    "verify_result",             # 서브에이전트 내부
    "generated_code",            # 서브에이전트 내부
    "execution_log",             # 서브에이전트 내부
    "test_output",               # 서브에이전트 내부
    "_prev_generated_code",      # 서브에이전트 내부
    "_prev_test_output",         # 서브에이전트 내부
}


def get_excluded_state_keys() -> frozenset[str]:
    """제외 대상 state 키 목록을 반환한다."""
    return frozenset(_EXCLUDED_STATE_KEYS)


class SubagentContextFilter:
    """서브에이전트 호출 시 컨텍스트를 최소화하는 유틸리티."""

    @staticmethod
    def build_init_state(
        task_message: str,
        *,
        max_iterations: int = 11,
        env_approved: bool = False,
        extra_state: dict | None = None,
    ) -> dict:
        """서브에이전트 초기 state를 생성한다.

        부모의 전체 state 대신, task description HumanMessage만 포함하는
        최소 state를 구성한다.
        """
        state: dict = {
            "messages": [HumanMessage(content=task_message)],
            "iteration": 0,
            "max_iterations": max_iterations,
        }

        if env_approved:
            state["env_approved"] = True

        # extra_state가 있으면, 제외 키를 필터링한 뒤 병합
        if extra_state:
            for key, value in extra_state.items():
                if key not in _EXCLUDED_STATE_KEYS:
                    state[key] = value

        return state

    @staticmethod
    def compact_result(result: dict) -> dict:
        """서브에이전트 실행 결과를 필수 필드만 추출하여 압축한다.

        전체 state 반환 대신 written_files, test_passed, exit_reason,
        code_summary만 포함하는 경량 dict를 반환한다.
        """
        # written_files: path 기준 중복 제거
        raw_files = result.get("written_files", [])
        seen_paths: set[str] = set()
        deduped_files: list[dict] = []
        for f in raw_files:
            path = f.get("path", "") if isinstance(f, dict) else str(f)
            if path and path not in seen_paths:
                seen_paths.add(path)
                deduped_files.append(f)

        # code_summary: generated_code 앞 500자
        generated_code = result.get("generated_code", "") or ""
        if len(generated_code) > 500:
            code_summary = generated_code[:500] + "..."
        else:
            code_summary = generated_code

        return {
            "written_files": deduped_files,
            "test_passed": result.get("test_passed", False),
            "exit_reason": result.get("exit_reason", "unknown"),
            "code_summary": code_summary,
        }

    @staticmethod
    def build_phase_task_message(
        user_message: str,
        plan_summary: str,
        architecture: str,
        phase: dict,
        phase_index: int,
        total_phases: int,
        prior_written_files: list[str],
        *,
        max_user_message_chars: int = 500,
        prior_stall_context: str = "",
        tech_stack: list[str] | None = None,
        constraints: list[str] | None = None,
        file_structure: list[str] | None = None,
    ) -> str:
        """멀티 페이즈 실행 시 서브에이전트에 전달할 태스크 메시지를 생성한다.

        파일 내용 대신 경로만 나열하여 메시지를 ~2-4KB로 유지한다.
        서브에이전트는 필요 시 read_file MCP 도구로 내용을 직접 읽는다.
        """
        # 사용자 메시지 자르기
        if len(user_message) > max_user_message_chars:
            user_message = user_message[:max_user_message_chars] + "..."

        parts: list[str] = [
            f"## Phase {phase_index + 1}/{total_phases}: {phase.get('title', 'Untitled')}",
            "",
            "### 사용자 요청",
            user_message,
            "",
            "### 전체 계획 요약",
            plan_summary,
            "",
            "### 아키텍처",
            architecture,
            "",
        ]

        # 기술 스택 — LLM이 올바른 패키지/프레임워크를 사용하도록 안내
        if tech_stack:
            parts.append("### 기술 스택")
            parts.append(", ".join(tech_stack))
            parts.append("")

        # 전체 파일 구조 — Phase 간 일관된 import 경로 보장
        if file_structure:
            parts.append("### 전체 파일 구조")
            for f in file_structure:
                parts.append(f"- `{f}`")
            parts.append("")

        # 제약 조건
        if constraints:
            parts.append("### 제약 조건")
            for c in constraints:
                parts.append(f"- {c}")
            parts.append("")

        parts.append("### 현재 페이즈 지시사항")

        # 페이즈 지시사항 추가
        instructions = phase.get("instructions", phase.get("description", ""))
        if isinstance(instructions, list):
            for item in instructions:
                parts.append(f"- {item}")
        else:
            parts.append(str(instructions))

        # 생성 예정 파일 체크리스트 (Planner가 지정한 파일 목록)
        planned_files = phase.get("files", [])
        if planned_files:
            parts.append("")
            parts.append("### 생성 필수 파일 체크리스트")
            parts.append("아래 파일을 **모두** `write_file` 도구로 생성해야 합니다:")
            for fpath in planned_files:
                parts.append(f"- [ ] `{fpath}`")

        # 이전 페이즈에서 생성된 파일 경로 나열
        if prior_written_files:
            parts.append("")
            parts.append("### 이전 페이즈 산출물 (경로만 — 필요 시 read_file로 읽기)")
            for fpath in prior_written_files:
                parts.append(f"- `{fpath}`")

        # 통합 리마인더 (첫 번째 페이즈 이후)
        if phase_index > 0:
            parts.append("")
            parts.append(
                "⚠️ 이전 페이즈 산출물과의 통합을 확인하세요. "
                "import 경로, 인터페이스 일관성, 타입 호환성을 검증한 뒤 작업하세요."
            )

        # 이전 Phase 강제 종료 컨텍스트 (StallDetector)
        if prior_stall_context:
            parts.append("")
            parts.append("### ⚠ 이전 Phase 강제 종료 알림")
            parts.append(
                "이전 Phase가 반복 루프로 인해 강제 종료되었습니다. "
                "아래 상황 요약을 참고하여 동일 문제를 반복하지 마세요:"
            )
            parts.append(f"> {prior_stall_context}")

        return "\n".join(parts)
