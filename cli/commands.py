"""CLI 슬래시 커맨드 처리.

Phase 10 통합: /permissions, /context, /tools 커맨드 추가.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from youngs75_a2a.cli.eval_runner import (
    format_eval_summary,
    format_remediation_summary,
    load_last_eval_results,
    load_last_remediation_report,
    run_evaluation_async,
)
from youngs75_a2a.core.tool_permissions import PermissionDecision

if TYPE_CHECKING:
    from youngs75_a2a.cli.renderer import CLIRenderer
    from youngs75_a2a.cli.session import CLISession


@dataclass
class CommandResult:
    """커맨드 실행 결과."""

    handled: bool = True
    should_quit: bool = False
    message: str = ""


AVAILABLE_AGENTS = [
    "coding_assistant",
    "deep_research",
    "simple_react",
    "orchestrator",
]


def handle_command(
    raw_input: str,
    session: CLISession,
    renderer: CLIRenderer,
) -> CommandResult:
    """슬래시 커맨드를 파싱하고 실행한다.

    Returns:
        CommandResult — handled=False이면 일반 메시지로 처리
    """
    if not raw_input.startswith("/"):
        return CommandResult(handled=False)

    parts = raw_input.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    match cmd:
        case "/quit" | "/exit" | "/q":
            return CommandResult(should_quit=True, message="세션을 종료합니다.")

        case "/help" | "/h":
            _show_help(renderer)
            return CommandResult()

        case "/agent":
            return _switch_agent(arg, session, renderer)

        case "/agents":
            _list_agents(renderer)
            return CommandResult()

        case "/clear":
            session.clear_history()
            renderer.system_message("대화 기록을 초기화했습니다.")
            return CommandResult()

        case "/session":
            _show_session(session, renderer)
            return CommandResult()

        case "/memory":
            _show_memory(session, renderer)
            return CommandResult()

        case "/skill":
            return _handle_skill(arg, session, renderer)

        case "/history":
            return _handle_history(arg, session, renderer)

        case "/eval":
            return _handle_eval(arg, renderer)

        case "/permissions":
            _show_permissions(session, renderer)
            return CommandResult()

        case "/context":
            _show_context(session, renderer)
            return CommandResult()

        case "/tools":
            _show_tools(session, renderer)
            return CommandResult()

        case _:
            renderer.error(f"알 수 없는 커맨드: {cmd}. /help를 입력하세요.")
            return CommandResult()


def _show_help(renderer: CLIRenderer) -> None:
    renderer.system_message(
        "사용 가능한 커맨드:\n"
        "  /help, /h          — 도움말\n"
        "  /agent <name>      — 에이전트 전환\n"
        "  /agents            — 사용 가능한 에이전트 목록\n"
        "  /skill list        — 등록된 스킬 목록\n"
        "  /skill activate <name> — 스킬 활성화 (L2 로드)\n"
        "  /history           — 최근 대화 기록 표시\n"
        "  /history clear     — 대화 기록 초기화\n"
        "  /eval              — 에이전트 평가 실행 (DeepEval)\n"
        "  /eval status       — 마지막 평가 결과 요약\n"
        "  /eval remediate    — Remediation 실행 (Loop 3)\n"
        "  /eval remediate status — 마지막 Remediation 결과\n"
        "  /permissions       — 현재 도구 권한 설정 표시\n"
        "  /context           — 로드된 프로젝트 컨텍스트 표시\n"
        "  /tools             — 도구 목록 + 권한 상태 표시\n"
        "  /clear             — 대화 기록 초기화\n"
        "  /session           — 현재 세션 정보\n"
        "  /memory            — 메모리 상태\n"
        "  /quit, /exit, /q   — 종료"
    )


def _switch_agent(
    name: str, session: CLISession, renderer: CLIRenderer
) -> CommandResult:
    name = name.strip()
    if not name:
        renderer.error("에이전트 이름을 지정하세요. 예: /agent coding_assistant")
        return CommandResult()
    if name not in AVAILABLE_AGENTS:
        renderer.error(f"알 수 없는 에이전트: {name}. /agents로 목록을 확인하세요.")
        return CommandResult()
    session.switch_agent(name)
    renderer.system_message(f"에이전트를 [{name}]으로 전환했습니다.")
    return CommandResult()


def _list_agents(renderer: CLIRenderer) -> None:
    lines = ["사용 가능한 에이전트:"]
    for name in AVAILABLE_AGENTS:
        lines.append(f"  - {name}")
    renderer.system_message("\n".join(lines))


def _show_session(session: CLISession, renderer: CLIRenderer) -> None:
    info = session.info
    renderer.system_message(
        f"세션 ID: {info.session_id}\n"
        f"시작: {info.started_at.isoformat()}\n"
        f"에이전트: {info.agent_name}\n"
        f"메시지 수: {info.message_count}"
    )


def _show_memory(session: CLISession, renderer: CLIRenderer) -> None:
    count = session.memory.total_count
    renderer.system_message(f"메모리 항목 수: {count}")


def _handle_skill(
    arg: str, session: CLISession, renderer: CLIRenderer
) -> CommandResult:
    """스킬 관련 커맨드를 처리한다."""
    sub_parts = arg.strip().split(maxsplit=1)
    sub = sub_parts[0].lower() if sub_parts else ""

    if sub == "list":
        skills = session.skills.list_skills()
        if not skills:
            renderer.system_message("등록된 스킬이 없습니다.")
        else:
            lines = ["등록된 스킬:"]
            for skill in skills:
                level = skill.loaded_level.value
                lines.append(
                    f"  - {skill.name}: {skill.metadata.description} [{level}]"
                )
            renderer.system_message("\n".join(lines))
        return CommandResult()

    if sub == "activate":
        name = sub_parts[1].strip() if len(sub_parts) > 1 else ""
        if not name:
            renderer.error("스킬 이름을 지정하세요. 예: /skill activate code_review")
            return CommandResult()
        activated = session.activate_skill(name)
        if activated:
            renderer.system_message(f"스킬 [{activated}] 활성화 완료")
        else:
            renderer.error(f"스킬을 찾을 수 없습니다: {name}")
        return CommandResult()

    renderer.error(
        "사용법:\n"
        "  /skill list            — 등록된 스킬 목록\n"
        "  /skill activate <name> — 스킬 활성화"
    )
    return CommandResult()


def _handle_history(
    arg: str, session: CLISession, renderer: CLIRenderer
) -> CommandResult:
    """대화 히스토리 커맨드를 처리한다."""
    sub = arg.strip().lower()

    if sub == "clear":
        session.clear_history()
        renderer.system_message("대화 기록을 초기화했습니다.")
        return CommandResult()

    # 기본: 최근 대화 기록 표시
    history = session.get_history_summary()
    if not history:
        renderer.system_message("대화 기록이 없습니다.")
        return CommandResult()

    lines = [f"최근 대화 기록 ({len(history)}건):"]
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if len(content) > 80:
            content = content[:77] + "..."
        prefix = "You" if role == "user" else "Agent"
        lines.append(f"  [{prefix}] {content}")
    renderer.system_message("\n".join(lines))
    return CommandResult()


# ── Phase 10 슬래시 커맨드: /permissions, /context, /tools ──


def _show_permissions(session: CLISession, renderer: CLIRenderer) -> None:
    """현재 도구 권한 설정을 표시한다."""
    mgr = session.permission_manager
    if mgr is None:
        renderer.system_message("도구 권한 관리자가 설정되지 않았습니다.")
        return

    lines = [
        f"도구 권한 설정 (workspace: {mgr.workspace})",
        "",
        "도구별 권한:",
    ]
    for tool_name, decision in sorted(mgr.DEFAULT_PERMISSIONS.items()):
        # 실제 적용 중인 권한 (프로젝트 설정/환경변수 오버라이드 포함)
        actual = mgr.check(tool_name)
        override = " (오버라이드)" if actual != decision else ""
        lines.append(f"  {tool_name}: {actual.value}{override}")

    # 거부 기록
    denials = mgr.denial_log
    if denials:
        lines.append("")
        lines.append(f"거부 기록 ({len(denials)}건):")
        for entry in denials[-5:]:
            lines.append(
                f"  [{entry['timestamp'][:19]}] {entry['tool_name']}: {entry['reason']}"
            )

    renderer.system_message("\n".join(lines))


def _show_context(session: CLISession, renderer: CLIRenderer) -> None:
    """로드된 프로젝트 컨텍스트를 표시한다."""
    ctx = session.project_context
    if not ctx:
        renderer.system_message("로드된 프로젝트 컨텍스트가 없습니다.")
        return

    # 긴 컨텍스트는 일부만 표시
    max_display = 500
    if len(ctx) > max_display:
        display = ctx[:max_display] + f"\n... ({len(ctx)}자 중 {max_display}자 표시)"
    else:
        display = ctx

    renderer.system_message(f"프로젝트 컨텍스트:\n{display}")


def _show_tools(session: CLISession, renderer: CLIRenderer) -> None:
    """도구 목록과 권한 상태를 표시한다."""
    mgr = session.permission_manager

    if mgr is None:
        renderer.system_message("도구 권한 관리자가 설정되지 않았습니다.")
        return

    lines = ["도구 목록 + 권한 상태:"]

    # 기본 등록된 도구의 권한 표시
    all_tools = sorted(mgr.DEFAULT_PERMISSIONS.keys())
    for tool_name in all_tools:
        decision = mgr.check(tool_name)
        # 권한에 따른 아이콘
        if decision == PermissionDecision.ALLOW:
            icon = "[허용]"
        elif decision == PermissionDecision.ASK:
            icon = "[확인필요]"
        else:
            icon = "[거부]"
        lines.append(f"  {tool_name}: {icon}")

    # 병렬 실행기 상태
    executor = session.tool_executor
    if executor:
        from youngs75_a2a.core.parallel_tool_executor import CONCURRENCY_SAFE_TOOLS

        lines.append("")
        lines.append("병렬 실행 가능 도구:")
        for tool_name in sorted(CONCURRENCY_SAFE_TOOLS):
            lines.append(f"  {tool_name}")
    else:
        lines.append("")
        lines.append("병렬 도구 실행기: 비활성")

    renderer.system_message("\n".join(lines))


def _handle_eval(arg: str, renderer: CLIRenderer) -> CommandResult:
    """평가 커맨드를 처리한다."""
    sub = arg.strip().lower()

    if sub == "status":
        return _eval_status(renderer)
    elif sub == "" or sub == "run":
        return _eval_run(renderer)
    elif sub == "remediate":
        return _eval_remediate(renderer)
    elif sub == "remediate status":
        return _eval_remediate_status(renderer)
    else:
        renderer.error(
            f"알 수 없는 eval 하위 커맨드: {sub}\n"
            "  /eval              — 평가 실행\n"
            "  /eval status       — 마지막 평가 결과 확인\n"
            "  /eval remediate    — Remediation 실행 (Loop 3)\n"
            "  /eval remediate status — 마지막 Remediation 결과 확인"
        )
        return CommandResult()


def _eval_run(renderer: CLIRenderer) -> CommandResult:
    """에이전트 평가를 실행한다."""
    import asyncio

    renderer.system_message("평가를 시작합니다... (시간이 걸릴 수 있습니다)")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        from youngs75_a2a.cli.eval_runner import _run_evaluation_sync

        result = _run_evaluation_sync()
    else:
        result = asyncio.run(run_evaluation_async())

    summary = format_eval_summary(result)
    if result.success:
        renderer.system_message(summary)
    else:
        renderer.error(summary)

    return CommandResult()


def _eval_status(renderer: CLIRenderer) -> CommandResult:
    """마지막 평가 결과를 표시한다."""
    result = load_last_eval_results()
    summary = format_eval_summary(result)

    if result.success:
        renderer.system_message(summary)
    else:
        renderer.error(summary)

    return CommandResult()


def _eval_remediate(renderer: CLIRenderer) -> CommandResult:
    """Remediation Agent를 실행한다 (Loop 3)."""
    import asyncio

    from youngs75_a2a.cli.eval_runner import run_remediation_async

    renderer.system_message(
        "Remediation Agent를 시작합니다... (시간이 걸릴 수 있습니다)"
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # 이미 이벤트 루프가 실행 중이면 동기적으로 처리
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(asyncio.run, run_remediation_async()).result()
    else:
        result = asyncio.run(run_remediation_async())

    summary = format_remediation_summary(result)
    if result.success:
        renderer.system_message(summary)
        if result.report_path:
            renderer.system_message(f"리포트 저장: {result.report_path}")

        # remediation 결과를 프롬프트에 자동 적용
        if result.report and hasattr(result.report, "get_prompt_changes"):
            changes = result.report.get_prompt_changes()
            if changes:
                try:
                    from youngs75_a2a.agents.coding_assistant.prompts import (
                        get_prompt_registry,
                    )

                    registry = get_prompt_registry()
                    updated = registry.apply_remediation(changes)
                    if updated:
                        renderer.system_message(
                            f"프롬프트 개선 적용: {', '.join(updated)}"
                        )
                except Exception:
                    pass  # 프롬프트 적용 실패는 무시
    else:
        renderer.error(summary)

    return CommandResult()


def _eval_remediate_status(renderer: CLIRenderer) -> CommandResult:
    """마지막 Remediation 결과를 표시한다."""
    result = load_last_remediation_report()
    summary = format_remediation_summary(result)

    if result.success:
        renderer.system_message(summary)
    else:
        renderer.error(summary)

    return CommandResult()
