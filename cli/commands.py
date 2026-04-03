"""CLI 슬래시 커맨드 처리."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from youngs75_a2a.cli.eval_runner import (
    format_eval_summary,
    load_last_eval_results,
    run_evaluation_async,
)

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

        case "/eval":
            return _handle_eval(arg, renderer)

        case _:
            renderer.error(f"알 수 없는 커맨드: {cmd}. /help를 입력하세요.")
            return CommandResult()


def _show_help(renderer: CLIRenderer) -> None:
    renderer.system_message(
        "사용 가능한 커맨드:\n"
        "  /help, /h          — 도움말\n"
        "  /agent <name>      — 에이전트 전환\n"
        "  /agents            — 사용 가능한 에이전트 목록\n"
        "  /eval              — 에이전트 평가 실행 (DeepEval)\n"
        "  /eval status       — 마지막 평가 결과 요약\n"
        "  /clear             — 대화 기록 초기화\n"
        "  /session           — 현재 세션 정보\n"
        "  /memory            — 메모리 상태\n"
        "  /quit, /exit, /q   — 종료"
    )


def _switch_agent(name: str, session: CLISession, renderer: CLIRenderer) -> CommandResult:
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


def _handle_eval(arg: str, renderer: CLIRenderer) -> CommandResult:
    """평가 커맨드를 처리한다."""
    sub = arg.strip().lower()

    if sub == "status":
        return _eval_status(renderer)
    elif sub == "" or sub == "run":
        return _eval_run(renderer)
    else:
        renderer.error(
            f"알 수 없는 eval 하위 커맨드: {sub}\n"
            "  /eval          — 평가 실행\n"
            "  /eval status   — 마지막 평가 결과 확인"
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
