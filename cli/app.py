"""대화형 CLI 메인 애플리케이션.

prompt-toolkit + rich 기반 대화형 루프.
에이전트 선택, 스트리밍 출력, 슬래시 커맨드를 지원한다.
"""

from __future__ import annotations

import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console

from youngs75_a2a.cli.commands import handle_command
from youngs75_a2a.cli.config import CLIConfig
from youngs75_a2a.cli.renderer import CLIRenderer
from youngs75_a2a.cli.session import CLISession


async def _run_agent_turn(user_input: str, session: CLISession, renderer: CLIRenderer) -> None:
    """에이전트에 메시지를 전달하고 응답을 출력한다.

    현재는 에이전트 그래프와의 통합 준비 단계로,
    실제 LLM 호출은 에이전트 인스턴스 연결 후 활성화된다.
    """
    session.add_message("user", user_input)

    # 에이전트 그래프 연동 (추후 Phase에서 활성화)
    # agent = _get_or_create_agent(session.info.agent_name)
    # async for event in agent.graph.astream({"messages": [HumanMessage(content=user_input)]}):
    #     ...

    response = f"[{session.info.agent_name}] 에이전트가 연결되면 여기에 응답이 표시됩니다."
    renderer.agent_message(response)
    session.add_message("assistant", response)


async def _main_loop(config: CLIConfig) -> None:
    """대화형 메인 루프."""
    console = Console()
    renderer = CLIRenderer(console)
    session = CLISession(agent_name=config.default_agent)

    prompt_session: PromptSession[str] = PromptSession(
        history=FileHistory(config.history_file),
    )

    renderer.welcome(session.info.agent_name)

    while True:
        try:
            user_input = await asyncio.to_thread(
                prompt_session.prompt,
                f"[{session.info.agent_name}] > ",
            )
        except (EOFError, KeyboardInterrupt):
            renderer.system_message("\n세션을 종료합니다.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # 슬래시 커맨드 처리
        result = handle_command(user_input, session, renderer)
        if result.should_quit:
            renderer.system_message(result.message)
            break
        if result.handled:
            continue

        # 에이전트 메시지 처리
        await _run_agent_turn(user_input, session, renderer)


def run_cli(config: CLIConfig | None = None) -> None:
    """CLI 진입점."""
    config = config or CLIConfig()
    asyncio.run(_main_loop(config))
