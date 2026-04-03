"""Rich 기반 CLI 출력 렌더러."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class CLIRenderer:
    """Rich 콘솔 기반 출력 렌더러."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def welcome(self, agent_name: str) -> None:
        """시작 배너 출력."""
        self.console.print(
            Panel(
                Text.from_markup(
                    f"[bold cyan]AI Coding Agent Harness[/]\n"
                    f"Active agent: [bold green]{agent_name}[/]\n"
                    f"Type [bold]/help[/] for commands, [bold]/quit[/] to exit"
                ),
                title="[bold]youngs75_a2a[/]",
                border_style="cyan",
            )
        )

    def user_message(self, content: str) -> None:
        """사용자 메시지 표시."""
        self.console.print(f"[bold blue]You:[/] {content}")

    def agent_message(self, content: str) -> None:
        """에이전트 응답 표시 (마크다운 렌더링)."""
        self.console.print("[bold green]Agent:[/]")
        self.console.print(Markdown(content))
        self.console.print()

    def system_message(self, content: str) -> None:
        """시스템 메시지 표시."""
        self.console.print(f"[dim]{content}[/]")

    def error(self, content: str) -> None:
        """에러 메시지 표시."""
        self.console.print(f"[bold red]Error:[/] {content}")

    def stream_token(self, token: str) -> None:
        """스트리밍 토큰 출력 (줄바꿈 없이)."""
        self.console.print(token, end="")

    def stream_end(self) -> None:
        """스트리밍 종료."""
        self.console.print()
