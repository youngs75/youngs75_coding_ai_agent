"""Rich 기반 CLI 출력 렌더러.

Claude Code UX 패턴 참조:
- 스피너를 사용한 진행 상태 표시
- 도구 호출 시 이름+인자 축약 표시
- 색상 체계: 성공(녹색), 경고(노랑), 에러(빨강), 시스템(dim)
- 코드 블록 구문 강조
- 소요시간 표시
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.live import Live
from rich.table import Table


# 색상 상수
_CLR_BRAND = "bold cyan"
_CLR_AGENT = "bold green"
_CLR_USER = "bold blue"
_CLR_SYSTEM = "dim"
_CLR_TOOL = "bold yellow"
_CLR_ERROR = "bold red"
_CLR_SUCCESS = "bold green"
_CLR_WARN = "bold yellow"
_CLR_DIM = "dim"
_CLR_MUTED = "dim italic"


class CLIRenderer:
    """Rich 콘솔 기반 출력 렌더러."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._live: Live | None = None
        self._spinner_text: str = ""
        self._turn_start: float = 0.0

    # ── 배너 ──

    def welcome(self, agent_name: str) -> None:
        """시작 배너 출력."""
        table = Table.grid(padding=(0, 1))
        table.add_row(
            Text("◆", style="bold cyan"),
            Text("AI Coding Agent Harness", style="bold white"),
        )
        table.add_row(
            Text(" ", style=""),
            Text(f"Active: {agent_name}", style="green"),
        )
        table.add_row(
            Text(" ", style=""),
            Text("/help for commands · /quit to exit", style="dim"),
        )

        self.console.print(
            Panel(
                table,
                title="[bold cyan]youngs75_a2a[/]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
        self.console.print()

    # ── 메시지 ──

    def user_message(self, content: str) -> None:
        """사용자 메시지 표시."""
        self.console.print(f"[{_CLR_USER}]❯[/] {content}")

    def agent_message(self, content: str) -> None:
        """에이전트 응답 표시 (마크다운 렌더링)."""
        self.console.print(f"\n[{_CLR_AGENT}]◆ Agent[/]")
        self.console.print(Markdown(content))
        self.console.print()

    def system_message(self, content: str) -> None:
        """시스템 메시지 표시."""
        self.console.print(f"[{_CLR_SYSTEM}]{content}[/]")

    def error(self, content: str) -> None:
        """에러 메시지 표시."""
        self.console.print(f"[{_CLR_ERROR}]✗ Error:[/] {content}")

    def warning(self, content: str) -> None:
        """경고 메시지 표시."""
        self.console.print(f"[{_CLR_WARN}]⚠ {content}[/]")

    def success(self, content: str) -> None:
        """성공 메시지 표시."""
        self.console.print(f"[{_CLR_SUCCESS}]✓[/] {content}")

    # ── 진행 상태 (스피너) ──

    def start_progress(self, label: str) -> None:
        """스피너와 함께 진행 상태를 시작한다."""
        self._stop_progress()
        self._spinner_text = label
        spinner = Spinner("dots", text=Text(f" {label}", style=_CLR_DIM))
        self._live = Live(
            spinner,
            console=self.console,
            transient=True,
            refresh_per_second=10,
        )
        self._live.start()

    def update_progress(self, label: str) -> None:
        """진행 상태 텍스트를 업데이트한다."""
        self._spinner_text = label
        if self._live is not None:
            spinner = Spinner("dots", text=Text(f" {label}", style=_CLR_DIM))
            self._live.update(spinner)

    def _stop_progress(self) -> None:
        """스피너를 정지한다."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def stop_progress_with_result(self, label: str, success: bool = True) -> None:
        """스피너를 멈추고 결과를 표시한다."""
        self._stop_progress()
        if success:
            self.console.print(f"  [{_CLR_SUCCESS}]✓[/] [{_CLR_DIM}]{label}[/]")
        else:
            self.console.print(f"  [{_CLR_ERROR}]✗[/] [{_CLR_DIM}]{label}[/]")

    # ── 도구 호출 표시 ──

    def tool_call(self, tool_name: str, args: dict | None = None) -> None:
        """도구 호출을 축약 표시한다."""
        self._stop_progress()
        args_str = ""
        if args:
            # 핵심 인자만 축약 표시
            path = args.get("path", args.get("query", ""))
            if path:
                args_str = f" {_truncate(str(path), 50)}"
        self.console.print(
            f"  [{_CLR_TOOL}]⚡[/] [{_CLR_DIM}]{tool_name}[/]{args_str}"
        )

    def tool_result(self, tool_name: str, success: bool = True) -> None:
        """도구 실행 결과를 표시한다."""
        icon = f"[{_CLR_SUCCESS}]✓[/]" if success else f"[{_CLR_ERROR}]✗[/]"
        self.console.print(f"  {icon} [{_CLR_DIM}]{tool_name} 완료[/]")

    # ── 토큰 스트리밍 ──

    def start_token_stream(self) -> None:
        """토큰 스트리밍 시작 — 스피너 정지 + Agent 헤더."""
        self._stop_progress()
        self.console.print(f"\n[{_CLR_AGENT}]◆ Agent[/]")

    def render_token(self, token: str) -> None:
        """토큰 단위 실시간 출력."""
        self.console.print(token, end="", markup=False, highlight=False)

    def flush_tokens(self) -> None:
        """토큰 스트리밍 완료 — 줄바꿈."""
        self.console.print()
        self.console.print()

    # ── 검증 결과 ──

    def verify_result(self, passed: bool, issues: list[str] | None = None,
                      suggestions: list[str] | None = None) -> None:
        """검증 결과를 시각적으로 표시한다."""
        if passed:
            self.console.print(f"\n  [{_CLR_SUCCESS}]✓ 검증 통과[/]")
        else:
            self.console.print(f"\n  [{_CLR_ERROR}]✗ 검증 실패[/]")
            if issues:
                for issue in issues[:5]:
                    self.console.print(f"    [{_CLR_DIM}]→ {issue}[/]")
        if suggestions:
            self.console.print(
                f"  [{_CLR_MUTED}]💡 제안 {len(suggestions)}건[/]"
            )

    # ── 턴 타이밍 ──

    def start_turn(self) -> None:
        """턴 시작 시간 기록."""
        self._turn_start = time.monotonic()

    def end_turn(self) -> None:
        """턴 종료 — 소요시간 표시."""
        if self._turn_start > 0:
            elapsed = time.monotonic() - self._turn_start
            self.console.print(
                f"  [{_CLR_MUTED}]⏱ {elapsed:.1f}s[/]"
            )
            self._turn_start = 0.0

    # ── 레거시 호환 ──

    def stream_token(self, token: str) -> None:
        """스트리밍 토큰 출력 (레거시 호환)."""
        self.console.print(token, end="")

    def stream_end(self) -> None:
        """스트리밍 종료 (레거시 호환)."""
        self.console.print()


def _truncate(text: str, max_len: int = 50) -> str:
    """텍스트를 최대 길이로 잘라낸다."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
