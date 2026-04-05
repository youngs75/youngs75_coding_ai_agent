"""Rich 기반 CLI 출력 렌더러.

Claude Code UX 패턴 참조:
- 스피너를 사용한 진행 상태 표시
- 도구 호출 시 이름+인자 축약 표시
- 색상 체계: 성공(녹색), 경고(노랑), 에러(빨강), 시스템(dim)
- 코드 블록 구문 강조 (Live Markdown 렌더링)
- 소요시간 표시
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


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
        # Live Markdown 스트리밍용 버퍼
        self._token_buffer: str = ""
        self._stream_live: Live | None = None

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

    # ── 구분선 ──

    def divider(self, title: str = "") -> None:
        """얇은 구분선을 출력한다."""
        if title:
            self.console.print(Rule(title, style="dim"))
        else:
            self.console.print(Rule(style="dim"))

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
        """스피너 및 스트림 라이브를 정지한다."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._stream_live is not None:
            self._stream_live.stop()
            self._stream_live = None

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
        self.console.print(f"  [{_CLR_TOOL}]⚡[/] [{_CLR_DIM}]{tool_name}[/]{args_str}")

    def tool_result(self, tool_name: str, success: bool = True) -> None:
        """도구 실행 결과를 표시한다."""
        icon = f"[{_CLR_SUCCESS}]✓[/]" if success else f"[{_CLR_ERROR}]✗[/]"
        self.console.print(f"  {icon} [{_CLR_DIM}]{tool_name} 완료[/]")

    # ── 스킬/에이전트 표시 ──

    def skill_activated(self, skill_names: list[str]) -> None:
        """자동 활성화된 스킬을 표시한다."""
        names = ", ".join(skill_names)
        self.console.print(f"  [{_CLR_TOOL}]⚙[/] [{_CLR_DIM}]스킬 활성화:[/] {names}")

    def model_label(self, stage: str, model: str) -> None:
        """사용 중인 모델/단계를 표시한다."""
        self.console.print(f"  [{_CLR_DIM}]▸ {stage}:[/] {model}")

    def subagent_delegate(self, agent_name: str) -> None:
        """서브에이전트 위임을 표시한다."""
        self.console.print(
            f"  [{_CLR_TOOL}]⇢[/] [{_CLR_DIM}]위임:[/] [{_CLR_AGENT}]{agent_name}[/]"
        )

    # ── 토큰 스트리밍 (Live Markdown 렌더링) ──

    def start_token_stream(self) -> None:
        """토큰 스트리밍 시작 — 스피너 정지 + Live Markdown 프리뷰."""
        self._stop_progress()
        self._token_buffer = ""
        self.console.print(f"\n[{_CLR_AGENT}]◆ Agent[/]")
        self._stream_live = Live(
            Text("▍", style="dim green"),
            console=self.console,
            transient=True,  # 프리뷰는 종료 시 제거, 최종 Markdown으로 교체
            refresh_per_second=8,
        )
        self._stream_live.start()

    def render_token(self, token: str) -> None:
        """토큰 단위 Live Markdown 렌더링."""
        self._token_buffer += token
        if self._stream_live is not None:
            try:
                self._stream_live.update(Markdown(self._token_buffer + " ▍"))
            except Exception:
                # Markdown 파싱 실패 시 일반 텍스트로 표시
                self._stream_live.update(Text(self._token_buffer + " ▍"))

    def flush_tokens(self) -> None:
        """토큰 스트리밍 완료 — Live 프리뷰 제거 후 최종 Markdown 렌더링."""
        if self._stream_live is not None:
            self._stream_live.stop()
            self._stream_live = None
        if self._token_buffer:
            self.console.print(Markdown(self._token_buffer))
            self.console.print()
        self._token_buffer = ""

    # ── 파일 저장 결과 ──

    def files_written(self, files: list[str]) -> None:
        """디스크에 저장된 파일 목록을 표시한다."""
        if not files:
            return
        self.console.print(f"\n  [{_CLR_SUCCESS}]📁 저장된 파일:[/]")
        for f in files:
            self.console.print(f"    [{_CLR_SUCCESS}]✓[/] [{_CLR_DIM}]{f}[/]")

    # ── 검증 결과 ──

    def verify_result(
        self,
        passed: bool,
        issues: list[str] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """검증 결과를 시각적으로 표시한다."""
        if passed:
            self.console.print(f"\n  [{_CLR_SUCCESS}]✓ 검증 통과[/]")
        else:
            self.console.print(f"\n  [{_CLR_ERROR}]✗ 검증 실패[/]")
            if issues:
                for issue in issues[:5]:
                    self.console.print(f"    [{_CLR_DIM}]→ {issue}[/]")
        if suggestions:
            self.console.print(f"  [{_CLR_MUTED}]💡 제안 {len(suggestions)}건[/]")

    # ── 턴 타이밍 ──

    def start_turn(self) -> None:
        """턴 시작 시간 기록."""
        self._turn_start = time.monotonic()

    def end_turn(self) -> None:
        """턴 종료 — 소요시간 표시."""
        if self._turn_start > 0:
            elapsed = time.monotonic() - self._turn_start
            self.console.print(f"  [{_CLR_MUTED}]⏱ {elapsed:.1f}s[/]")
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
