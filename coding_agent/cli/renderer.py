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

    def welcome(self, agent_name: str, workspace: str | None = None) -> None:
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
        if workspace:
            table.add_row(
                Text(" ", style=""),
                Text(f"Workspace: {workspace}", style="bold yellow"),
            )
        table.add_row(
            Text(" ", style=""),
            Text("/help for commands · /quit to exit", style="dim"),
        )

        self.console.print(
            Panel(
                table,
                title="[bold cyan]coding_agent[/]",
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

    def complete_node(self, label: str, summary: str = "") -> None:
        """노드 완료를 영구 표시한다 (transient 스피너 대체)."""
        self._stop_progress()
        suffix = f" — {summary}" if summary else ""
        self.console.print(f"  [{_CLR_SUCCESS}]✓[/] [{_CLR_DIM}]{label}{suffix}[/]")

    # ── 도구 호출 표시 ──

    def tool_call(self, tool_name: str, args: dict | None = None) -> None:
        """도구 호출을 축약 표시한다.

        도구별 컨텍스트를 추출하여 사용자에게 유용한 정보를 표시한다.
        """
        self._stop_progress()
        label, detail = _describe_tool_call(tool_name, args)
        detail_str = f" {detail}" if detail else ""
        self.console.print(f"  [{_CLR_TOOL}]⚡[/] [{_CLR_DIM}]{label}[/]{detail_str}")

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

    # ── SubAgent 프로세스 수명주기 ──

    def subagent_spawn(self, agent_type: str, pid: int, task: str = "") -> None:
        """SubAgent 프로세스 spawn을 표시한다."""
        task_preview = _truncate(task, 60) if task else ""
        self.console.print(
            f"  [{_CLR_TOOL}]⚡[/] SubAgent spawn: "
            f"[{_CLR_AGENT}]{agent_type}[/] "
            f"[{_CLR_DIM}](pid={pid})[/]"
            + (f" [{_CLR_DIM}]{task_preview}[/]" if task_preview else "")
        )

    def subagent_running(self, agent_type: str, pid: int, elapsed_s: float, mem_mb: float = 0.0) -> None:
        """SubAgent 실행 상태를 표시한다."""
        mem_info = f", mem={mem_mb:.0f}MB" if mem_mb > 0 else ""
        self.console.print(
            f"  [{_CLR_DIM}]⠋[/] SubAgent 실행 중: "
            f"[{_CLR_AGENT}]{agent_type}[/] "
            f"[{_CLR_DIM}]({elapsed_s:.1f}s{mem_info})[/]"
        )

    def subagent_completed(self, agent_type: str, pid: int, duration_s: float) -> None:
        """SubAgent 정상 완료를 표시한다."""
        self.console.print(
            f"  [{_CLR_SUCCESS}]✓[/] SubAgent 완료: "
            f"[{_CLR_AGENT}]{agent_type}[/] "
            f"[{_CLR_DIM}](pid={pid}, {duration_s:.1f}s, exit=0)[/]"
        )

    def subagent_failed(self, agent_type: str, pid: int, error: str = "") -> None:
        """SubAgent 실패를 표시한다."""
        err_preview = _truncate(error, 80) if error else ""
        self.console.print(
            f"  [{_CLR_ERROR}]✗[/] SubAgent 실패: "
            f"[{_CLR_AGENT}]{agent_type}[/] "
            f"[{_CLR_DIM}](pid={pid})[/]"
            + (f" [{_CLR_ERROR}]{err_preview}[/]" if err_preview else "")
        )

    def subagent_destroyed(self, count: int = 1) -> None:
        """SubAgent 자원 회수를 표시한다."""
        self.console.print(
            f"  [{_CLR_DIM}]🗑 SubAgent 자원 회수 완료 ({count}건)[/]"
        )

    # ── 토큰 스트리밍 ──

    def start_token_stream(self) -> None:
        """토큰 스트리밍 시작 — 스피너 정지 + Agent 헤더."""
        self._stop_progress()
        self._token_buffer = ""
        self.console.print(f"\n[{_CLR_AGENT}]◆ Agent[/]")

    def render_token(self, token: str) -> None:
        """토큰 단위 실시간 출력 (raw text — 깜박임 없는 부드러운 스트리밍)."""
        self._token_buffer += token
        self.console.print(token, end="", markup=False, highlight=False)

    def flush_tokens(self) -> None:
        """토큰 스트리밍 완료 — 줄바꿈."""
        self.console.print()
        self.console.print()
        self._token_buffer = ""

    # ── 계획 표시 (Human-in-the-loop) ──

    def show_plan(self, plan_text: str) -> None:
        """Planner Agent가 생성한 계획을 사용자에게 표시한다."""
        self.console.print()
        self.console.print(
            Panel(
                Markdown(plan_text),
                title="[bold cyan]📋 구현 계획[/]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def ask_plan_approval(self) -> bool:
        """계획 승인 여부를 사용자에게 묻는다 (blocking)."""
        try:
            response = self.console.input(
                f"  [{_CLR_BRAND}]?[/] 계획을 승인하시겠습니까? (y/n, Enter=승인): "
            )
            return response.strip().lower() in ("y", "yes", "ㅇ", "네", "")
        except (EOFError, KeyboardInterrupt):
            return False

    def ask_rejection_feedback(self) -> str:
        """거부 시 피드백을 수집한다 (blocking). 빈 문자열이면 피드백 없음."""
        try:
            response = self.console.input(
                f"  [{_CLR_BRAND}]?[/] 수정할 내용을 알려주세요 (Enter=취소): "
            )
            return response.strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    # ── Workspace 충돌 HITL ──

    def show_workspace_conflict(self, conflict_info: dict) -> None:
        """Workspace 충돌 분석 결과를 사용자에게 표시한다."""
        conflicts = conflict_info.get("conflicts", [])
        recommendation = conflict_info.get("recommendation", "none")
        detail = conflict_info.get("recommendation_detail", "")
        files = conflict_info.get("files_to_clean", [])
        existing_fw = conflict_info.get("existing_framework", "")

        lines = []
        if existing_fw:
            lines.append(f"**기존 프레임워크**: `{existing_fw}`\n")

        lines.append("**발견된 충돌:**")
        for c in conflicts:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                c.get("severity", ""), "⚪"
            )
            lines.append(
                f"- {severity_icon} [{c.get('type', '')}] {c.get('description', '')}"
            )
            for f in c.get("existing_files", []):
                lines.append(f"  - `{f}`")

        if files:
            lines.append(f"\n**정리 대상 파일** ({len(files)}개):")
            for f in files:
                lines.append(f"- `{f}`")

        lines.append(f"\n**권장 조치**: `{recommendation}`")
        if detail:
            lines.append(f"> {detail}")

        self.console.print()
        self.console.print(
            Panel(
                Markdown("\n".join(lines)),
                title="[bold yellow]⚠ Workspace 충돌 감지[/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    def ask_conflict_approval(self) -> bool:
        """충돌 정리 승인 여부를 사용자에게 묻는다 (blocking)."""
        try:
            response = self.console.input(
                f"  [{_CLR_BRAND}]?[/] 기존 파일을 정리하시겠습니까? (y/n, Enter=승인): "
            )
            return response.strip().lower() in ("y", "yes", "ㅇ", "네", "")
        except (EOFError, KeyboardInterrupt):
            return False

    # ── 환경 승인 HITL ──

    def show_env_approval(self, env_info: dict) -> None:
        """테스트 환경 설정 정보를 사용자에게 표시한다."""
        lines = [
            f"**Workspace**: `{env_info.get('workspace', '?')}`",
            f"**venv 경로**: `{env_info.get('venv_path', '?')}`",
            f"**감지된 런타임**: {', '.join(env_info.get('runtimes', {}).keys()) or '없음'}",
            "",
            "**설치할 의존성**:",
            f"```\n{env_info.get('dependencies', '(없음)')}\n```",
        ]
        self.console.print()
        self.console.print(
            Panel(
                Markdown("\n".join(lines)),
                title="[bold yellow]🔧 테스트 환경 설정 승인 요청[/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    def ask_env_approval(self) -> bool:
        """환경 설정 승인 여부를 사용자에게 묻는다 (blocking)."""
        try:
            response = self.console.input(
                f"  [{_CLR_BRAND}]?[/] 환경 설정을 승인하시겠습니까? (y/n, Enter=승인): "
            )
            return response.strip().lower() in ("y", "yes", "ㅇ", "네", "")
        except (EOFError, KeyboardInterrupt):
            return False

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

    def context_status(
        self,
        input_tokens: int,
        output_tokens: int,
        max_tokens: int,
        compact_threshold: float = 0.8,
    ) -> None:
        """컨텍스트 사용량 상태를 표시한다."""
        pct = (input_tokens / max_tokens * 100) if max_tokens > 0 else 0
        threshold_pct = compact_threshold * 100

        # 프로그레스 바 (20칸)
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)

        # 색상: 정상(dim) / 주의(yellow, 60%+) / 위험(red, 80%+)
        if pct >= threshold_pct:
            color = _CLR_ERROR
            label = "compaction 필요"
        elif pct >= threshold_pct * 0.75:
            color = _CLR_WARN
            label = ""
        else:
            color = _CLR_MUTED
            label = ""

        suffix = f" [{_CLR_WARN}]{label}[/]" if label else ""
        self.console.print(
            f"  [{color}]📊 {bar} {_fmt_tokens(input_tokens)}+{_fmt_tokens(output_tokens)}"
            f" ({pct:.0f}% of {_fmt_tokens(max_tokens)}){suffix}[/]"
        )

    # ── 레거시 호환 ──

    def stream_token(self, token: str) -> None:
        """스트리밍 토큰 출력 (레거시 호환)."""
        self.console.print(token, end="")

    def stream_end(self) -> None:
        """스트리밍 종료 (레거시 호환)."""
        self.console.print()


def _fmt_tokens(n: int) -> str:
    """토큰 수를 축약 표시한다 (DeepAgents 패턴)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _truncate(text: str, max_len: int = 50) -> str:
    """텍스트를 최대 길이로 잘라낸다."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _describe_tool_call(tool_name: str, args: dict | None) -> tuple[str, str]:
    """도구 호출을 사용자 친화적 (label, detail) 쌍으로 변환한다."""
    args = args or {}

    if tool_name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        lines = content.count("\n") + 1 if content else 0
        size = f"({lines}줄)" if lines else ""
        return "write_file", f"{path} {size}".strip()

    if tool_name == "read_file":
        return "read_file", args.get("path", "")

    if tool_name == "run_python":
        code = args.get("code", "")
        # code 내용에서 목적을 추론
        if "py_compile" in code:
            # lint: py_compile로 문법 체크
            import re
            m = re.search(r"['\"]([^'\"]+\.py)['\"]", code)
            target = m.group(1) if m else ""
            return "검증: 문법 체크", target
        if "pytest" in code:
            return "검증: 테스트 실행", ""
        if "node" in code and "--check" in code:
            return "검증: JS 문법 체크", ""
        if "tsc" in code:
            return "검증: TypeScript 체크", ""
        if "eslint" in code or "npx" in code:
            return "검증: Lint", ""
        # 일반 Python 실행
        return "run_python", _truncate(code.split("\n")[0], 50)

    if tool_name == "search_code":
        return "search_code", args.get("query", "")

    if tool_name == "list_directory":
        return "list_directory", args.get("path", ".")

    if tool_name == "search_web":
        return "search_web", _truncate(args.get("query", ""), 50)

    if tool_name == "validate_consistency":
        return "validate_consistency", args.get("target_dir", ".")

    # 기본: tool_name + path/query 인자
    path = args.get("path", args.get("query", ""))
    return tool_name, _truncate(str(path), 50) if path else ""
