"""CLI 실동작 데모 — 실제 LLM + MCP 호출로 UX를 검증한다.

_run_agent_turn()을 직접 호출하여 스킬 자동 활성화, 2단계 파이프라인,
도구 호출, Orchestrator 위임을 확인한다.
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("SKILLS_DIR", "./data/skills")

from rich.console import Console  # noqa: E402

from youngs75_a2a.cli.app import (  # noqa: E402
    _init_skill_registry,
    _run_agent_turn,
)
from youngs75_a2a.cli.renderer import CLIRenderer  # noqa: E402
from youngs75_a2a.cli.session import CLISession  # noqa: E402
from youngs75_a2a.core.parallel_tool_executor import ParallelToolExecutor  # noqa: E402
from youngs75_a2a.core.project_context import ProjectContextLoader  # noqa: E402
from youngs75_a2a.core.tool_permissions import ToolPermissionManager  # noqa: E402


async def main():
    console = Console()
    renderer = CLIRenderer(console)

    # ── 초기화 (CLI _main_loop과 동일) ──
    skill_registry, discovered_skills = _init_skill_registry(
        os.environ.get("SKILLS_DIR")
    )
    session = CLISession(
        agent_name="coding_assistant",
        skill_registry=skill_registry,
    )

    workspace = os.getenv("CODE_TOOLS_WORKSPACE", os.getcwd())
    context_loader = ProjectContextLoader(workspace)
    context_section = context_loader.build_system_prompt_section()
    if context_section:
        session.project_context = context_section
    session.permission_manager = ToolPermissionManager(workspace)
    session.tool_executor = ParallelToolExecutor()

    # ── 시작 배너 ──
    renderer.welcome(session.info.agent_name)
    if discovered_skills:
        renderer.success(
            f"스킬 {len(discovered_skills)}개 로드: {', '.join(discovered_skills)}"
        )
    renderer.success("프로젝트 컨텍스트 로드 완료")
    console.print()

    # ── 테스트 1: 코드 생성 (generate → 스킬 3개 + FAST→STRONG) ──
    console.rule("[bold cyan]테스트 1: 코드 생성 (generate)")
    console.print("[bold blue]❯[/] 이진 탐색 함수 작성해줘\n")
    await _run_agent_turn("이진 탐색 함수 작성해줘", session, renderer)
    console.print()

    # ── 테스트 2: 버그 수정 (fix → debug 스킬) ──
    console.rule("[bold cyan]테스트 2: 버그 수정 (fix)")
    console.print(
        "[bold blue]❯[/] 이 코드의 버그를 수정해줘: "
        "def fib(n): return fib(n-1) + fib(n-2)\n"
    )
    # 에이전트 캐시 클리어 (스킬 자동 활성화를 다시 확인하기 위해)
    session._agents.clear()
    await _run_agent_turn(
        "이 코드의 버그를 수정해줘: def fib(n): return fib(n-1) + fib(n-2)",
        session,
        renderer,
    )
    console.print()

    # ── 테스트 3: 파일 분석 (analyze → MCP 도구 호출 + 스킬 2개) ──
    console.rule("[bold cyan]테스트 3: 파일 분석 (analyze + MCP 도구)")
    console.print("[bold blue]❯[/] pyproject.toml 파일을 읽고 의존성을 분석해줘\n")
    session._agents.clear()
    await _run_agent_turn(
        "pyproject.toml 파일을 읽고 의존성을 분석해줘", session, renderer
    )
    console.print()

    # ── 테스트 4: Orchestrator 위임 ──
    console.rule("[bold cyan]테스트 4: Orchestrator 서브에이전트 위임")
    session.switch_agent("orchestrator")
    session._agents.clear()
    console.print("[bold blue]❯[/] 퀵소트 함수를 작성해줘\n")
    await _run_agent_turn("퀵소트 함수를 작성해줘", session, renderer)
    console.print()

    console.rule("[bold green]전체 데모 완료")


if __name__ == "__main__":
    asyncio.run(main())
