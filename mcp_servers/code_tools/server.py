"""Code Tools MCP 서버 — 파일 I/O, 코드 검색, 코드 실행.

Coding Agent Harness의 핵심 도구를 제공하는 MCP 서버.
에이전트가 프로젝트 컨텍스트를 파악하고, 코드를 작성/수정/실행할 수 있게 한다.

실행:
    python -m youngs75_a2a.mcp_servers.code_tools.server
포트: 3003 (환경변수 CODE_TOOLS_PORT로 변경 가능)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_PORT = int(os.getenv("CODE_TOOLS_PORT", "3003"))
_WORKSPACE = os.getenv("CODE_TOOLS_WORKSPACE", os.getcwd())

mcp = FastMCP(
    name="CodeToolsServer",
    host="0.0.0.0",
    port=_PORT,
)


def _safe_path(path: str) -> Path:
    """workspace 밖으로의 접근을 차단한다."""
    resolved = Path(_WORKSPACE, path).resolve()
    workspace = Path(_WORKSPACE).resolve()
    if not str(resolved).startswith(str(workspace)):
        raise ValueError(f"접근 거부: workspace 밖 경로 ({resolved})")
    return resolved


@mcp.tool()
def read_file(path: str, max_lines: int = 500) -> str:
    """프로젝트 파일의 내용을 읽는다.

    Args:
        path: workspace 기준 상대 경로 (예: "youngs75_a2a/core/base_agent.py")
        max_lines: 최대 읽을 줄 수 (기본 500)
    """
    target = _safe_path(path)
    if not target.exists():
        return f"Error: 파일이 존재하지 않습니다 — {path}"
    if not target.is_file():
        return f"Error: 파일이 아닙니다 — {path}"

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    truncated = len(lines) > max_lines
    content = "\n".join(
        f"{i + 1:4d} | {line}" for i, line in enumerate(lines[:max_lines])
    )
    if truncated:
        content += f"\n... ({len(lines) - max_lines}줄 생략, 총 {len(lines)}줄)"
    return content


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """프로젝트 파일을 생성하거나 덮어쓴다.

    Args:
        path: workspace 기준 상대 경로
        content: 파일에 쓸 전체 내용
    """
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"OK: {path} ({len(content)}자, {content.count(chr(10)) + 1}줄)"


@mcp.tool()
def list_directory(path: str = ".", pattern: str = "*") -> str:
    """디렉토리 내 파일/폴더 목록을 반환한다.

    Args:
        path: workspace 기준 상대 경로 (기본: 프로젝트 루트)
        pattern: glob 패턴 (예: "*.py", "**/*.py")
    """
    target = _safe_path(path)
    if not target.is_dir():
        return f"Error: 디렉토리가 아닙니다 — {path}"

    matches = sorted(target.rglob(pattern) if "**" in pattern else target.glob(pattern))
    # 숨김 파일, __pycache__, .git 제외
    skip = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}
    filtered = [m for m in matches if not any(part in skip for part in m.parts)]

    if not filtered:
        return f"매칭 결과 없음: {path}/{pattern}"

    workspace = Path(_WORKSPACE).resolve()
    lines = []
    for m in filtered[:200]:
        rel = m.relative_to(workspace)
        suffix = "/" if m.is_dir() else f" ({m.stat().st_size}B)"
        lines.append(f"  {rel}{suffix}")

    result = f"총 {len(filtered)}개 (최대 200개 표시):\n" + "\n".join(lines)
    return result


@mcp.tool()
def search_code(
    pattern: str, path: str = ".", file_pattern: str = "*.py", max_results: int = 50
) -> str:
    """프로젝트 코드에서 텍스트 패턴을 검색한다 (grep).

    Args:
        pattern: 검색할 텍스트 또는 정규식
        path: 검색 시작 디렉토리 (workspace 기준)
        file_pattern: 검색할 파일 패턴 (기본: "*.py")
        max_results: 최대 결과 수 (기본 50)
    """
    target = _safe_path(path)
    if not target.exists():
        return f"Error: 경로가 존재하지 않습니다 — {path}"

    try:
        cmd = [
            "grep",
            "-rn",
            "--include",
            file_pattern,
            "-m",
            str(max_results),
            pattern,
            str(target),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: 검색 시간 초과 (10초)"
    except FileNotFoundError:
        # grep이 없으면 Python fallback
        return _search_python_fallback(pattern, target, file_pattern, max_results)

    if not output:
        return f"매칭 결과 없음: '{pattern}' in {path}/{file_pattern}"

    # 절대 경로를 상대 경로로 변환
    workspace = str(Path(_WORKSPACE).resolve())
    output = output.replace(workspace + "/", "")

    lines = output.splitlines()
    if len(lines) >= max_results:
        output += f"\n... (결과가 {max_results}개 제한에 도달)"
    return output


def _search_python_fallback(
    pattern: str, target: Path, file_pattern: str, max_results: int
) -> str:
    """grep이 없는 환경을 위한 Python 기반 검색."""
    import re

    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    results = []
    workspace = Path(_WORKSPACE).resolve()
    for fpath in sorted(target.rglob(file_pattern)):
        if any(part in {".git", "__pycache__", ".venv"} for part in fpath.parts):
            continue
        try:
            for i, line in enumerate(fpath.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    rel = fpath.relative_to(workspace)
                    results.append(f"{rel}:{i}:{line.rstrip()}")
                    if len(results) >= max_results:
                        return "\n".join(results) + f"\n... ({max_results}개 제한 도달)"
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(results) if results else f"매칭 결과 없음: '{pattern}'"


@mcp.tool()
def run_python(code: str, timeout: int = 30) -> str:
    """Python 코드를 실행하고 결과를 반환한다.

    Args:
        code: 실행할 Python 코드
        timeout: 실행 제한 시간(초, 기본 30)
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=_WORKSPACE
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=min(timeout, 60),
            cwd=_WORKSPACE,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        parts = []
        if stdout:
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            parts.append(f"[stderr]\n{stderr}")
        parts.append(f"[exit_code] {result.returncode}")

        return "\n".join(parts) if parts else "[출력 없음]"
    except subprocess.TimeoutExpired:
        return f"Error: 실행 시간 초과 ({timeout}초)"
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print(f"Code Tools MCP 서버 시작: http://0.0.0.0:{_PORT}/mcp")
    print(f"Workspace: {_WORKSPACE}")
    mcp.run(transport="streamable-http")
