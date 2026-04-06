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


# ── Shell 실행 ─────────────────────────────────────────────

# 허용된 명령어 화이트리스트 — 이 목록에 없는 명령어는 차단
_ALLOWED_COMMANDS: set[str] = {
    # Python 패키지 관리
    "pip", "pip3", "uv", "python", "python3",
    # JavaScript/TypeScript
    "npm", "npx", "yarn", "pnpm", "node",
    # Go
    "go",
    # Rust
    "cargo", "rustc",
    # Java
    "mvn", "gradle", "javac", "java",
    # 테스트 실행
    "pytest", "jest", "vitest", "mocha",
    # 빌드/린트
    "make", "cmake", "tsc",
    # 유틸리티
    "cat", "ls", "find", "grep", "head", "tail", "wc", "diff", "sort",
    "mkdir", "cp", "mv", "touch", "echo", "which", "env", "printenv",
    # Git (읽기 전용)
    "git",
}

# 절대 금지 — 화이트리스트에 있더라도 이 서브커맨드는 차단
_BLOCKED_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "sudo",
    "> /dev/",
    "| sh",
    "| bash",
    "curl | ",
    "wget | ",
]


@mcp.tool()
def run_shell(command: str, timeout: int = 120) -> str:
    """Shell 명령어를 실행하고 결과를 반환한다.

    보안: 허용된 명령어만 실행 가능 (pip, npm, go, cargo, pytest, jest 등).
    workspace 디렉토리에서 실행되며, 타임아웃이 적용된다.

    Args:
        command: 실행할 shell 명령어 (예: "pip install -r requirements.txt")
        timeout: 실행 제한 시간(초, 기본 120, 최대 300)
    """
    command = command.strip()
    if not command:
        return "Error: 빈 명령어"

    # 보안 검증 1: 금지 패턴 체크
    for blocked in _BLOCKED_PATTERNS:
        if blocked in command:
            return f"Error: 보안 정책으로 차단됨 — '{blocked}' 패턴 금지"

    # 보안 검증 2: 첫 번째 명령어가 화이트리스트에 있는지 확인
    first_token = command.split()[0].split("/")[-1]  # 경로 제거
    if first_token not in _ALLOWED_COMMANDS:
        return (
            f"Error: 허용되지 않은 명령어 — '{first_token}'\n"
            f"허용 목록: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )

    # 파이프라인 명령어의 각 단계도 검증
    if "|" in command:
        for segment in command.split("|"):
            seg_cmd = segment.strip().split()[0].split("/")[-1] if segment.strip() else ""
            if seg_cmd and seg_cmd not in _ALLOWED_COMMANDS:
                return f"Error: 파이프라인 내 허용되지 않은 명령어 — '{seg_cmd}'"

    effective_timeout = min(timeout, 300)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=_WORKSPACE,
            env={**os.environ, "PYTHONPATH": _WORKSPACE},
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        parts = []
        if stdout:
            # 출력이 너무 길면 truncate
            if len(stdout) > 5000:
                stdout = stdout[:5000] + f"\n... (총 {len(result.stdout)}자, 5000자까지 표시)"
            parts.append(f"[stdout]\n{stdout}")
        if stderr:
            if len(stderr) > 3000:
                stderr = stderr[:3000] + f"\n... (총 {len(result.stderr)}자, 3000자까지 표시)"
            parts.append(f"[stderr]\n{stderr}")
        parts.append(f"[exit_code] {result.returncode}")

        return "\n".join(parts) if parts else "[출력 없음]"
    except subprocess.TimeoutExpired:
        return f"Error: 실행 시간 초과 ({effective_timeout}초)"


@mcp.tool()
def str_replace(path: str, old_str: str, new_str: str) -> str:
    """파일에서 특정 문자열을 찾아 교체한다.

    전체 파일 덮어쓰기 대신 정확한 부분만 수정하여 안전성을 높인다.

    Args:
        path: workspace 기준 상대 경로
        old_str: 교체할 기존 문자열 (정확히 일치해야 함)
        new_str: 새로운 문자열
    """
    target = _safe_path(path)
    if not target.exists():
        return f"Error: 파일이 존재하지 않습니다 — {path}"
    if not target.is_file():
        return f"Error: 파일이 아닙니다 — {path}"

    content = target.read_text(encoding="utf-8")
    count = content.count(old_str)

    if count == 0:
        return "Error: 찾을 수 없음 — old_str이 파일에 존재하지 않습니다. 정확한 문자열을 확인하세요."
    if count > 1:
        return (
            f"Error: 모호한 매치 — old_str이 {count}번 발견되었습니다. "
            "더 많은 컨텍스트(주변 줄)를 포함하여 고유하게 만들어주세요."
        )

    new_content = content.replace(old_str, new_str, 1)
    target.write_text(new_content, encoding="utf-8")
    return f"✓ {path}: 교체 완료 ({len(old_str)} → {len(new_str)} 문자)"


@mcp.tool()
def apply_patch(patch: str) -> str:
    """Unified diff 형식의 패치를 파일에 적용한다.

    Args:
        patch: unified diff 형식 패치 문자열.
               --- a/path 와 +++ b/path 헤더로 대상 파일을 식별한다.
    """
    lines = patch.strip().splitlines()

    # 대상 파일 찾기 (+++ b/path)
    target_path = None
    for line in lines:
        if line.startswith("+++ b/"):
            target_path = line[6:].strip()
            break
        if line.startswith("+++ "):
            target_path = line[4:].strip()
            break

    if not target_path:
        return "Error: 패치에서 대상 파일을 찾을 수 없습니다 (+++ 헤더 필요)"

    target = _safe_path(target_path)

    # 새 파일 생성 (--- /dev/null)
    is_new_file = any(line.startswith("--- /dev/null") for line in lines)
    if is_new_file:
        target.parent.mkdir(parents=True, exist_ok=True)
        new_content = []
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue
            if in_hunk:
                if line.startswith("+"):
                    new_content.append(line[1:])
                elif line.startswith("-"):
                    continue
                elif not line.startswith("\\"):
                    new_content.append(line[1:] if line.startswith(" ") else line)
        target.write_text("\n".join(new_content) + "\n", encoding="utf-8")
        return f"✓ {target_path}: 새 파일 생성"

    if not target.exists():
        return f"Error: 파일이 존재하지 않습니다 — {target_path}"

    original = target.read_text(encoding="utf-8")
    original_lines = original.splitlines(keepends=True)

    # 헝크 파싱 및 적용
    hunks = []
    current_hunk = None
    for line in lines:
        if line.startswith("@@"):
            # @@ -start,count +start,count @@ 파싱
            import re

            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                current_hunk = {
                    "orig_start": int(m.group(1)) - 1,
                    "lines": [],
                }
                hunks.append(current_hunk)
        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if not hunks:
        return "Error: 패치에 유효한 헝크(@@ ... @@)가 없습니다"

    # 헝크를 역순으로 적용 (인덱스 밀림 방지)
    result_lines = list(original_lines)
    for hunk in reversed(hunks):
        offset = hunk["orig_start"]
        new_lines = []
        remove_count = 0

        for hl in hunk["lines"]:
            if hl.startswith("-"):
                remove_count += 1
            elif hl.startswith("+"):
                content = hl[1:]
                if not content.endswith("\n"):
                    content += "\n"
                new_lines.append(content)
            elif hl.startswith(" "):
                content = hl[1:]
                if not content.endswith("\n"):
                    content += "\n"
                new_lines.append(content)
                remove_count += 1
            elif hl.startswith("\\"):
                continue

        result_lines[offset : offset + remove_count] = new_lines

    target.write_text("".join(result_lines), encoding="utf-8")
    return f"✓ {target_path}: 패치 적용 완료 ({len(hunks)}개 헝크)"


if __name__ == "__main__":
    print(f"Code Tools MCP 서버 시작: http://0.0.0.0:{_PORT}/mcp")
    print(f"Workspace: {_WORKSPACE}")
    mcp.run(transport="streamable-http")
