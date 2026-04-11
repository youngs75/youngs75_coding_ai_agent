"""Code Tools MCP 서버 — 파일 I/O, 코드 검색, 코드 실행.

Coding Agent Harness의 핵심 도구를 제공하는 MCP 서버.
에이전트가 프로젝트 컨텍스트를 파악하고, 코드를 작성/수정/실행할 수 있게 한다.

실행:
    python -m coding_agent.mcp_servers.code_tools.server
포트: 3003 (환경변수 CODE_TOOLS_PORT로 변경 가능)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_PORT = int(os.getenv("CODE_TOOLS_PORT", "3003"))
_WORKSPACE = os.getenv("CODE_TOOLS_WORKSPACE", os.getcwd())
# set_workspace 허용 루트 — 이 경로 하위만 workspace로 설정 가능
_MOUNT_ROOT = os.getenv("MCP_MOUNT_ROOT", "/")

mcp = FastMCP(
    name="CodeToolsServer",
    host="0.0.0.0",
    port=_PORT,
)


def _normalize_to_relative(path: str) -> str:
    """경로를 workspace 기준 상대 경로로 정규화한다.

    절대 경로가 workspace 하위이면 상대 경로로 변환하고,
    이미 상대 경로이면 '..' 등을 제거하여 정규화한다.
    """
    workspace = Path(_WORKSPACE).resolve()
    p = Path(path)

    if p.is_absolute():
        resolved = p.resolve()
        # workspace 하위인 경우 상대 경로로 변환
        try:
            return str(resolved.relative_to(workspace))
        except ValueError:
            # workspace 밖 절대 경로 — 그대로 반환 (_safe_path에서 차단됨)
            return path
    else:
        # 상대 경로: "./" 제거, ".." 정규화
        normalized = Path(os.path.normpath(path))
        return str(normalized)


def _safe_path(path: str) -> Path:
    """workspace 밖으로의 접근을 차단한다."""
    # 먼저 상대 경로로 정규화
    rel_path = _normalize_to_relative(path)
    resolved = Path(_WORKSPACE, rel_path).resolve()
    workspace = Path(_WORKSPACE).resolve()
    if not str(resolved).startswith(str(workspace)):
        raise ValueError(f"접근 거부: workspace 밖 경로 ({resolved})")
    return resolved


@mcp.tool()
def set_workspace(path: str) -> str:
    """작업 디렉토리(workspace)를 변경한다.

    에이전트가 파일을 읽고 쓰는 기준 디렉토리를 동적으로 전환한다.
    마운트된 호스트 경로 하위만 허용된다.

    Args:
        path: 새 workspace 절대 경로 (컨테이너 내부 경로)
    """
    global _WORKSPACE
    resolved = Path(path).resolve()
    mount_root = Path(_MOUNT_ROOT).resolve()
    if not str(resolved).startswith(str(mount_root)):
        return f"Error: 허용 범위 밖 경로 — {resolved} (허용: {mount_root} 하위)"
    if not resolved.is_dir():
        # 디렉토리가 없으면 생성
        resolved.mkdir(parents=True, exist_ok=True)
    _WORKSPACE = str(resolved)
    return f"Workspace 변경됨: {_WORKSPACE}"


@mcp.tool()
def get_workspace() -> str:
    """현재 workspace 경로를 반환한다."""
    return _WORKSPACE


@mcp.tool()
def read_file(path: str, max_lines: int = 500) -> str:
    """프로젝트 파일의 내용을 읽는다.

    Args:
        path: workspace 기준 상대 경로 (예: "coding_agent/core/base_agent.py")
        max_lines: 최대 읽을 줄 수 (기본 500)
    """
    path = _normalize_to_relative(path)
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
    rel_path = _normalize_to_relative(path)
    target = _safe_path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"OK: {rel_path} ({len(content)}자, {content.count(chr(10)) + 1}줄)"


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
    "cd", "pwd",
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
def run_shell(command: str, timeout: int = 120, cwd: str = ".") -> str:
    """Shell 명령어를 실행하고 결과를 반환한다.

    보안: 허용된 명령어만 실행 가능 (pip, npm, go, cargo, pytest, jest 등).
    workspace 디렉토리에서 실행되며, 타임아웃이 적용된다.

    Args:
        command: 실행할 shell 명령어 (예: "pip install -r requirements.txt")
        timeout: 실행 제한 시간(초, 기본 120, 최대 300)
        cwd: 작업 디렉토리 (workspace 기준 상대 경로, 기본 ".")
    """
    command = command.strip()
    if not command:
        return "Error: 빈 명령어"

    # cwd 처리: workspace 기준 상대 경로로 해석
    try:
        work_dir = _safe_path(cwd) if cwd != "." else Path(_WORKSPACE).resolve()
    except ValueError as e:
        return f"Error: {e}"
    if not work_dir.is_dir():
        return f"Error: 디렉토리가 존재하지 않습니다: {cwd}"

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
    # 주의: 코드 문자열 내부의 | (비트 OR, 논리 OR 등)를 파이프로 오인하지 않도록
    # 실제 셸 파이프는 공백을 동반하므로 " | " 패턴으로 분리한다.
    import re as _re
    pipe_segments = _re.split(r'\s\|\s', command)
    if len(pipe_segments) > 1:
        for segment in pipe_segments[1:]:  # 첫 세그먼트는 이미 검증됨
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
            cwd=str(work_dir),
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



# ── 프로젝트 템플릿 도구 ─────────────────────────────────────

_PROJECT_TEMPLATES: dict[str, dict] = {
    "flask-vue": {
        "name": "flask-vue",
        "description": "Flask + Vue.js SPA 풀스택 프로젝트",
        "stack": {"backend": "Flask", "frontend": "Vue 3 + Vite", "language": "Python / JavaScript"},
        "structure": [
            "backend/", "backend/app/__init__.py", "backend/app/routes/__init__.py",
            "backend/app/routes/api.py", "backend/requirements.txt", "backend/run.py",
            "frontend/", "frontend/src/App.vue", "frontend/src/main.js",
            "frontend/index.html", "frontend/package.json", "frontend/vite.config.js",
            "README.md",
        ],
        "files": {
            "backend/app/__init__.py": '"""Flask app factory."""\nfrom flask import Flask\nfrom flask_cors import CORS\n\n\ndef create_app(config=None):\n    app = Flask(__name__)\n    CORS(app)\n\n    if config:\n        app.config.update(config)\n\n    from app.routes.api import api_bp\n    app.register_blueprint(api_bp, url_prefix="/api")\n\n    return app\n',
            "backend/app/routes/__init__.py": "",
            "backend/app/routes/api.py": '"""API 라우트."""\nfrom flask import Blueprint, jsonify\n\napi_bp = Blueprint("api", __name__)\n\n\n@api_bp.route("/health")\ndef health():\n    return jsonify({"status": "ok"})\n\n\n@api_bp.route("/hello")\ndef hello():\n    return jsonify({"message": "Hello from Flask!"})\n',
            "backend/requirements.txt": "flask>=3.0\nflask-cors>=4.0\ngunicorn>=22.0\n",
            "backend/run.py": '"""개발 서버 실행."""\nfrom app import create_app\n\napp = create_app()\n\nif __name__ == "__main__":\n    app.run(debug=True, port=5000)\n',
            "frontend/src/App.vue": '<template>\n  <div id="app">\n    <h1>{{ msg }}</h1>\n  </div>\n</template>\n\n<script setup>\nimport { ref, onMounted } from \'vue\'\n\nconst msg = ref(\'Loading...\')\n\nonMounted(async () => {\n  const res = await fetch(\'/api/hello\')\n  const data = await res.json()\n  msg.value = data.message\n})\n</script>\n',
            "frontend/src/main.js": "import { createApp } from 'vue'\nimport App from './App.vue'\n\ncreateApp(App).mount('#app')\n",
            "frontend/index.html": '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>{project_name}</title>\n</head>\n<body>\n  <div id="app"></div>\n  <script type="module" src="/src/main.js"></script>\n</body>\n</html>\n',
            "frontend/package.json": '{{\n  "name": "{project_name}-frontend",\n  "private": true,\n  "version": "0.1.0",\n  "scripts": {{\n    "dev": "vite",\n    "build": "vite build",\n    "preview": "vite preview"\n  }},\n  "dependencies": {{\n    "vue": "^3.4"\n  }},\n  "devDependencies": {{\n    "@vitejs/plugin-vue": "^5.0",\n    "vite": "^5.0"\n  }}\n}}\n',
            "frontend/vite.config.js": "import { defineConfig } from 'vite'\nimport vue from '@vitejs/plugin-vue'\n\nexport default defineConfig({\n  plugins: [vue()],\n  server: {\n    proxy: {\n      '/api': 'http://localhost:5000'\n    }\n  }\n})\n",
            "README.md": "# {project_name}\n\nFlask + Vue.js SPA 프로젝트.\n\n## 시작하기\n\n### Backend\n```bash\ncd backend\npip install -r requirements.txt\npython run.py\n```\n\n### Frontend\n```bash\ncd frontend\nnpm install\nnpm run dev\n```\n",
        },
    },
    "react-express": {
        "name": "react-express",
        "description": "React + Express.js 풀스택 프로젝트",
        "stack": {"backend": "Express.js", "frontend": "React + Vite", "language": "JavaScript"},
        "structure": [
            "client/", "client/src/App.jsx", "client/src/main.jsx",
            "client/index.html", "client/package.json", "client/vite.config.js",
            "server/", "server/index.js", "server/routes/api.js", "server/package.json",
            "README.md",
        ],
        "files": {
            "server/index.js": "const express = require('express');\nconst cors = require('cors');\nconst apiRouter = require('./routes/api');\n\nconst app = express();\nconst PORT = process.env.PORT || 3000;\n\napp.use(cors());\napp.use(express.json());\napp.use('/api', apiRouter);\n\napp.listen(PORT, () => {\n  console.log(`Server running on port ${PORT}`);\n});\n",
            "server/routes/api.js": "const express = require('express');\nconst router = express.Router();\n\nrouter.get('/health', (req, res) => {\n  res.json({ status: 'ok' });\n});\n\nrouter.get('/hello', (req, res) => {\n  res.json({ message: 'Hello from Express!' });\n});\n\nmodule.exports = router;\n",
            "server/package.json": '{{\n  "name": "{project_name}-server",\n  "version": "0.1.0",\n  "private": true,\n  "scripts": {{\n    "start": "node index.js",\n    "dev": "node --watch index.js"\n  }},\n  "dependencies": {{\n    "cors": "^2.8",\n    "express": "^4.18"\n  }}\n}}\n',
            "client/src/App.jsx": "import { useState, useEffect } from 'react'\n\nfunction App() {\n  const [message, setMessage] = useState('Loading...')\n\n  useEffect(() => {\n    fetch('/api/hello')\n      .then(res => res.json())\n      .then(data => setMessage(data.message))\n  }, [])\n\n  return (\n    <div>\n      <h1>{message}</h1>\n    </div>\n  )\n}\n\nexport default App\n",
            "client/src/main.jsx": "import React from 'react'\nimport ReactDOM from 'react-dom/client'\nimport App from './App'\n\nReactDOM.createRoot(document.getElementById('root')).render(\n  <React.StrictMode>\n    <App />\n  </React.StrictMode>\n)\n",
            "client/index.html": '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>{project_name}</title>\n</head>\n<body>\n  <div id="root"></div>\n  <script type="module" src="/src/main.jsx"></script>\n</body>\n</html>\n',
            "client/package.json": '{{\n  "name": "{project_name}-client",\n  "private": true,\n  "version": "0.1.0",\n  "scripts": {{\n    "dev": "vite",\n    "build": "vite build",\n    "preview": "vite preview"\n  }},\n  "dependencies": {{\n    "react": "^18.3",\n    "react-dom": "^18.3"\n  }},\n  "devDependencies": {{\n    "@vitejs/plugin-react": "^4.0",\n    "vite": "^5.0"\n  }}\n}}\n',
            "client/vite.config.js": "import { defineConfig } from 'vite'\nimport react from '@vitejs/plugin-react'\n\nexport default defineConfig({\n  plugins: [react()],\n  server: {\n    proxy: {\n      '/api': 'http://localhost:3000'\n    }\n  }\n})\n",
            "README.md": "# {project_name}\n\nReact + Express.js 풀스택 프로젝트.\n\n## 시작하기\n\n### Server\n```bash\ncd server\nnpm install\nnpm run dev\n```\n\n### Client\n```bash\ncd client\nnpm install\nnpm run dev\n```\n",
        },
    },
    "fastapi-react": {
        "name": "fastapi-react",
        "description": "FastAPI + React 풀스택 프로젝트",
        "stack": {"backend": "FastAPI", "frontend": "React + Vite", "language": "Python / JavaScript"},
        "structure": [
            "backend/", "backend/app/__init__.py", "backend/app/main.py",
            "backend/app/routers/__init__.py", "backend/app/routers/api.py",
            "backend/requirements.txt",
            "frontend/", "frontend/src/App.jsx", "frontend/src/main.jsx",
            "frontend/index.html", "frontend/package.json", "frontend/vite.config.js",
            "README.md",
        ],
        "files": {
            "backend/app/__init__.py": "",
            "backend/app/main.py": '"""FastAPI 메인 앱."""\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\nfrom app.routers import api\n\napp = FastAPI(title="{project_name}")\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=["*"],\n    allow_credentials=True,\n    allow_methods=["*"],\n    allow_headers=["*"],\n)\n\napp.include_router(api.router, prefix="/api")\n\n\n@app.get("/")\nasync def root():\n    return {"message": "Welcome to {project_name}"}\n',
            "backend/app/routers/__init__.py": "",
            "backend/app/routers/api.py": '"""API 라우터."""\nfrom fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n@router.get("/health")\nasync def health():\n    return {"status": "ok"}\n\n\n@router.get("/hello")\nasync def hello():\n    return {"message": "Hello from FastAPI!"}\n',
            "backend/requirements.txt": "fastapi>=0.110\nuvicorn[standard]>=0.29\n",
            "frontend/src/App.jsx": "import { useState, useEffect } from 'react'\n\nfunction App() {\n  const [message, setMessage] = useState('Loading...')\n\n  useEffect(() => {\n    fetch('/api/hello')\n      .then(res => res.json())\n      .then(data => setMessage(data.message))\n  }, [])\n\n  return (\n    <div>\n      <h1>{message}</h1>\n    </div>\n  )\n}\n\nexport default App\n",
            "frontend/src/main.jsx": "import React from 'react'\nimport ReactDOM from 'react-dom/client'\nimport App from './App'\n\nReactDOM.createRoot(document.getElementById('root')).render(\n  <React.StrictMode>\n    <App />\n  </React.StrictMode>\n)\n",
            "frontend/index.html": '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>{project_name}</title>\n</head>\n<body>\n  <div id="root"></div>\n  <script type="module" src="/src/main.jsx"></script>\n</body>\n</html>\n',
            "frontend/package.json": '{{\n  "name": "{project_name}-frontend",\n  "private": true,\n  "version": "0.1.0",\n  "scripts": {{\n    "dev": "vite",\n    "build": "vite build",\n    "preview": "vite preview"\n  }},\n  "dependencies": {{\n    "react": "^18.3",\n    "react-dom": "^18.3"\n  }},\n  "devDependencies": {{\n    "@vitejs/plugin-react": "^4.0",\n    "vite": "^5.0"\n  }}\n}}\n',
            "frontend/vite.config.js": "import { defineConfig } from 'vite'\nimport react from '@vitejs/plugin-react'\n\nexport default defineConfig({\n  plugins: [react()],\n  server: {\n    proxy: {\n      '/api': 'http://localhost:8000'\n    }\n  }\n})\n",
            "README.md": "# {project_name}\n\nFastAPI + React 풀스택 프로젝트.\n\n## 시작하기\n\n### Backend\n```bash\ncd backend\npip install -r requirements.txt\nuvicorn app.main:app --reload\n```\n\n### Frontend\n```bash\ncd frontend\nnpm install\nnpm run dev\n```\n",
        },
    },
    "django-htmx": {
        "name": "django-htmx",
        "description": "Django + HTMX 서버사이드 렌더링 프로젝트",
        "stack": {"backend": "Django", "frontend": "HTMX", "language": "Python"},
        "structure": [
            "config/", "config/__init__.py", "config/settings.py", "config/urls.py", "config/wsgi.py",
            "core/", "core/__init__.py", "core/views.py", "core/urls.py",
            "templates/", "templates/base.html", "templates/core/index.html",
            "static/css/style.css", "manage.py", "requirements.txt", "README.md",
        ],
        "files": {
            "config/__init__.py": "",
            "config/settings.py": '"""Django 설정."""\nimport os\nfrom pathlib import Path\n\nBASE_DIR = Path(__file__).resolve().parent.parent\nSECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-change-in-production")\nDEBUG = os.getenv("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")\nALLOWED_HOSTS = ["*"]\n\nINSTALLED_APPS = [\n    "django.contrib.admin",\n    "django.contrib.auth",\n    "django.contrib.contenttypes",\n    "django.contrib.sessions",\n    "django.contrib.messages",\n    "django.contrib.staticfiles",\n    "django_htmx",\n    "core",\n]\n\nMIDDLEWARE = [\n    "django.middleware.security.SecurityMiddleware",\n    "django.contrib.sessions.middleware.SessionMiddleware",\n    "django.middleware.common.CommonMiddleware",\n    "django.middleware.csrf.CsrfViewMiddleware",\n    "django.contrib.auth.middleware.AuthenticationMiddleware",\n    "django.contrib.messages.middleware.MessageMiddleware",\n    "django_htmx.middleware.HtmxMiddleware",\n]\n\nROOT_URLCONF = "config.urls"\n\nTEMPLATES = [\n    {\n        "BACKEND": "django.template.backends.django.DjangoTemplates",\n        "DIRS": [BASE_DIR / "templates"],\n        "APP_DIRS": True,\n        "OPTIONS": {\n            "context_processors": [\n                "django.template.context_processors.debug",\n                "django.template.context_processors.request",\n                "django.contrib.auth.context_processors.auth",\n                "django.contrib.messages.context_processors.messages",\n            ],\n        },\n    },\n]\n\nWSGI_APPLICATION = "config.wsgi.application"\n\nDATABASES = {\n    "default": {\n        "ENGINE": "django.db.backends.sqlite3",\n        "NAME": BASE_DIR / "db.sqlite3",\n    }\n}\n\nSTATIC_URL = "/static/"\nSTATICFILES_DIRS = [BASE_DIR / "static"]\nDEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"\n',
            "config/urls.py": '"""URL 설정."""\nfrom django.contrib import admin\nfrom django.urls import path, include\n\nurlpatterns = [\n    path("admin/", admin.site.urls),\n    path("", include("core.urls")),\n]\n',
            "config/wsgi.py": '"""WSGI config."""\nimport os\nfrom django.core.wsgi import get_wsgi_application\n\nos.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")\napplication = get_wsgi_application()\n',
            "core/__init__.py": "",
            "core/views.py": '"""핵심 뷰."""\nfrom django.shortcuts import render\n\n\ndef index(request):\n    """메인 페이지."""\n    return render(request, "core/index.html")\n\n\ndef hello(request):\n    """HTMX 부분 렌더링 예시."""\n    if request.htmx:\n        return render(request, "core/_hello.html", {"message": "Hello from Django + HTMX!"})\n    return render(request, "core/index.html")\n',
            "core/urls.py": '"""Core URL 패턴."""\nfrom django.urls import path\nfrom . import views\n\napp_name = "core"\n\nurlpatterns = [\n    path("", views.index, name="index"),\n    path("hello/", views.hello, name="hello"),\n]\n',
            "templates/base.html": '<!DOCTYPE html>\n<html lang="ko">\n<head>\n  <meta charset="UTF-8">\n  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n  <title>{project_name}</title>\n  <script src="https://unpkg.com/htmx.org@2.0.4"></script>\n  {%% load static %%}\n  <link rel="stylesheet" href="{%% static \'css/style.css\' %%}">\n</head>\n<body>\n  <main>\n    {%% block content %%}{%% endblock %%}\n  </main>\n</body>\n</html>\n',
            "templates/core/index.html": '{%% extends "base.html" %%}\n\n{%% block content %%}\n<h1>Welcome to {project_name}</h1>\n<button hx-get="{%% url \'core:hello\' %%}" hx-target="#result" hx-swap="innerHTML">\n  Say Hello\n</button>\n<div id="result"></div>\n{%% endblock %%}\n',
            "templates/core/_hello.html": "<p>{{ message }}</p>\n",
            "static/css/style.css": "body {\n  font-family: system-ui, -apple-system, sans-serif;\n  max-width: 800px;\n  margin: 2rem auto;\n  padding: 0 1rem;\n}\n\nbutton {\n  padding: 0.5rem 1rem;\n  cursor: pointer;\n}\n",
            "manage.py": '#!/usr/bin/env python\n"""Django management."""\nimport os\nimport sys\n\n\ndef main():\n    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")\n    from django.core.management import execute_from_command_line\n    execute_from_command_line(sys.argv)\n\n\nif __name__ == "__main__":\n    main()\n',
            "requirements.txt": "django>=5.0\ndjango-htmx>=1.17\ngunicorn>=22.0\n",
            "README.md": "# {project_name}\n\nDjango + HTMX 서버사이드 렌더링 프로젝트.\n\n## 시작하기\n\n```bash\npip install -r requirements.txt\npython manage.py migrate\npython manage.py runserver\n```\n",
        },
    },
}


@mcp.tool()
def list_templates() -> str:
    """사용 가능한 프레임워크 템플릿 목록을 반환한다.

    각 템플릿의 이름, 설명, 기술 스택 정보를 포함한다.
    """
    results = []
    for tpl in _PROJECT_TEMPLATES.values():
        stack_str = ", ".join(f"{k}: {v}" for k, v in tpl["stack"].items())
        results.append(f"- **{tpl['name']}**: {tpl['description']}\n  Stack: {stack_str}")
    return "사용 가능한 템플릿:\n\n" + "\n\n".join(results)


@mcp.tool()
def get_template_info(template_name: str) -> str:
    """특정 템플릿의 상세 정보를 반환한다.

    디렉토리 구조, 주요 파일, 의존성 목록, 설정 파일 등을 포함한다.

    Args:
        template_name: 템플릿 이름 (예: "flask-vue", "react-express")
    """
    tpl = _PROJECT_TEMPLATES.get(template_name)
    if not tpl:
        available = ", ".join(_PROJECT_TEMPLATES.keys())
        return f"Error: 존재하지 않는 템플릿 '{template_name}'. 사용 가능: {available}"

    stack_str = ", ".join(f"{k}: {v}" for k, v in tpl["stack"].items())
    structure_str = "\n".join(f"  {s}" for s in tpl["structure"])

    deps = []
    for fname, content in tpl["files"].items():
        if fname.endswith("requirements.txt") or fname.endswith("package.json"):
            deps.append(f"  [{fname}]\n  {content.strip()}")

    deps_str = "\n\n".join(deps) if deps else "  (없음)"

    return (
        f"템플릿: {tpl['name']}\n"
        f"설명: {tpl['description']}\n"
        f"스택: {stack_str}\n\n"
        f"디렉토리 구조:\n{structure_str}\n\n"
        f"의존성:\n{deps_str}\n\n"
        f"총 파일 수: {len(tpl['files'])}개"
    )


@mcp.tool()
def create_project(template_name: str, project_name: str, options: str = "{}") -> str:
    """workspace 내에 프로젝트 보일러플레이트를 생성한다.

    선택한 템플릿을 기반으로 디렉토리 구조와 파일을 생성한다.
    workspace 밖 경로 접근은 차단된다.

    Args:
        template_name: 템플릿 이름 (예: "flask-vue")
        project_name: 생성할 프로젝트 이름 (디렉토리명)
        options: JSON 문자열 형태의 추가 옵션 (예: '{"db_type": "postgres"}')
    """
    tpl = _PROJECT_TEMPLATES.get(template_name)
    if not tpl:
        available = ", ".join(_PROJECT_TEMPLATES.keys())
        return f"Error: 존재하지 않는 템플릿 '{template_name}'. 사용 가능: {available}"

    try:
        json.loads(options) if options else {}
    except json.JSONDecodeError as e:
        return f"Error: options JSON 파싱 실패 — {e}"

    project_dir = _safe_path(project_name)

    if project_dir.exists():
        return f"Error: 이미 존재하는 디렉토리 — {project_name}"

    created_files = []
    try:
        for rel_path, content in tpl["files"].items():
            file_path = project_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            rendered = content.replace("{project_name}", project_name)
            file_path.write_text(rendered, encoding="utf-8")
            created_files.append(rel_path)
    except OSError as e:
        return f"Error: 파일 생성 실패 — {e}"

    files_str = "\n".join(f"  {f}" for f in created_files)
    return (
        f"프로젝트 '{project_name}' 생성 완료! (템플릿: {template_name})\n\n"
        f"생성된 파일 ({len(created_files)}개):\n{files_str}\n\n"
        f"경로: {project_dir}"
    )


# ══════════════════════════════════════════════════════════════
#  검색 도구 (Tavily / Serper / ArXiv)
# ══════════════════════════════════════════════════════════════


def _search_with_serper(
    query: str, max_results: int = 5, search_type: str = "search"
) -> str:
    """Serper API로 검색. 키가 없거나 실패 시 빈 문자열 반환."""
    import httpx

    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return ""

    try:
        resp = httpx.post(
            f"https://google.serper.dev/{search_type}",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception:
        return ""

    data = resp.json()
    output = []
    for item in data.get("organic", [])[:max_results]:
        output.append(f"### {item.get('title', 'N/A')}")
        output.append(f"URL: {item.get('link', '')}")
        output.append(f"{item.get('snippet', '')}")
        output.append("")
    return "\n".join(output)


def _search_with_tavily(
    query: str, max_results: int = 5, topic: str = "general"
) -> str:
    """Tavily API로 검색. 키가 없으면 에러 메시지 반환."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY가 설정되지 않았습니다."

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, max_results=max_results, topic=topic)

    output = []
    for r in results.get("results", []):
        output.append(f"### {r.get('title', 'N/A')}")
        output.append(f"URL: {r.get('url', '')}")
        output.append(f"{r.get('content', '')}")
        output.append("")
    return "\n".join(output) if output else "검색 결과가 없습니다."


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """웹에서 정보를 검색합니다. (Serper → Tavily 폴백)

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5)
    """
    result = _search_with_serper(query, max_results)
    if not result:
        result = _search_with_tavily(query, max_results)
    return result


@mcp.tool()
def search_news(query: str, max_results: int = 5) -> str:
    """최신 뉴스를 검색합니다. (Serper → Tavily 폴백)

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수 (기본 5)
    """
    result = _search_with_serper(query, max_results, search_type="news")
    if not result:
        result = _search_with_tavily(query, max_results, topic="news")
    return result


@mcp.tool()
def search_papers(query: str, max_results: int = 5) -> str:
    """arXiv에서 학술 논문을 검색합니다.

    Args:
        query: 검색 쿼리 (영어 권장)
        max_results: 최대 결과 수 (기본 5)
    """
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    output = []
    for result in client.results(search):
        output.append(f"### {result.title}")
        output.append(f"Authors: {', '.join(a.name for a in result.authors[:3])}")
        output.append(f"Published: {result.published.strftime('%Y-%m-%d')}")
        output.append(f"URL: {result.entry_id}")
        summary = result.summary.replace("\n", " ")[:300]
        output.append(f"Abstract: {summary}...")
        output.append("")

    return "\n".join(output) if output else "검색 결과가 없습니다."


@mcp.tool()
def search_recent_papers(query: str, max_results: int = 5) -> str:
    """arXiv에서 최신 논문을 날짜순으로 검색합니다.

    Args:
        query: 검색 쿼리 (영어 권장)
        max_results: 최대 결과 수 (기본 5)
    """
    import arxiv

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    output = []
    for result in client.results(search):
        output.append(f"### {result.title}")
        output.append(f"Authors: {', '.join(a.name for a in result.authors[:3])}")
        output.append(f"Published: {result.published.strftime('%Y-%m-%d')}")
        output.append(f"URL: {result.entry_id}")
        categories = ", ".join(result.categories[:3])
        output.append(f"Categories: {categories}")
        summary = result.summary.replace("\n", " ")[:300]
        output.append(f"Abstract: {summary}...")
        output.append("")

    return "\n".join(output) if output else "검색 결과가 없습니다."


@mcp.tool()
def validate_consistency(target_dir: str = ".") -> str:
    """workspace 내 Python 파일들의 import/함수 시그니처 일관성을 검증한다.

    코드 생성 후 파일 간 불일치를 감지하는 AST 기반 정적 분석 도구.

    검증 항목:
    1. 함수 정의 vs 호출 시그니처 불일치 (인자 개수)
    2. import 대상 파일/모듈 존재 여부
    3. SyntaxError 감지

    Args:
        target_dir: 검증할 디렉토리 (workspace 기준 상대경로, 기본 ".")
    """
    import ast

    target = _safe_path(_normalize_to_relative(target_dir))
    issues: list[str] = []

    # 함수 정의/호출/import 수집
    definitions: dict[str, dict] = {}
    calls: list[tuple[str, int, str, int]] = []
    imports: list[tuple[str, str]] = []

    workspace_root = Path(_WORKSPACE).resolve()

    for py_file in target.rglob("*.py"):
        if any(p in py_file.parts for p in (".venv", "__pycache__", "node_modules")):
            continue
        rel = str(py_file.relative_to(workspace_root))
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            issues.append(f"[SYNTAX] {rel}:{e.lineno}: {e.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = node.args
                params = [a.arg for a in args.args if a.arg not in ("self", "cls")]
                defaults_count = len(args.defaults)
                required = max(0, len(params) - defaults_count)
                definitions[node.name] = {
                    "file": rel, "required": required,
                    "total": len(params), "line": node.lineno,
                }
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append((node.func.id, len(node.args), rel, node.lineno))
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append((node.module, rel))

    # 시그니처 불일치 검출
    for func_name, arg_count, call_file, call_line in calls:
        if func_name not in definitions or func_name.startswith("_"):
            continue
        defn = definitions[func_name]
        if arg_count < defn["required"]:
            issues.append(
                f"[SIGNATURE] {call_file}:{call_line}: {func_name}() "
                f"인자 {arg_count}개, 정의({defn['file']}:{defn['line']})는 "
                f"필수 {defn['required']}개"
            )
        elif arg_count > defn["total"]:
            issues.append(
                f"[SIGNATURE] {call_file}:{call_line}: {func_name}() "
                f"인자 {arg_count}개, 정의({defn['file']}:{defn['line']})는 "
                f"최대 {defn['total']}개"
            )

    # import 대상 존재 확인 (프로젝트 내부 모듈만)
    for module_path, imp_file in imports:
        parts = module_path.split(".")
        top_dir = workspace_root / parts[0]
        if not (top_dir.exists() or top_dir.with_suffix(".py").exists()):
            continue  # 외부 패키지
        candidate = workspace_root / "/".join(parts)
        if not (candidate.exists() or candidate.with_suffix(".py").exists()
                or (candidate / "__init__.py").exists()):
            issues.append(f"[IMPORT] {imp_file}: '{module_path}' 모듈 미존재")

    if not issues:
        return "✓ 일관성 검증 통과: 문제 없음"
    return f"⚠ {len(issues)}건의 일관성 문제 발견:\n" + "\n".join(issues)


if __name__ == "__main__":
    print(f"통합 MCP 서버 시작: http://0.0.0.0:{_PORT}/mcp")
    print(f"Workspace: {_WORKSPACE}")
    print("도구: 코드(12) + 검색(4) = 16개")
    mcp.run(transport="streamable-http")
