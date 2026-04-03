"""Code Tools MCP 서버 (로컬 실행).

Coding Agent Harness에 파일 I/O, 코드 검색, 코드 실행 도구를 제공한다.

실행: python -m youngs75_a2a.tests.run_code_tools_mcp
포트: 3003 (환경변수 CODE_TOOLS_PORT로 변경 가능)
"""

import os
import sys

sys.path.insert(0, ".")

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass

# workspace를 프로젝트 루트로 설정
os.environ.setdefault("CODE_TOOLS_WORKSPACE", os.getcwd())

from youngs75_a2a.mcp_servers.code_tools.server import mcp  # noqa: E402

if __name__ == "__main__":
    port = int(os.getenv("CODE_TOOLS_PORT", "3003"))
    workspace = os.getenv("CODE_TOOLS_WORKSPACE", os.getcwd())
    print(f"Code Tools MCP 서버 시작: http://0.0.0.0:{port}/mcp")
    print(f"Workspace: {workspace}")
    mcp.run(transport="streamable-http")
