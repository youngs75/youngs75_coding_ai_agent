<!-- Parent: ../../AGENTS.md -->

# code_tools

## Purpose
MCP 서버 — 파일 I/O, 코드 검색, 코드 실행 도구를 제공한다.

## Key Files
| File | Description |
|------|-------------|
| `server.py` | MCP 서버 구현 — `read_file`, `write_file`, `search_code`, `run_command` 등 |

## For AI Agents
- 에이전트가 MCP 프로토콜로 이 서버의 도구를 호출
- `write_file` — workspace 내 파일 저장 (보안: workspace 밖 쓰기 금지)
- `run_command` — 코드 실행/테스트 (타임아웃 보호)
- Docker에서 독립 컨테이너로 실행, `CODE_TOOLS_WORKSPACE` 환경변수로 작업 디렉토리 지정
