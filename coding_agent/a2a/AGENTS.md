<!-- Parent: ../../AGENTS.md -->

# a2a

## Purpose
A2A(Agent-to-Agent) 프로토콜 브릿지. 에이전트를 A2A 서버로 노출하고 외부 에이전트와 통신한다.

## Key Files
| File | Description |
|------|-------------|
| `server.py` | A2A 서버 (Starlette + Uvicorn) |
| `executor.py` | A2A 요청 실행기 — LangGraph 에이전트를 A2A 태스크로 래핑 |
| `router.py` | 요청 라우팅 |
| `streaming.py` | SSE 스트리밍 응답 |
| `discovery.py` | Agent Card 디스커버리 |
| `resilience.py` | A2A 통신 재시도/타임아웃 |

## For AI Agents
- A2A SDK 0.3.25 기반
- Docker Compose에서 각 에이전트가 독립 A2A 서버로 배포됨
- `executor.py`가 LangGraph `astream_events`를 A2A `SendTaskResponse`로 변환
