<!-- Parent: ../../AGENTS.md -->

# cli

## Purpose
대화형 CLI 인터페이스. prompt-toolkit + rich 기반으로 사용자와 에이전트 간 상호작용을 제공한다.

## Key Files
| File | Description |
|------|-------------|
| `app.py` | CLI 메인 루프 — 사용자 입력 → Orchestrator 호출 → 응답 렌더링 |
| `session.py` | CLI 세션 관리 (session_id 생성/유지) |
| `commands.py` | 슬래시 명령어 (`/memory`, `/clear`, `/export` 등) |
| `renderer.py` | rich 기반 응답 렌더링 |
| `config.py` | CLI 설정 |
| `eval_runner.py` | CLI에서 평가 파이프라인 실행 |

## For AI Agents
- `session.py`의 `session_id`는 Langfuse 세션 추적에도 사용됨
- CLI → Orchestrator → SubAgent 경로로 동작
- HITL interrupt 시 CLI가 사용자 승인을 받아 `Command(resume=True)` 전달
