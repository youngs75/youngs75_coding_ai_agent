<!-- Parent: ../AGENTS.md -->

# subagents

## Purpose
SubAgent 동적 생성/관리 레지스트리. 런타임에 태스크 성격에 따라 SubAgent를 생성하고 수명주기를 추적한다.

## Key Files
| File | Description |
|------|-------------|
| `registry.py` | `SubAgentRegistry` — SubAgent 등록/조회/해제 |
| `worker.py` | SubAgent 워커 실행 로직 |
| `process_manager.py` | SubAgent 프로세스 수명주기 관리 |
| `schemas.py` | SubAgent 상태 스키마 (`IDLE → RUNNING → COMPLETED/FAILED`) |

## For AI Agents
- SubAgent 상태 전이: `IDLE → RUNNING → COMPLETED | FAILED`
- Orchestrator가 `registry.spawn()`으로 동적 생성, 완료 후 `registry.release()`
- 최대 동시 실행 수는 설정으로 제한
