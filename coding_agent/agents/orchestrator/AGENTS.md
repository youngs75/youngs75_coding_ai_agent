<!-- Parent: ../AGENTS.md -->

# orchestrator

## Purpose
진입점 에이전트. 사용자 요청을 분류하고 Planner를 경유하여 적절한 SubAgent로 라우팅한다.

## Key Files
| File | Description |
|------|-------------|
| `agent.py` | `OrchestratorAgent` — 요청 분류 + HITL 계획 승인 + SubAgent 위임 |
| `config.py` | Orchestrator 설정 |
| `coordinator.py` | SubAgent 조율 로직 |
| `task_graph.py` | 태스크 의존성 그래프 |
| `schemas.py` | Orchestrator 상태 스키마 |

## For AI Agents
- HITL 흐름: `classify → plan → [interrupt: 승인 대기] → delegate → respond`
- `skill_registry`를 하위 에이전트에 전달해야 스킬 시스템이 활성화됨
- checkpointer(MemorySaver)가 필수 — interrupt/resume에 사용
