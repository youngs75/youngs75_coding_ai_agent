<!-- Parent: ../../AGENTS.md -->

# agents

## Purpose
에이전트 구현체 디렉토리. 각 에이전트는 `core/base_agent.py`의 `BaseGraphAgent`를 상속하며, LangGraph 상태 그래프로 동작한다.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `orchestrator/` | 진입점 에이전트 — 요청 분류 + Planner 경유 라우팅 (see `orchestrator/AGENTS.md`) |
| `planner/` | 아키텍처 설계 + 태스크 분해 (see `planner/AGENTS.md`) |
| `coding_assistant/` | 코드 생성 + 테스트 실행 + 자동 수정 루프 (see `coding_assistant/AGENTS.md`) |
| `verifier/` | 코드 검증 에이전트 (see `verifier/AGENTS.md`) |

## For AI Agents
- 새 에이전트 추가 시 `BaseGraphAgent`를 상속하고 `init_nodes()`, `init_edges()`를 구현
- 에이전트 간 통신은 Orchestrator를 경유하거나 A2A 프로토콜 사용
- 각 에이전트의 상세 구조는 하위 AGENTS.md 참조
