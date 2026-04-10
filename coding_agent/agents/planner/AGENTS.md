<!-- Parent: ../AGENTS.md -->

# planner

## Purpose
아키텍처 설계 + 태스크 분해 에이전트. REASONING 모델로 구현 계획을 수립한다.

## Key Files
| File | Description |
|------|-------------|
| `agent.py` | `PlannerAgent` — 계획 수립 + 태스크 분해 |
| `config.py` | Planner 설정 (REASONING 모델 지정) |
| `prompts.py` | 계획 수립 시스템 프롬프트 |
| `schemas.py` | 계획/태스크 스키마 |

## For AI Agents
- REASONING 티어 모델 사용 (qwen-max)
- 출력: 파일 목록, 구현 순서, 파일 간 의존성 그래프
- Orchestrator에서 HITL interrupt 후 사용자 승인을 받아 실행
