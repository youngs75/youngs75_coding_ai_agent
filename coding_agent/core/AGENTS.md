<!-- Parent: ../../AGENTS.md -->

# core

## Purpose
에이전트 공통 프레임워크. 모든 에이전트가 공유하는 기반 클래스, 안전장치, LLM 추상화를 제공한다.

## Key Files
| File | Description |
|------|-------------|
| `base_agent.py` | `BaseGraphAgent` — LangGraph 기반 에이전트 공통 베이스 |
| `model_tiers.py` | 4-Tier 모델 체계 (`create_model()`) + LiteLLM Proxy/SDK 분기 |
| `mcp_loader.py` | MCP 도구 로더 (langchain-mcp-adapters 연동) |
| `context_manager.py` | 컨텍스트 윈도우 관리 + 컴팩션 |
| `stall_detector.py` | 도구 호출 반복 패턴 감지 (StallDetector) |
| `turn_budget.py` | 턴 단위 토큰 예산 추적 (TurnBudgetTracker) |
| `abort_controller.py` | 비동기 중단 제어 |
| `tool_call_utils.py` | LLM 응답 tool_calls 정규화 유틸리티 |
| `tool_permissions.py` | 도구 실행 권한 검증 |
| `resilience.py` | 재시도/타임아웃 래퍼 |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `memory/` | CoALA 4종 메모리 시스템 (see `memory/AGENTS.md`) |
| `skills/` | 3-Level 스킬 시스템 (see `skills/AGENTS.md`) |
| `middleware/` | LLM 미들웨어 체인 (see `middleware/AGENTS.md`) |
| `subagents/` | SubAgent 동적 레지스트리 (see `subagents/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- 이 디렉토리는 **모든 에이전트의 공통 기반**이므로, 변경 시 전체 에이전트에 영향이 감
- `model_tiers.py` 수정 시 Docker(LiteLLM Proxy)와 로컬(SDK 직접) 양쪽 경로 모두 확인
- 안전장치(`stall_detector`, `turn_budget`, `abort_controller`)는 서로 독립적으로 동작

### Testing Requirements
- `pytest tests/test_model_tiers.py` — 모델 생성/티어 매핑
- `pytest tests/test_stall_detector.py` — 반복 감지
- `pytest tests/test_context_manager.py` — 컨텍스트 관리
