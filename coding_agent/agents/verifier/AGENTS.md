<!-- Parent: ../AGENTS.md -->

# verifier

## Purpose
코드 검증 에이전트. 생성된 코드의 정확성, 안전성, 스타일을 검증한다.

## Key Files
| File | Description |
|------|-------------|
| `agent.py` | `VerificationAgent` — 코드 검증 그래프 |
| `config.py` | 검증 설정 (허용 확장자, 삭제 라인 제한 등) |
| `prompts.py` | 검증 시스템 프롬프트 (`VERIFY_SYSTEM_PROMPT`) |
| `schemas.py` | 검증 결과 스키마 |

## For AI Agents
- DEFAULT 티어 모델 사용
- CodingAssistant의 4노드 그래프에서는 정적 검증만 사용 (LLM 검증은 비활성)
- 독립 실행 시 LLM 기반 검증 수행
