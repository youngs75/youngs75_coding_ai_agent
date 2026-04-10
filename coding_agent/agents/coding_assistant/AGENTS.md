<!-- Parent: ../AGENTS.md -->

# coding_assistant

## Purpose
핵심 코딩 에이전트. 코드 생성 → 테스트 실행 → 에러 주입 → 재생성 루프를 4노드 그래프로 수행한다.

## Key Files
| File | Description |
|------|-------------|
| `agent.py` | `CodingAssistantAgent` — 4노드 그래프 (RETRIEVE_MEMORY → GENERATE → RUN_TESTS → INJECT_ERROR) |
| `config.py` | `CodingConfig` — 모델 티어, 최대 반복, 도구 제한 등 설정 |
| `prompts.py` | 시스템 프롬프트 (`GENERATE_FINAL_SYSTEM_PROMPT`, `EXECUTE_SYSTEM_PROMPT`) |
| `schemas.py` | `CodingState` — LangGraph 상태 스키마 |

## Architecture
```
RETRIEVE_MEMORY → GENERATE → RUN_TESTS → [END or INJECT_ERROR → GENERATE] × 3
```
- LLM 1회/iteration (STRONG 모델이 write_file 도구로 직접 파일 저장)
- Harness는 기계적 도구 제공자, 판단은 LLM에게 위임 (SLM 환경 원칙)
- 미들웨어 체인: Resilience → Summarization(110K, DEFAULT LLM) → MessageWindow(100K, 규칙) → Memory
- 에러 주입 시 traceback 참조 파일 내용을 함께 전달 → LLM이 불일치를 직접 확인

## For AI Agents
- `_generate_code()` — `_generate_final()` 호출 + 마크다운 폴백 + 정적 검증 통합
- `_inject_error()` — 에러 원문만 전달, 분류/힌트 없음
- `_run_tests()` — 언어 자동 감지 + 의존성 설치 + 테스트 실행
- `_should_retry_tests()` — 그래프 라우터 (통과 → END, 실패 → INJECT_ERROR)
