<!-- Parent: ../AGENTS.md -->

# middleware

## Purpose
LLM 호출 전후에 메시지를 가공하는 미들웨어 체인. 컨텍스트 윈도우 관리, LLM 기반 요약, 스킬/메모리 주입 등을 처리한다.

## Key Files
| File | Description |
|------|-------------|
| `chain.py` | `MiddlewareChain` — 미들웨어 순차 실행 (양파 패턴) |
| `base.py` | `AgentMiddleware` 추상 클래스 + `ModelRequest`/`ModelResponse` |
| `summarization.py` | **LLM 기반 대화 요약** — 110K 초과 시 DEFAULT 모델로 중복/반복 압축, 핵심 정보 보존 |
| `message_window.py` | **토큰 기반 3단계 컴팩션** — 마이크로(도구 결과 cleared) → 윈도우(에러 우선 보존) → 긴급(FIFO) |
| `skill.py` | 스킬 컨텍스트 주입 미들웨어 |
| `memory.py` | 메모리 컨텍스트 주입 미들웨어 |
| `resilience.py` | 재시도/폴백/AbortController 미들웨어 |

## For AI Agents

### 실행 순서 (양파 패턴, 바깥→안쪽)
```
Resilience → Summarization(110K, LLM) → MessageWindow(100K, 규칙) → Memory → LLM
```

**순서가 중요**: Summarization이 MessageWindow보다 먼저 실행되어야 LLM 요약이 트리거됨. 순서가 뒤집히면 규칙 기반이 먼저 잘라내어 LLM 요약이 영원히 동작하지 않음.

### 설계 원칙 (Claude Code/Codex/DeepAgents 분석 반영)
- 토큰 기반 트리거 (메시지 수가 아님)
- LLM 요약이 규칙 기반보다 우선
- 에러 메시지 우선 보존
- 128K 모델 컨텍스트 충분 활용
