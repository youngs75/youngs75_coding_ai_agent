<!-- Parent: ../../AGENTS.md -->

# utils

## Purpose
프로젝트 전역 유틸리티. 로깅, 환경변수, 프로파일링, E2E 세션 관리 등 공통 기능을 제공한다.

## Key Files
| File | Description |
|------|-------------|
| `e2e_session.py` | E2E 세션 ID 관리 — `HARNESS_SESSION_ID` 환경변수 기반, Langfuse 세션 추적용 |
| `logging.py` | 구조화 로깅 설정 |
| `env.py` | 환경변수 로더 |
| `file_io.py` | 파일 I/O 유틸리티 |
| `token_optimizer.py` | 토큰 사용량 최적화 |
| `timing.py` | 실행 시간 측정 |
| `profiler.py` | 성능 프로파일러 |
| `llm_cache.py` | LLM 응답 캐시 |

## For AI Agents
- 반복 작업(Langfuse 조회, 벤치마크 비교 등)은 이 디렉토리에 유틸리티로 추가
- 새 유틸리티는 CLI에서 직접 실행 가능하거나 import 가능한 함수 형태로 작성
