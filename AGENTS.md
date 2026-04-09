# Repository Guidelines

## 프로젝트 개요
AI Assistant Coding Agent Harness — MCP 도구 기반 코드 생성/검증/실행 에이전트 프레임워크.
최종 산출물: A2A 프로토콜 통합 Coding Agent Harness

## 프로젝트 구조

```
youngs75_coding_ai_agent/           # 패키지명: coding_agent
├── core/                           # 공통 프레임워크 (BaseGraphAgent, MCP Loader)
│   ├── memory/                     # CoALA 4종 메모리 시스템
│   ├── skills/                     # 3-Level 스킬 시스템 (자동 활성화)
│   ├── middleware/                  # LLM 미들웨어 체인
│   └── subagents/                  # SubAgent 동적 선택 레지스트리
├── agents/                         # 에이전트 구현체
│   ├── orchestrator/               # Orchestrator (기본 에이전트, Planner 경유 라우팅)
│   ├── planner/                    # Planner (아키텍처 설계 + 태스크 분해)
│   ├── coding_assistant/           # CodingAssistant (코드 생성 + 자동 파일 저장)
│   └── verifier/                   # VerificationAgent (코드 검증)
├── a2a_local/                      # A2A 프로토콜 브릿지 (a2a-sdk 충돌 방지)
├── mcp_servers/                    # MCP 서버
│   └── code_tools/                 # 파일 I/O, 코드 검색, 코드 실행
├── eval_pipeline/                  # DeepEval 기반 평가 파이프라인 (Langfuse 연동)
├── cli/                            # 대화형 CLI (prompt-toolkit + rich)
├── scripts/                        # 스텝별 실행 엔트리포인트
├── tests/                          # 테스트 (960+ passed)
├── docs/                           # 문서 (Architecture, API, User Guide)
├── docker/                         # Docker Compose Harness (통합 구성)
├── pyproject.toml                  # 프로젝트 의존성 (uv 기반)
├── AGENTS.md                       # 이 파일 — AI와 기여자가 따를 규칙 문서
└── .ai/sessions/                   # 세션 인수인계 파일 저장 위치
```

규칙이 여러 곳에 흩어져 있어도 기준 문서는 항상 `AGENTS.md`로 통일합니다.

## 커뮤니케이션 규칙
사용자와의 모든 소통은 항상 한국어로 진행합니다. 작업 전 계획 공유, 진행 상황 보고, 작업 후 결과 요약도 모두 한국어로 작성합니다. 코드 주석은 기존 스타일을 따르되, 새로 작성할 때는 한국어를 우선 사용합니다.

## 세션 파일 명명 규칙
세션 파일은 `.ai/sessions/session-YYYY-MM-DD-NNNN.md` 형식을 사용합니다.

- `YYYY-MM-DD`: 세션 당일 날짜
- `NNNN`: 같은 날짜 내 순번 (`0001`부터 시작)
- 같은 날짜 파일이 있으면 가장 큰 번호에 `+1`을 적용합니다.

## Resume 규칙
사용자가 `resume` 또는 `이어서`라고 요청하면 가장 최근 세션 파일을 찾아 이어서 작업합니다.

- `.ai/sessions/`에서 명명 규칙에 맞는 파일만 후보로 봅니다.
- 가장 최신 날짜를 우선 선택하고, 같은 날짜면 가장 큰 순번을 선택합니다.
- 초기 컨텍스트에 파일이 없어 보여도 실제 파일 시스템을 다시 확인합니다.
- 세션 파일 조회 또는 읽기가 샌드박스 제한으로 실패하면, `.ai/sessions/` 확인과 대상 파일 읽기에 필요한 최소 범위에서 권한 상승을 요청한 뒤 즉시 재시도합니다.
- 권한 상승이 필요한 이유는 세션 복구를 위한 실제 파일 시스템 확인임을 사용자에게 짧게 알립니다.
- 선택한 세션 파일은 전체를 읽습니다.
- 사용자에게 이전 작업 내용과 다음 할 일을 한국어로 간단히 브리핑합니다.

## Handoff 규칙
새 세션 파일은 사용자가 명시적으로 종료를 요청한 경우에만 생성합니다. 허용 트리거 예시는 `handoff`, `정리해줘`, `세션 저장`, `종료하자`, `세션 종료`입니다.

- 저장 위치는 항상 `.ai/sessions/`입니다.
- 기존 `session-*.md` 파일은 절대 수정하지 않습니다.
- 자동 저장이나 단계별 저장은 하지 않습니다.
- 새 파일에는 프로젝트 개요, 최근 작업 내역, 현재 상태, 다음 단계, 중요 참고사항을 포함합니다.
- 저장 후 사용자에게 생성된 파일 경로를 알립니다.

## 개발 및 검증 규칙

### 환경 설정
```bash
# 초기 설정 (최초 1회 — Python 3.13 + 의존성 + .env + MCP)
make setup

# 이후 수동 설정이 필요한 경우
source .venv/bin/activate
uv sync
export $(grep -v '^#' .env | xargs)
```

### 테스트 실행
```bash
# 전체 테스트
pytest tests/

# 평가 테스트
pytest tests/eval/

# 개별 스크립트 검증
python scripts/run_pipeline.py --help
```

### Docker 배포
```bash
# Harness 기동 (LiteLLM + MCP + CodingAssistant + Orchestrator)
make up

# 상태 확인
make ps

# 종료
make down
```

구현 후에는 로그나 실행 결과로 정상 동작을 확인한 뒤에만 완료를 보고합니다.

## 커밋 및 PR 규칙
Conventional Commits 형식을 권장합니다. 예시: `feat: add coding agent scaffold`

- `.env` 파일, `.db` 파일, `.claude/` 디렉토리, `.deepeval/` 디렉토리는 커밋하지 않습니다.
- PR에는 변경 목적, 검증 방법, 보류 이슈를 포함합니다.

## Human-in-the-loop (HITL) 패턴
Planner Agent가 코딩 태스크에 대한 구현 계획을 수립하면, 사용자 승인을 받은 후 실행합니다.

- **구현**: LangGraph `interrupt()` + `aget_state()` 기반
- **흐름**: `classify → plan → [interrupt: 계획 표시 + 승인 대기] → delegate → respond`
- `astream_events`는 `GraphInterrupt`를 exception으로 전파하지 않으므로, 이벤트 스트림 종료 후 `aget_state()`로 pending interrupt를 감지합니다.
- 승인 시 `Command(resume=True)`로 그래프를 재개하고, 거부 시 실행을 중단합니다.
- checkpointer(MemorySaver/SqliteSaver)가 필수입니다.

## 4-Tier 모델 체계
| 티어 | 모델 (DashScope) | 용도 | 환경변수 |
|------|-----------------|------|----------|
| **REASONING** | qwen-max | 계획/아키텍처 설계 | `REASONING_MODEL` |
| **STRONG** | qwen-coder-plus | 코드 생성/도구 호출 | `STRONG_MODEL` |
| **DEFAULT** | qwen-plus | 검증/분석 | `DEFAULT_MODEL` |
| **FAST** | qwen-turbo | 파싱/분류 | `FAST_MODEL` |

- `LLM_PROVIDER=dashscope` — DashScope(Qwen 공식 API) 사용 (기본)
- `LLM_PROVIDER=openrouter` — OpenRouter 경유 사용
- DashScope 설정: `DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`

## 주요 기술 스택
- **A2A SDK** 0.3.25 — Agent-to-Agent 프로토콜
- **LangGraph** — 상태 그래프 기반 에이전트 오케스트레이션 (interrupt/Command HITL 포함)
- **LangChain** 1.2.13+ — LLM 추상화
- **MCP** (langchain-mcp-adapters 0.2.2) — Model Context Protocol 도구 연동
- **DashScope** — Qwen 공식 API (OpenAI 호환 모드)
- **Pydantic Settings** 2.2+ — 설정/스키마 검증
- **Langfuse** v4 — 관측성 + 실험 파이프라인
- **DeepEval** — LLM 평가 메트릭
- **Starlette** + **Uvicorn** — ASGI 서버
- **Docker Compose** — 프로덕션 배포
