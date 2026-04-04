# AI Assistant Coding Agent Harness

MCP 도구 기반의 코드 생성/검증/실행 에이전트 프레임워크.
Claude Code & Codex 아키텍처 분석을 반영한 프로덕션급 Coding AI Agent.

## 아키텍처

```
사용자 요청 → [parse] → [execute: LLM + MCP 도구] ⇄ [tools] → [verify] → 결과
                              ↑ ReAct 루프 ↑
                         read_file, search_code,
                         list_directory, run_python,
                         write_file, str_replace,
                         apply_patch
```

### 핵심 구성

| 모듈 | 역할 |
|------|------|
| `core/` | 공통 프레임워크 (BaseGraphAgent, MCP Loader, 멀티티어 모델) |
| `core/context_manager.py` | 자동 컨텍스트 컴팩션 + max_tokens 복구 |
| `core/parallel_tool_executor.py` | 병렬 도구 실행기 (concurrency-safe 분류) |
| `core/project_context.py` | 프로젝트 컨텍스트 자동 발견 + 시스템 프롬프트 주입 |
| `core/tool_permissions.py` | 계층적 도구 권한 모델 (ALLOW/ASK/DENY) |
| `agents/` | 에이전트 구현체 (CodingAssistant, DeepResearch, SimpleReAct, Orchestrator) |
| `a2a_local/` | A2A 프로토콜 통합 (a2a-sdk 네이밍 충돌 방지) |
| `mcp_servers/code_tools/` | 파일 I/O, 코드 검색, 코드 실행, str_replace, apply_patch |
| `eval_pipeline/` | DeepEval 기반 평가 파이프라인 (Langfuse 연동) |
| `cli/` | 대화형 CLI (prompt-toolkit + rich, 토큰 스트리밍) |
| `docker/` | Docker Compose Harness |

### 설계 원칙

- **P1** (Agent-as-a-Judge): 최소 구조 — parse → execute → verify
- **P2** (RubricRewards): Generator/Verifier 모델 분리 (STRONG/DEFAULT 티어)
- **P3** (AutoHarness): Safety Envelope — 도구 권한 + workspace 경로 제한
- **P5** (GAM): JIT 원본 참조 — MCP 도구로 프로젝트 컨텍스트 수집
- **Claude Code 패턴**: 자동 컨텍스트 컴팩션, 병렬 도구 실행, 프로젝트 메모리
- **Codex 패턴**: Unified diff 패치, 선택적 히스토리 전파, 계층적 권한

## 빠른 시작

```bash
# 의존성 설치
uv sync

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENROUTER_API_KEY 입력

# MCP 서버 기동 (Docker)
cd docker && docker compose -f docker-compose.mcp.yml up -d && cd ..

# 대화형 CLI 실행
youngs75-agent
```

### CLI 사용법

```bash
# 코드 생성 요청
[coding_assistant] > Python으로 간단한 계산기를 만들어줘

# 에이전트 전환
[coding_assistant] > /agents          # 목록 보기
[coding_assistant] > /agent deep_research

# 도움말
[coding_assistant] > /help
```

### 기타 실행 방법

```bash
# 전체 테스트
make test

# eval 테스트
make test-eval

# lint
make lint

# Docker 전체 기동 (환경별)
make up-dev
make up-staging
make up-prod

# Langfuse 기동
make langfuse-up
```

## 멀티티어 모델 (OpenRouter 오픈소스)

| 티어 | 용도 | 모델 | 비용 (per 1M tokens) |
|------|------|------|---------------------|
| **STRONG** | 코드 생성 | `qwen/qwen3-coder` | $0.22 / $1.00 |
| **DEFAULT** | 검증/추론 | `deepseek/deepseek-v3.2` | $0.26 / $0.38 |
| **FAST** | 파싱/경량 | `qwen/qwen3.5-9b` | $0.05 / $0.15 |

환경변수로 오버라이드 가능:
```bash
STRONG_MODEL=qwen/qwen3-coder
STRONG_PROVIDER=openrouter
DEFAULT_MODEL=deepseek/deepseek-v3.2
FAST_MODEL=qwen/qwen3.5-9b
```

## 프로덕션 기능 (Phase 10)

### 자동 컨텍스트 컴팩션
- 토큰 사용량 80% 초과 시 LLM 기반 자동 히스토리 요약
- `max_tokens` 응답 시 최대 3회 자동 복구
- 서브에이전트 호출 시 선택적 히스토리 전파 (LastNTurns)

### 도구 병렬 실행
- 읽기 도구 (`read_file`, `search_code` 등) 동시 실행
- 쓰기/실행 도구는 독점 순차 실행
- 결과 순서 보장

### Diff 기반 파일 수정
- `str_replace`: 정확한 문자열 교체 (모호성 방지)
- `apply_patch`: Unified diff 형식 패치 적용

### 프로젝트 컨텍스트 주입
- `.agent/context.md`, `AGENTS.md` 자동 발견
- 시스템 프롬프트에 동적 주입

### 도구 권한 모델
- 읽기(ALLOW), 쓰기(ALLOW), 실행(ASK), 삭제(ASK)
- 민감 파일 패턴 감지 (.env, credentials 등)
- 프로젝트별 설정 오버라이드 (`.agent/permissions.yaml`)

## 기술 스택

- **A2A SDK** 0.3.25 — Agent-to-Agent 프로토콜
- **LangGraph** — 상태 그래프 기반 에이전트 오케스트레이션
- **LangChain** — LLM 추상화
- **MCP** (langchain-mcp-adapters) — Model Context Protocol 도구 연동
- **OpenRouter** — 오픈소스 LLM 게이트웨이 (Qwen, DeepSeek)
- **Langfuse** v4 — 관측성 + 실험 파이프라인
- **DeepEval** — LLM 평가 메트릭
- **Docker Compose** — MCP 서버 + 인프라 배포
- **GitHub Actions** — CI/CD (lint → test → docker build)

## 프로젝트 구조

```
youngs75_coding_ai_agent/
├── core/                      # 공통 프레임워크
│   ├── context_manager.py     # 자동 컨텍스트 컴팩션
│   ├── parallel_tool_executor.py  # 병렬 도구 실행기
│   ├── project_context.py     # 프로젝트 컨텍스트 로더
│   ├── tool_permissions.py    # 도구 권한 모델
│   ├── model_tiers.py         # 멀티티어 모델 해석
│   ├── batch_executor.py      # 비동기 배치 실행기
│   └── memory/                # 메모리 시스템
├── agents/                    # 에이전트 구현체
│   ├── coding_assistant/      # CodingAssistant (parse → execute → verify)
│   ├── deep_research/         # DeepResearch (clarify → brief → research → report)
│   ├── orchestrator/          # Orchestrator (에이전트 선택 + 위임)
│   └── simple_react/          # SimpleReAct (MCP 도구 루프)
├── a2a_local/                 # A2A 프로토콜 통합
├── mcp_servers/code_tools/    # MCP 서버 (7개 도구)
├── eval_pipeline/             # DeepEval 평가 파이프라인
├── cli/                       # 대화형 CLI
├── docker/                    # Docker Compose
├── scripts/                   # 실행 스크립트
├── tests/                     # 테스트 (571+ passed)
├── docs/                      # 문서 (Architecture, API, User Guide)
├── config/                    # 환경별 설정 (dev/staging/prod)
├── .ai/analysis/              # Claude Code & Codex 분석 보고서
├── .ai/sessions/              # 세션 인수인계 파일
├── Makefile                   # 개발 편의 명령어
└── .github/workflows/         # CI/CD (ci.yml, cd.yml)
```
