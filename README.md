# AI Assistant Coding Agent Harness

MCP 도구 기반의 코드 생성/검증/실행 에이전트 프레임워크.
Claude Code & Codex 아키텍처 분석을 반영한 프로덕션급 Coding AI Agent.

## 아키텍처

```
사용자 요청 → [Orchestrator] → 에이전트 선택 → [Subagent 실행] → 결과

Orchestrator ─┬─→ CodingAssistant  (코드 생성/수정/리팩토링)
              ├─→ DeepResearch     (심층 조사/기술 분석)
              ├─→ SimpleReAct      (파일 조회/간단한 질의응답)
              └─→ Coordinate       (복합 작업 병렬 오케스트레이션)
```

### CodingAssistant 2단계 파이프라인

```
[parse] → [retrieve_memory] → [execute: FAST + MCP 도구] ⇄ [tools]
                                        │ 도구 호출 없으면
                                        ↓
                              [generate_final: STRONG] → [verify] → 결과
```

- **1단계 (FAST)**: 저비용 모델로 도구 호출 판단 + ReAct 루프
- **2단계 (STRONG)**: 고성능 모델로 최종 코드 생성
- ReAct 루프 비용 약 **90% 절감** (FAST $0.07 vs STRONG $0.65 per 1M tokens)

### 핵심 구성

| 모듈 | 역할 |
|------|------|
| `core/` | 공통 프레임워크 (BaseGraphAgent, MCP Loader, 멀티티어 모델) |
| `core/skills/` | 3-Level Progressive Loading 스킬 시스템 + task_type 기반 자동 활성화 |
| `core/memory/` | CoALA 4종 메모리 (Working/Episodic/Semantic/Procedural) |
| `core/context_manager.py` | 자동 컨텍스트 컴팩션 + max_tokens 복구 |
| `core/parallel_tool_executor.py` | 병렬 도구 실행기 (concurrency-safe 분류) |
| `core/project_context.py` | 프로젝트 컨텍스트 자동 발견 + 시스템 프롬프트 주입 |
| `core/tool_permissions.py` | 계층적 도구 권한 모델 (ALLOW/ASK/DENY) + 금지 경로 차단 |
| `agents/` | 에이전트 구현체 (Orchestrator, CodingAssistant, DeepResearch, SimpleReAct) |
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
git clone <repo-url> && cd youngs75_coding_ai_agent
make setup        # Python 3.13 + 의존성 + .env + MCP 서버 — 한번에 완료
youngs75-agent    # CLI 실행
```

`make setup`이 자동으로 처리하는 것:
1. **uv** 패키지 매니저 확인/설치
2. **Python 3.13** 확인 (없으면 `uv python install 3.13`)
3. **의존성** 설치 (`uv sync`)
4. **.env** 파일 생성 + 필수 API 키 안내
5. **MCP 서버** Docker 기동 (선택)

> 필수 API 키는 **OPENROUTER_API_KEY** 하나뿐입니다. ([발급](https://openrouter.ai/keys))

### CLI 사용법

CLI 시작 시 기본 에이전트는 **Orchestrator**이며, 사용자 요청에 따라 적합한 Subagent로 자동 라우팅합니다.

```bash
# 시작 시 표시:
# ✓ 스킬 7개 로드: code_review, commit, debug, explain, ...
# ✓ 프로젝트 컨텍스트 로드 완료

# 코드 생성 요청 → Orchestrator → CodingAssistant로 자동 위임
[orchestrator] > Python으로 간단한 계산기를 만들어줘
#   ⇢ 위임: coding_assistant
#   ⚙ 스킬 활성화: code_review, refactor, test_generation
#   ⠋ 도구 호출 판단 (FAST)
#   ⠋ 코드 생성 (STRONG)
#   ✓ 검증 통과
#   ⏱ 12.3s

# 리서치 요청 → DeepResearch로 자동 위임
[orchestrator] > LangGraph와 CrewAI의 차이점을 조사해줘
#   ⇢ 위임: deep_research

# 파일 조회 → SimpleReAct로 자동 위임
[orchestrator] > 프로젝트 루트에 어떤 파일이 있는지 알려줘
#   ⇢ 위임: simple_react

# 특정 에이전트 직접 사용
[orchestrator] > /agent coding_assistant

# 도움말
[orchestrator] > /help

# 스킬 관리
[orchestrator] > /skill list
[orchestrator] > /skill activate security_review
```

### 주요 명령어

```bash
make setup       # 초기 설정 (최초 1회)
make mcp-up      # MCP 도구 서버 기동
make mcp-down    # MCP 도구 서버 종료
make test        # 테스트 실행
make lint        # 린트 검사
make help        # 전체 명령어 목록
```

## 멀티티어 모델 (OpenRouter 오픈소스)

전 티어 Qwen 시리즈 통일. 목적별 최적 모델을 자동 선택합니다.

| 티어 | 용도 | 모델 | 비용 (per 1M tokens) | 컨텍스트 |
|------|------|------|---------------------|----------|
| **STRONG** | 최종 코드 생성 | `qwen/qwen3-coder-plus` | $0.65 / $3.25 | 1M |
| **DEFAULT** | 검증/추론 | `qwen/qwen3-coder-next` | $0.12 / $0.75 | 262K |
| **FAST** | 도구 호출 판단/파싱 | `qwen/qwen3.5-flash-02-23` | $0.07 / $0.26 | 1M |

### 목적별 모델 매핑 (purpose → tier)

| purpose | tier | 설명 |
|---------|------|------|
| `generation` | STRONG | 최종 코드 생성 (generate_final 노드) |
| `tool_planning` | FAST | 도구 호출 판단 + ReAct 루프 (execute_code 노드) |
| `verification` | DEFAULT | 코드 검증 (verify_result 노드) |
| `parsing` | FAST | 요청 분석 (parse_request 노드) |

환경변수로 오버라이드 가능:
```bash
STRONG_MODEL=qwen/qwen3-coder-plus
DEFAULT_MODEL=qwen/qwen3-coder-next
FAST_MODEL=qwen/qwen3.5-flash-02-23
PURPOSE_TIERS='{"generation":"strong","tool_planning":"fast","verification":"default"}'
```

## Skills 시스템

### 자동 활성화

parse 결과의 `task_type`에 따라 관련 스킬이 **자동으로 L2 활성화**됩니다. 수동 활성화(`/skill activate`)도 가능합니다.

| task_type | 자동 활성화 스킬 | 용도 |
|-----------|-----------------|------|
| `generate` | code_review, refactor, test_generation | 코드 품질 + 테스트 가이드 |
| `fix` | debug | 에러 진단 + 수정 방안 |
| `refactor` | refactor | 리팩토링 원칙 |
| `explain` | explain | 코드 설명 가이드 |
| `analyze` | code_review, security_review | 보안 + 품질 리뷰 |

### 3-Level Progressive Loading

- **L1 (메타데이터)**: name, description, tags — 항상 컨텍스트에 주입
- **L2 (본문)**: 프롬프트 body — 활성화 시 로드, 시스템 프롬프트에 주입
- **L3 (참조)**: 외부 파일 — 실행 시 로드

### 등록된 스킬 (7개)

| 스킬 | 설명 | 태그 |
|------|------|------|
| `code_review` | 품질/보안/성능 코드 리뷰 | review, quality, security |
| `debug` | 에러 진단 + 수정 | debug, fix, error |
| `refactor` | 코드 리팩토링 | refactor, quality, cleanup |
| `explain` | 코드 설명 (한국어) | explain, documentation |
| `test_generation` | pytest 테스트 자동 생성 | testing, pytest, quality |
| `security_review` | OWASP Top 10 보안 리뷰 | security, owasp, audit |
| `commit` | Conventional Commits 메시지 생성 | git, commit, workflow |

## 프로덕션 기능

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
- 금지 경로 자동 DENY: `.claude/`, `.git/`, `__pycache__/`, `node_modules/`
- 민감 파일 패턴 감지 (.env, credentials 등)

### Semantic Memory (AGENTS.md 자동 로딩)
- `SemanticMemoryLoader`가 AGENTS.md + pyproject.toml에서 프로젝트 규칙/기술스택 자동 추출
- 커뮤니케이션 규칙, 커밋 규칙, 개발 규칙, 기술스택, 프로젝트 구조 5개 섹션

### Langfuse 관측성
- LLM 호출 트레이싱 (노드별 토큰 사용량, 소요시간)
- 에이전트 메트릭 수집 (`AgentMetricsCollector`)
- CLI 레벨 토글: `CLI_LANGFUSE_ENABLED=1`

## 기술 스택

- **A2A SDK** 0.3.25 — Agent-to-Agent 프로토콜
- **LangGraph** — 상태 그래프 기반 에이전트 오케스트레이션
- **LangChain** 1.2.13+ — LLM 추상화
- **MCP** (langchain-mcp-adapters 0.2.2) — Model Context Protocol 도구 연동
- **OpenRouter** — 오픈소스 LLM 게이트웨이 (Qwen 시리즈)
- **Pydantic Settings** 2.2+ — 설정/스키마 검증
- **Langfuse** v4 — 관측성 + 실험 파이프라인
- **DeepEval** — LLM 평가 메트릭
- **Starlette** + **Uvicorn** — ASGI 서버
- **Docker Compose** — MCP 서버 + 인프라 배포
- **GitHub Actions** — CI/CD (lint → test → docker build)

## 프로젝트 구조

```
youngs75_coding_ai_agent/
├── core/                      # 공통 프레임워크
│   ├── config.py              # BaseAgentConfig (모델 팩토리)
│   ├── base_agent.py          # BaseGraphAgent (Template Method)
│   ├── model_tiers.py         # 멀티티어 모델 해석 + 비용 분석
│   ├── context_manager.py     # 자동 컨텍스트 컴팩션
│   ├── parallel_tool_executor.py  # 병렬 도구 실행기
│   ├── project_context.py     # 프로젝트 컨텍스트 로더
│   ├── tool_permissions.py    # 도구 권한 모델 + 금지 경로
│   ├── mcp_loader.py          # MCP 도구 로딩/캐싱
│   ├── memory/                # CoALA 4종 메모리 시스템
│   ├── skills/                # 3-Level 스킬 시스템 (자동 활성화)
│   └── subagents/             # SubAgent 동적 선택 레지스트리
├── agents/                    # 에이전트 구현체
│   ├── coding_assistant/      # CodingAssistant (2단계 파이프라인)
│   ├── deep_research/         # DeepResearch (clarify → brief → research → report)
│   ├── orchestrator/          # Orchestrator (기본 에이전트, 자동 라우팅 + 복합 작업)
│   └── simple_react/          # SimpleReAct (MCP 도구 루프)
├── a2a_local/                 # A2A 프로토콜 통합
├── mcp_servers/code_tools/    # MCP 서버 (7개 도구)
├── eval_pipeline/             # DeepEval 평가 파이프라인
├── data/skills/               # 스킬 YAML 정의 (7개)
├── cli/                       # 대화형 CLI
├── docker/                    # Docker Compose
├── scripts/                   # 실행 스크립트
├── tests/                     # 테스트 (750+ passed)
├── docs/                      # 문서 (Architecture, API, User Guide)
├── config/                    # 환경별 설정 (dev/staging/prod)
├── .ai/analysis/              # Claude Code & Codex 분석 보고서
├── .ai/sessions/              # 세션 인수인계 파일
├── Makefile                   # 개발 편의 명령어
└── .github/workflows/         # CI/CD (ci.yml, cd.yml)
```
