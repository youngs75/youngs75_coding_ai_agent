# AI Assistant Coding Agent Harness

MCP 도구 기반의 코드 생성/검증/실행 에이전트 프레임워크.

## 아키텍처

```
사용자 요청 → [parse] → [execute: LLM + MCP 도구] ⇄ [tools] → [verify] → 결과
                              ↑ ReAct 루프 ↑
                         read_file, search_code,
                         list_directory, run_python,
                         write_file
```

### 핵심 구성

| 모듈 | 역할 |
|------|------|
| `core/` | 공통 프레임워크 (BaseGraphAgent, MCP Loader) |
| `agents/` | 에이전트 구현체 (CodingAssistant, DeepResearch, Orchestrator) |
| `a2a/` | A2A 프로토콜 통합 |
| `mcp_servers/code_tools/` | 파일 I/O, 코드 검색, 코드 실행 MCP 서버 |
| `eval_pipeline/` | DeepEval 기반 평가 파이프라인 (Langfuse 연동) |
| `docker/` | Docker Compose Harness (12개 서비스) |

### 설계 원칙 (논문 7편 기반)

- **P1** (Agent-as-a-Judge): 최소 구조 — parse → execute → verify
- **P2** (RubricRewards): Generator/Verifier 모델 분리
- **P3** (AutoHarness): Safety Envelope
- **P5** (GAM): JIT 원본 참조 — MCP 도구로 프로젝트 컨텍스트 수집

## 빠른 시작

```bash
# 의존성 설치
uv sync

# 환경변수 설정
cp .env.example .env
# .env 파일에 API 키 입력

# MCP 서버 실행
python -m youngs75_a2a.tests.run_code_tools_mcp &

# 에이전트 테스트
python -m youngs75_a2a.tests.test_coding_assistant 1

# Langfuse 실험
python -m youngs75_a2a.scripts.10_run_langfuse_experiment --run-name "test"
```

## 기술 스택

- **A2A SDK** 0.3.25 — Agent-to-Agent 프로토콜
- **LangGraph** — 상태 그래프 기반 에이전트 오케스트레이션
- **MCP** — Model Context Protocol 도구 연동
- **Langfuse** v4 — 관측성 + 실험 파이프라인
- **DeepEval** — LLM 평가 메트릭
