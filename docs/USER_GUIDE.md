# 사용자 가이드

**프로젝트**: youngs75-coding-ai-agent  
**패키지**: youngs75_a2a  
**Python**: 3.13 | **패키지 관리자**: uv

---

## 목차

1. [빠른 시작](#1-빠른-시작)
2. [CLI 사용법](#2-cli-사용법)
3. [Docker 배포](#3-docker-배포)
4. [에이전트 커스터마이징](#4-에이전트-커스터마이징)
5. [평가 파이프라인](#5-평가-파이프라인)
6. [FAQ](#6-faq)

---

## 1. 빠른 시작

### 1.1 사전 요구사항

- Python 3.13
- [uv](https://docs.astral.sh/uv/) 패키지 관리자
- OpenAI API 키 (또는 OpenRouter API 키)

### 1.2 설치

```bash
# 저장소 클론
git clone <repository-url>
cd youngs75_coding_ai_agent

# 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --all-groups
```

### 1.3 환경변수 설정

```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env
```

`.env` 파일을 열고 최소한 다음 항목을 설정한다:

```env
# 필수: DashScope (Qwen 공식 API) — 기본 프로바이더
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

# 4-Tier 모델 설정 (DashScope 기본값)
REASONING_MODEL=qwen-max           # 계획/아키텍처 설계
STRONG_MODEL=qwen-coder-plus       # 코드 생성/도구 호출
DEFAULT_MODEL=qwen-plus            # 검증/분석
FAST_MODEL=qwen-turbo              # 파싱/분류

# 대안: OpenRouter 경유 (LLM_PROVIDER=openrouter로 전환)
OPENROUTER_API_KEY=sk-or-...
```

### 1.4 MCP 서버 실행 (코딩 에이전트용)

코딩 에이전트는 MCP 서버를 통해 프로젝트 파일에 접근한다.

```bash
# Code Tools MCP 서버 실행 (백그라운드)
python -m youngs75_a2a.mcp_servers.code_tools.server &
```

기본 포트는 3003이다. `CODE_TOOLS_PORT` 환경변수로 변경 가능.

### 1.5 CLI 실행

```bash
# 방법 1: 설치된 스크립트로 실행
youngs75-agent

# 방법 2: 모듈로 실행
python -m youngs75_a2a.cli
```

실행하면 대화형 프롬프트가 표시된다:

```
┌─────────────────────────────────────────┐
│ AI Coding Agent Harness                  │
│ Active agent: coding_assistant           │
│ Type /help for commands, /quit to exit   │
└─────────────────────────────────────────┘
[coding_assistant] >
```

### 1.6 첫 번째 대화

```
[coding_assistant] > 파이썬으로 피보나치 함수를 작성해줘

  > 요청 분석 중...
  > 코드 생성 중...
Agent:
def fibonacci(n: int) -> int:
    """n번째 피보나치 수를 반환한다."""
    if n <= 0:
        raise ValueError("n은 양의 정수여야 합니다")
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

  > 코드 검증 중...

검증 통과
```

---

## 2. CLI 사용법

### 2.1 슬래시 커맨드

CLI에서 `/`로 시작하는 입력은 슬래시 커맨드로 처리된다.

```
/help              도움말 표시
/agent <name>      에이전트 전환
/agents            사용 가능한 에이전트 목록
/skill list        등록된 스킬 목록
/skill activate <name>  스킬 활성화
/history           최근 대화 기록
/history clear     대화 기록 초기화
/eval              평가 실행 (DeepEval)
/eval status       마지막 평가 결과
/eval remediate    Remediation 실행
/session           현재 세션 정보
/memory            메모리 상태
/clear             대화 기록 초기화
/quit              종료
```

### 2.2 에이전트 전환

4종류의 에이전트를 자유롭게 전환할 수 있다.

```
[coding_assistant] > /agents
사용 가능한 에이전트:
  - coding_assistant
  - deep_research
  - simple_react
  - orchestrator

[coding_assistant] > /agent deep_research
에이전트를 [deep_research]으로 전환했습니다.

[deep_research] > 양자 컴퓨팅의 최신 동향을 조사해줘
```

#### 에이전트별 특성

| 에이전트 | 용도 | MCP 도구 |
|----------|------|----------|
| `coding_assistant` | 코드 생성/수정/검증 | code_tools (파일 I/O, 검색, 실행) |
| `deep_research` | 심층 연구 보고서 작성 | tavily, arxiv, serper |
| `simple_react` | 간단한 웹 검색/질의응답 | tavily |
| `orchestrator` | 요청 분석 후 적합한 에이전트에 위임 | (하위 에이전트 위임) |

### 2.3 스킬 시스템

스킬 파일(YAML/JSON)을 `SKILLS_DIR`에 배치하면 에이전트가 활용한다.

스킬 파일 예시 (`skills/code_review.yaml`):

```yaml
name: code_review
description: 코드 리뷰 수행
tags: [review, quality]
version: "1.0.0"
body: |
  당신은 코드 리뷰어입니다. 다음 코드를 검토하세요...
references:
  - path: ./templates/review_checklist.md
    description: 리뷰 체크리스트
```

```bash
# 환경변수 설정
export SKILLS_DIR=./skills

# CLI에서 스킬 확인 및 활성화
[coding_assistant] > /skill list
등록된 스킬:
  - code_review: 코드 리뷰 수행 [1]

[coding_assistant] > /skill activate code_review
스킬 [code_review] 활성화 완료
```

### 2.4 메모리 시스템

CLI는 CoALA 기반 4종 메모리를 자동으로 관리한다.

- **Working Memory**: 현재 대화 컨텍스트 (자동)
- **Semantic Memory**: 프로젝트 규칙/컨벤션 (수동 등록)
- **Episodic Memory**: 실행 결과 이력 (자동 저장, 세션 스코프)
- **Procedural Memory**: 학습된 코드 패턴 (검증 통과 시 자동 누적)

```
[coding_assistant] > /memory
메모리 항목 수: 12
```

### 2.5 토큰 스트리밍

CLI는 `astream_events(v2)`를 사용하여 LLM 토큰을 실시간으로 스트리밍한다. 노드 전환 시 진행 상태도 표시된다.

```
  > 요청 분석 중...
  > 코드 생성 중...
Agent:
(여기서 토큰이 실시간으로 출력됨)
  > 코드 검증 중...
```

### 2.6 Langfuse 관측성

CLI에서 Langfuse 관측성을 활성화하면 모든 에이전트 실행이 자동으로 트레이싱된다.

```bash
# .env에 설정
LANGFUSE_HOST=http://localhost:3100
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# CLI에서 확인
[coding_assistant] > /session
세션 ID: a1b2c3d4e5f6
...
```

비활성화하려면:

```bash
CLI_LANGFUSE_ENABLED=0
```

---

## 3. Docker 배포

### 3.1 전체 서비스 기동

```bash
cd docker

# 전체 서비스 빌드 + 기동 (7개 컨테이너)
docker compose up -d

# 상태 확인
docker compose ps
```

Docker Compose는 다음 7개 서비스를 기동한다:

| 서비스 | 포트 | 역할 |
|--------|------|------|
| `mcp-tavily` | 3001 | Tavily 웹 검색 MCP |
| `mcp-arxiv` | 3000 | arXiv 논문 검색 MCP |
| `mcp-serper` | 3002 | Serper 구글 검색 MCP |
| `agent-simple-react` | 18081 | SimpleReAct 에이전트 (A2A) |
| `agent-deep-research` | 18082 | DeepResearch 에이전트 (A2A) |
| `agent-deep-research-a2a` | 18083 | DeepResearchA2A 에이전트 |
| `cli` | - | 대화형 CLI |

### 3.2 CLI 대화형 모드

```bash
# CLI 컨테이너로 대화형 모드 진입
docker compose run --rm cli
```

### 3.3 개별 서비스 로그 확인

```bash
# 특정 서비스 로그
docker compose logs -f agent-deep-research

# 모든 서비스 로그
docker compose logs -f
```

### 3.4 서비스 종료

```bash
docker compose down
```

### 3.5 Langfuse 인프라 기동

Langfuse 관측성 플랫폼을 함께 사용하려면:

```bash
# Langfuse 인프라 기동
docker compose -f docker-compose.langfuse.yaml up -d

# 에이전트 서비스와 함께 기동
docker compose -f docker-compose.yml -f docker-compose.langfuse.yaml up -d
```

### 3.6 의존성 순서

Docker Compose의 서비스 기동 순서:

```
MCP 서버 (헬스체크 통과)
    -> Agent 서버 (헬스체크 통과)
        -> CLI
```

각 서비스는 `healthcheck`로 준비 상태를 확인한 후 다음 서비스가 시작된다.

---

## 4. 에이전트 커스터마이징

### 4.1 새 에이전트 추가하기

1. `agents/` 디렉토리에 새 패키지를 생성한다.

```
agents/
└── my_agent/
    ├── __init__.py
    ├── agent.py      # BaseGraphAgent 상속
    ├── config.py      # BaseAgentConfig 상속
    └── schemas.py     # 상태 스키마
```

2. `agent.py`에서 `BaseGraphAgent`를 상속한다.

```python
from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.base_state import BaseGraphState
from langgraph.graph import START, END, StateGraph
from typing import ClassVar

class MyAgent(BaseGraphAgent):
    NODE_NAMES: ClassVar[dict[str, str]] = {
        "PROCESS": "process",
    }

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("PROCESS"), self._process)

    def init_edges(self, graph: StateGraph) -> None:
        graph.add_edge(START, self.get_node_name("PROCESS"))
        graph.add_edge(self.get_node_name("PROCESS"), END)

    async def _process(self, state):
        # 구현
        ...
```

3. MCP 도구가 필요하면 `async_init()`을 오버라이드한다.

```python
async def async_init(self) -> None:
    self._tools = await self._mcp_loader.load()
```

4. CLI에서 사용하려면 `cli/app.py`의 `_create_agent()` 함수에 등록한다.

5. `pyproject.toml`에 패키지 매핑을 추가한다.

```toml
[tool.setuptools.package-dir]
"youngs75_a2a.agents.my_agent" = "agents/my_agent"
```

### 4.2 모델 설정 커스터마이징

#### 멀티티어 모델 변경

```env
# .env에서 티어별 모델 변경
STRONG_MODEL=claude-sonnet-4-20250514
STRONG_PROVIDER=anthropic

DEFAULT_MODEL=gpt-5.4
DEFAULT_PROVIDER=openai

FAST_MODEL=gpt-4.1-mini
FAST_PROVIDER=openai
```

#### Purpose -> Tier 매핑 변경

```env
# JSON 형식으로 오버라이드
PURPOSE_TIERS={"generation":"strong","verification":"default","parsing":"fast","default":"default"}
```

#### Coding 에이전트 전용 모델 오버라이드

```env
# 티어 시스템보다 우선 적용
CODING_GEN_MODEL=claude-sonnet-4-20250514
CODING_VERIFY_MODEL=gpt-4.1-mini
```

#### OpenRouter 사용

```env
OPENROUTER_API_KEY=sk-or-...
STRONG_PROVIDER=openrouter
STRONG_MODEL=meta-llama/llama-4-maverick
```

### 4.3 MCP 서버 추가

새 MCP 서버를 연동하려면:

1. 에이전트 Config의 `mcp_servers`에 서버 추가:

```python
class MyConfig(BaseAgentConfig):
    mcp_servers: dict[str, str] = Field(
        default_factory=lambda: {
            "code_tools": os.getenv("CODE_TOOLS_MCP_URL", "http://localhost:3003/mcp/"),
            "my_tool": os.getenv("MY_TOOL_MCP_URL", "http://localhost:3004/mcp/"),
        },
    )
```

2. MCP 서버를 실행한다.

3. `MCPToolLoader`가 자동으로 헬스체크 후 도구를 로딩한다.

### 4.4 프롬프트 커스터마이징

코딩 에이전트의 프롬프트는 `agents/coding_assistant/prompts.py`에서 관리된다.

```python
# 프롬프트 레지스트리 접근
from youngs75_a2a.agents.coding_assistant.prompts import get_prompt_registry

registry = get_prompt_registry()

# 현재 프롬프트 확���
print(registry.get_prompt("parse"))
print(registry.get_current_version("parse"))  # "v1"

# 프롬프트 버전 목록
print(registry.list_versions("parse"))
```

Remediation Agent가 자동으로 프롬프트를 개선할 수도 있다 (`/eval remediate` 커맨드).

### 4.5 Safety Envelope 설정

```python
from youngs75_a2a.core.action_validator import ActionValidator

validator = ActionValidator(
    allowed_extensions=[".py", ".js", ".ts"],
    max_delete_lines=50,
    allowed_directories=["./src", "./tests"],
)

report = validator.validate(generated_code, target_files=["./src/main.py"])
if not report.is_safe:
    print(report.summary())
```

---

## 5. 평가 파이프라인

### 5.1 개요

Closed-Loop 평가 시스템: Dataset 생성 -> 메트릭 평가 -> 개선안 추천

```
Loop 1: Dataset      →  Loop 2: Evaluation  →  Loop 3: Remediation
(Golden Dataset 생성)    (DeepEval 메트릭)       (프롬프트 개선 추천)
                              ↓
                       Langfuse Dashboard
                       (관측성 모니터링)
```

### 5.2 CLI에서 평가 실행

```
[coding_assistant] > /eval
평가를 시작합니다... (시간이 걸릴 수 있습니다)
...
평가 완료: 통과 8/10, 평균 점수 0.85

[coding_assistant] > /eval status
마지막 평가 결과:
  통과: 8/10
  평균 점수: 0.85
  ...

[coding_assistant] > /eval remediate
Remediation Agent를 시작합니다...
...
추천 3건:
  [HIGH] parse 프롬프트에 JSON 스키마 예시 추가
  [MED] verify 프롬프트에 보안 패턴 강화
  [LOW] execute 프롬프트 응답 형식 가이드라인 추가

프롬프트 개선 적용: parse, verify

[coding_assistant] > /eval remediate status
마지막 Remediation 결과:
  추천: 3건
  적용: 2건
```

### 5.3 스크립트로 평가 실행

```bash
# Langfuse 실험 실행
python -m youngs75_a2a.scripts.10_run_langfuse_experiment --run-name "test"

# 코딩 에이전트 직접 테스트
python -m youngs75_a2a.tests.test_coding_assistant 1
```

### 5.4 온라인 평가 (Langfuse External Evaluation)

프로덕션 환경에서 Langfuse 트레이스를 자동으로 평가한다.

```
1. Fetch:    Langfuse SDK로 프로덕션 트레이스 조회 (시간 범위, 태그 필터)
2. Evaluate: DeepEval 메트릭으로 각 트레이스 평가 (13개 메트릭)
3. Push:     "deepeval.*" 접두사 스코어로 Langfuse에 기록
```

Langfuse 대시보드에서 `deepeval.*` 필터로 평가 결과를 모니터링할 수 있다.

### 5.5 메트릭 종류

| 카테고리 | 개수 | 메트릭 |
|----------|------|--------|
| RAG | 4 | Faithfulness, Relevancy, Contextual Precision/Recall |
| Agent | 2 | Task Completion, Tool Usage |
| Custom | 7 | 코딩 정확성, 인용 품질, 보안 등 |
| 합계 | 13 | |

---

## 6. FAQ

### Q: `youngs75_a2a`와 `a2a_local` 디렉토리의 관계는?

`a2a_local/` 디렉토리는 `pyproject.toml`의 패키지 매핑에 의해 `youngs75_a2a.a2a`로 임포트된다. 코드에서는 항상 `from youngs75_a2a.a2a import ...`로 사용한다.

```toml
# pyproject.toml
[tool.setuptools.package-dir]
"youngs75_a2a.a2a" = "a2a_local"
```

### Q: MCP 서버가 꺼져 있으면 에이전트가 멈추나?

아니다. `MCPToolLoader`는 Graceful Degradation을 지원한다. MCP 서버 접근이 불가하면 빈 도구 목록으로 진행하며, 에이전트는 도구 없이 LLM만으로 응답한다.

### Q: 체크포인터를 영속화하려면?

CLI 설정에서 SQLite 백엔드를 사용한다:

```env
CLI_CHECKPOINTER=sqlite
CLI_CHECKPOINTER_SQLITE_PATH=.checkpoints.db
```

### Q: OpenRouter로 오픈소스 모델을 사용하려면?

```env
OPENROUTER_API_KEY=sk-or-...
STRONG_MODEL=meta-llama/llama-4-maverick
STRONG_PROVIDER=openrouter
```

`create_chat_model()`이 `openrouter` 프로바이더를 감지하면 OpenAI 호환 API (`openai_api_base`)를 사용한다.

### Q: Docker 없이 에이전트를 A2A 서버로 띄우려면?

```python
from youngs75_a2a.a2a import LGAgentExecutor, run_server
from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent

import asyncio

async def main():
    agent = await CodingAssistantAgent.create()
    run_server(
        executor=LGAgentExecutor(graph=agent.graph),
        name="coding-agent",
        port=8080,
    )

asyncio.run(main())
```

### Q: 에이전트 실행을 프로그래밍 방식으로 호출하려면?

```python
import asyncio
from langchain_core.messages import HumanMessage
from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
from youngs75_a2a.agents.coding_assistant.config import CodingConfig

async def main():
    agent = await CodingAssistantAgent.create(config=CodingConfig())
    result = await agent.graph.ainvoke({
        "messages": [HumanMessage("파이썬으로 피보나치 함수를 작성해줘")],
        "iteration": 0,
        "max_iterations": 3,
    })
    print(result["generated_code"])
    print(result["verify_result"])

asyncio.run(main())
```

### Q: 커스텀 스킬 파일 형식은?

YAML 또는 JSON 형식이다. `SKILLS_DIR` 환경변수로 디렉토리를 지정한다.

```yaml
# skills/my_skill.yaml
name: my_skill
description: 내 커스텀 스킬
tags: [custom, python]
version: "1.0.0"
enabled: true
body: |
  당신은 ... 전문가입니다.
  다음 작업을 수행하세요:
  ...
references:
  - path: ./templates/checklist.md
    description: 체크리스트
```

- **L1** (항상 로드): `name`, `description`, `tags` -> 프롬프트 컨텍스트에 자동 주입
- **L2** (`/skill activate` 시): `body` 로드
- **L3** (실행 시): `references` 파일 내용 로드

### Q: Remediation이 프롬프트를 자동으로 변경하나?

`/eval remediate` 커맨드 실행 시, `RecommendationReport.get_prompt_changes()` 결과를 `PromptRegistry.apply_remediation()`에 전달하여 프롬프트에 개선 사항을 추가한다. 변경된 프롬프트는 세션 내에서만 유효하며, 원본 프롬프트 파일은 수정되지 않는다.

### Q: 테스트를 실행하려면?

```bash
# 전체 테스트
uv run pytest

# 특정 테스트
uv run pytest tests/test_core.py -v
```
