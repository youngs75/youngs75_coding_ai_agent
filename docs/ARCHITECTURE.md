# 아키텍처 문서

**프로젝트**: youngs75-coding-ai-agent
**패키지**: youngs75_a2a
**버전**: 0.1.0

---

## 1. 시스템 전체 구성도

```mermaid
graph TB
    subgraph User["사용자 인터페이스"]
        CLI["CLI<br/>(prompt-toolkit + rich)"]
    end

    subgraph Agents["에이전트 계층"]
        ORC["Orchestrator<br/>(기본 에이전트)<br/>요청 분류 + HITL 승인 + 위임"]
        PL["Planner<br/>아키텍처 설계 + 태스크 분해<br/>(REASONING 티어)"]
        CA["CodingAssistant<br/>2단계 파이프라인<br/>(FAST→STRONG)"]
        DR["DeepResearch<br/>심층 연구"]
        SR["SimpleReAct<br/>MCP 도구 기반"]
    end

    subgraph Core["코어 프레임워크"]
        BGA["BaseGraphAgent<br/>그래프 에이전트 템플릿"]
        BAC["BaseAgentConfig<br/>모델 팩토리 + 설정"]
        MCL["MCPToolLoader<br/>MCP 도구 로딩/캐싱"]
        AV["ActionValidator<br/>Safety Envelope"]
        MEM["Memory<br/>CoALA 4종 메모리"]
        SKL["Skills<br/>3-Level Progressive<br/>+ 자동 활성화"]
        SUB["SubAgents<br/>동적 선택 레지스트리"]
        MT["ModelTiers<br/>멀티티어 모델 해석<br/>(5 purpose)"]
    end

    subgraph A2A["A2A 프로토콜 계층"]
        EXE["Executor<br/>Base + LangGraph"]
        SRV["Server<br/>run_server() 한 줄 조립"]
        DIS["Discovery<br/>AgentCard 레지스트리"]
        RTR["Router<br/>능력 기반 라우팅"]
        RES["Resilience<br/>재시도 + 서킷브레이커"]
        STR["Streaming<br/>청크 단위 스트리밍"]
    end

    subgraph MCP["MCP 서버"]
        CT["code_tools<br/>파일 I/O, 검색, 실행"]
        TV["tavily<br/>웹 검색"]
        AX["arxiv<br/>논문 검색"]
        SP["serper<br/>구글 검색"]
    end

    subgraph Eval["평가 파이프라인"]
        L1["Loop 1<br/>Dataset 생성"]
        L2["Loop 2<br/>메트릭 평가"]
        L3["Loop 3<br/>Remediation"]
        OBS["Observability<br/>Langfuse 통합"]
    end

    subgraph Infra["인프라"]
        LF["Langfuse<br/>관측성 플랫폼"]
        DC["Docker Compose<br/>7개 서비스"]
    end

    CLI --> ORC
    ORC -->|"⇢ 위임"| CA
    ORC -->|"⇢ 위임"| DR
    ORC -->|"⇢ 위임"| SR

    CA --> BGA
    DR --> BGA
    SR --> BGA
    ORC --> BGA

    BGA --> BAC
    BAC --> MT
    CA --> MCL
    SR --> MCL
    DR --> MCL
    CA --> AV
    CA --> MEM
    CA --> SKL

    MCL --> CT
    MCL --> TV
    MCL --> AX
    MCL --> SP

    EXE --> SRV
    SRV --> DIS
    RTR --> DIS
    RTR --> RES
    RES --> STR

    L1 --> L2
    L2 --> L3
    L2 --> OBS
    OBS --> LF

    DC --> MCP
    DC --> Agents
    DC --> CLI
```

---

## 2. 요청 흐름 (Orchestrator-First 아키텍처)

모든 사용자 요청은 Orchestrator를 거쳐 적합한 Subagent로 라우팅됩니다.

```mermaid
sequenceDiagram
    participant U as 사용자
    participant CLI as CLI
    participant ORC as Orchestrator
    participant CA as CodingAssistant
    participant DR as DeepResearch
    participant SR as SimpleReAct

    U->>CLI: 요청 입력
    CLI->>ORC: classify (LLM)
    
    alt 코드 생성/수정/리뷰
        ORC->>CA: ⇢ 위임: coding_assistant
        CA-->>CLI: 코드 + 검증 결과
    else 심층 조사/기술 분석
        ORC->>DR: ⇢ 위임: deep_research
        DR-->>CLI: 연구 보고서
    else 파일 조회/간단한 질의
        ORC->>SR: ⇢ 위임: simple_react
        SR-->>CLI: 도구 실행 결과
    else 복합 작업
        ORC->>ORC: coordinate (병렬 오케스트레이션)
        ORC-->>CLI: 통합 응답
    end
    
    CLI-->>U: 실시간 스트리밍 출력
```

---

## 3. 3계층 분리 아키텍처

본 프레임워크는 **관심사 분리** 원칙에 따라 3개 계층으로 구성된다.

| 계층 | 디렉토리 | 역할 | 의존 방향 |
|------|----------|------|-----------|
| **Core** | `core/` | 도메인 무관 프레임워크 | 없음 (최하위) |
| **A2A** | `a2a_local/` | 프로토콜 통합 | Core |
| **Agents** | `agents/` | 도메인별 에이전트 구현 | Core, A2A |

```mermaid
graph LR
    A[agents/] --> C[core/]
    A --> B[a2a_local/]
    B --> C
    CLI[cli/] --> A
    CLI --> C
    EVAL[eval_pipeline/] --> A
    EVAL --> C
```

새 에이전트를 추가할 때 `agents/` 디렉토리에 구현하면 `core/`와 `a2a_local/` 인프라를 그대로 재사용할 수 있다.

---

## 4. 에이전트별 그래프 흐름도

### 4.1 CodingAssistantAgent (2단계 파이프라인)

논문 인사이트 기반 설계:
- **P1** (Agent-as-a-Judge): parse → execute → verify 최소 3단계
- **P2** (RubricRewards): Generator/Verifier 모델 분리 + 도구 호출/코드 생성 모델 분리
- **P5** (GAM): MCP 도구를 통한 JIT 원본 참조

```mermaid
graph TD
    START((START)) --> PARSE["parse_request<br/>요청 분석 + 스킬 자동 활성화<br/>(DEFAULT 티어)"]
    PARSE --> MEM["retrieve_memory<br/>Episodic/Procedural 검색"]
    MEM --> EXEC["execute_code<br/>도구 호출 판단<br/>(FAST 티어)"]
    EXEC --> ROUTE{도구 호출?}
    ROUTE -->|Yes| TOOLS["execute_tools<br/>MCP 도구 실행<br/>(병렬/순차)"]
    TOOLS --> EXEC
    ROUTE -->|"No + 도구 사용 이력 있음"| GEN["generate_final<br/>최종 코드 생성<br/>(STRONG 티어)"]
    ROUTE -->|"No + 도구 미사용"| VERIFY
    GEN --> VERIFY["verify_result<br/>코드 검증<br/>(DEFAULT 티어)"]
    VERIFY --> RETRY{검증 통과?}
    RETRY -->|passed=true| END_OK((END))
    RETRY -->|"passed=false<br/>iteration < max"| EXEC
    RETRY -->|"passed=false<br/>iteration >= max"| END_FAIL((END))
```

**2단계 파이프라인 라우팅 로직:**
- 도구 호출 있음 → `EXECUTE_TOOLS` → `EXECUTE` (ReAct 루프, FAST 모델)
- 도구 호출 없음 + 이전에 도구 사용함 → `GENERATE_FINAL` (STRONG 모델로 최종 생성)
- 도구 호출 없음 + 도구 미사용 → `VERIFY` (FAST 출력 그대로 검증, STRONG 생략)

**비용 최적화**: 단순 코드 생성(도구 불필요)은 FAST 모델만 사용하여 비용 90% 절감.

**목적별 모델 매핑:**
| purpose | tier | 노드 |
|---------|------|------|
| `parsing` | FAST | parse_request |
| `tool_planning` | FAST | execute_code (ReAct 루프) |
| `generation` | STRONG | generate_final |
| `verification` | DEFAULT | verify_result |

**상태 스키마**: `CodingState`
- `messages`: 대화 이력 (add_messages 누적)
- `semantic_context`: Semantic Memory (프로젝트 규칙/컨벤션)
- `skill_context`: Skills 본문 (L2, 자동 활성화된 스킬)
- `episodic_log`: Episodic Memory (이전 실행 이력)
- `procedural_skills`: Procedural Memory (학습된 코드 패턴)
- `parse_result`: 요청 분석 결과 (task_type, language, description 등)
- `generated_code`: 생성된 코드
- `verify_result`: 검증 결과 (`passed`, `issues`, `suggestions`)
- `project_context`: JIT 원본 참조 결과
- `iteration` / `max_iterations`: 반복 제어
- `tool_call_count`: ReAct 루프 내 도구 호출 누적 횟수

### 4.2 OrchestratorAgent (기본 에이전트)

모든 사용자 요청의 진입점. 요청을 분류하여 적합한 Subagent에 위임.

```mermaid
graph TD
    START((START)) --> CLS["classify<br/>요청 분류<br/>(LLM 기반)"]
    CLS --> ROUTE{분류 결과}
    ROUTE -->|"coding_assistant<br/>deep_research<br/>simple_react"| DEL["delegate<br/>Subagent 위임<br/>(A2A 또는 로컬)"]
    ROUTE -->|"coordinate<br/>(복합 작업)"| COORD["coordinate<br/>병렬 워커 오케스트레이션"]
    DEL --> RSP["respond<br/>응답 포맷팅"]
    COORD --> RSP
    RSP --> END((END))
```

**위임 전략 (우선순위):**
1. A2A 프로토콜 (HTTP 엔드포인트가 설정된 경우)
2. 로컬 에이전트 직접 호출 (폴백)

**등록된 Subagent:**
| 에이전트 | 설명 |
|----------|------|
| `coding_assistant` | 코드 생성, 수정, 리팩토링, 버그 수정, 코드 리뷰 |
| `deep_research` | 심층 조사, 리서치, 기술 분석, 보고서 작성 |
| `simple_react` | MCP 도구를 사용한 간단한 질의응답, 파일 조회 |

### 4.3 DeepResearchAgent

다단계 심층 연구 워크플로우.

```mermaid
graph TD
    START((START)) --> MEM_R["retrieve_memory<br/>Semantic/Episodic 검색"]
    MEM_R --> CLARIFY["clarify_with_user<br/>질문 명확화"]
    CLARIFY --> BRIEF["write_research_brief<br/>연구 브리프 작성"]
    BRIEF --> SUPER["research_supervisor<br/>(서브그래프)"]

    subgraph Supervisor["Supervisor 서브그래프"]
        direction TB
        S_START((시작)) --> S_DECIDE{연구 계속?}
        S_DECIDE -->|Yes| S_RESEARCH["conduct_research<br/>병렬 연구 실행"]
        S_RESEARCH --> S_COMPRESS["compress_research<br/>연구 결과 압축"]
        S_COMPRESS --> S_DECIDE
        S_DECIDE -->|No| S_END((종료))
    end

    SUPER --> REPORT["final_report_generation<br/>최종 보고서"]
    REPORT --> MEM_W["record_episodic<br/>연구 이력 기록"]
    MEM_W --> END((END))
```

### 4.4 SimpleMCPReActAgent

LangGraph `create_react_agent` 기반 단일 노드 에이전트.

```mermaid
graph LR
    START((START)) --> REACT["react_agent<br/>(create_react_agent)<br/>MCP 도구 + LLM"]
    REACT --> END((END))
```

---

## 5. 코어 프레임워크 구조

### 5.1 BaseGraphAgent (Template Method 패턴)

모든 에이전트의 기반 클래스. 서브클래스는 `init_nodes()`와 `init_edges()`만 구현하면 된다.

```mermaid
classDiagram
    class BaseGraphAgent {
        +graph: CompiledStateGraph
        +agent_config: BaseAgentConfig
        +model: BaseChatModel
        +permission_manager: ToolPermissionManager
        +tool_executor: ParallelToolExecutor
        +context_manager: ContextManager
        +build_graph()
        +init_nodes(graph)*
        +init_edges(graph)*
        +create(**kwargs)$ async
        +async_init()
    }

    class CodingAssistantAgent {
        +_coding_config: CodingConfig
        +_mcp_loader: MCPToolLoader
        +_memory_store: MemoryStore
        +_skill_registry: SkillRegistry
        +_tool_planning_model: BaseChatModel
        +_gen_model: BaseChatModel
        +init_nodes(graph)
        +init_edges(graph)
    }

    class DeepResearchAgent {
        +_research_config: ResearchConfig
        +init_nodes(graph)
        +init_edges(graph)
    }

    class SimpleMCPReActAgent {
        +_react_config: SimpleReActConfig
        +_mcp_loader: MCPToolLoader
        +init_nodes(graph)
        +init_edges(graph)
    }

    class OrchestratorAgent {
        +_orch_config: OrchestratorConfig
        +init_nodes(graph)
        +init_edges(graph)
    }

    BaseGraphAgent <|-- CodingAssistantAgent
    BaseGraphAgent <|-- DeepResearchAgent
    BaseGraphAgent <|-- SimpleMCPReActAgent
    BaseGraphAgent <|-- OrchestratorAgent
```

### 5.2 BaseAgentConfig (모델 해석 체계)

```mermaid
graph TD
    PURPOSE["get_model(purpose)"] --> TIER["get_tier_config(purpose)"]
    TIER --> PT["purpose_tiers 매핑"]
    PT -->|generation| STRONG["STRONG<br/>qwen/qwen3-coder-plus"]
    PT -->|tool_planning| FAST_TP["FAST<br/>qwen/qwen3.5-flash-02-23"]
    PT -->|verification| DEFAULT["DEFAULT<br/>qwen/qwen3-coder-next"]
    PT -->|parsing| FAST_P["FAST<br/>qwen/qwen3.5-flash-02-23"]
    STRONG --> CREATE["create_chat_model()"]
    FAST_TP --> CREATE
    DEFAULT --> CREATE
    FAST_P --> CREATE
```

CodingConfig는 환경변수(`CODING_GEN_MODEL` 등)가 있으면 티어보다 우선 적용한다.

### 5.3 메모리 시스템 (CoALA 논문 기반)

```mermaid
graph TD
    subgraph Memory["CoALA 4종 메모리"]
        W["Working Memory<br/>(= messages)"]
        S["Semantic Memory<br/>프로젝트 규칙/컨벤션<br/>(AGENTS.md 자동 로딩)"]
        E["Episodic Memory<br/>실행 결과 이력<br/>(세션 스코프, 최대 5개)"]
        P["Procedural Memory<br/>학습된 코드 패턴<br/>(Voyager식 누적)"]
    end

    subgraph Search["GAM 2단계 검색"]
        T1["1단계: TagBasedSearch<br/>태그 기반 경량 필터링"]
        T2["2단계: ContentBasedSearch<br/>BM25 컨텐츠 순위"]
    end

    MS["MemoryStore<br/>CRUD + 검색"]
    MS --> Memory
    MS --> Search
    T1 --> T2

    subgraph Novelty["중복 방지"]
        NF["Jaccard 유사도<br/>novelty_threshold"]
    end

    P --> NF
```

### 5.4 Skills 시스템 (3-Level Progressive Loading + 자동 활성화)

```mermaid
graph TD
    subgraph Loading["3-Level Progressive Loading"]
        L1["L1: 메타데이터<br/>(name, description, tags)<br/>항상 컨텍스트 주입"]
        L2["L2: 본문<br/>(prompt body)<br/>활성화 시 로드"]
        L3["L3: 참조 파일<br/>(references)<br/>실행 시 로드"]
        L1 --> L2 --> L3
    end

    subgraph AutoActivation["task_type 기반 자동 활성화"]
        PARSE["parse_request<br/>task_type 추출"] --> MAP["TASK_TYPE_TAGS 매핑"]
        MAP -->|generate| GEN["quality → code_review,<br/>refactor, test_generation"]
        MAP -->|fix| FIX["fix, debug → debug"]
        MAP -->|refactor| REF["refactor → refactor"]
        MAP -->|explain| EXP["explain → explain"]
        MAP -->|analyze| ANA["review, security →<br/>code_review, security_review"]
    end

    GEN & FIX & REF & EXP & ANA --> ACTIVATE["registry.auto_activate_for_task()"]
    ACTIVATE --> L2
    L2 --> INJECT["시스템 프롬프트에 L2 본문 주입"]
```

- `SkillLoader`: YAML/JSON 파일에서 스킬 로드
- `SkillRegistry`: 스킬 등록, 검색, 활성화 관리 + `auto_activate_for_task()`
- 수동 활성화: `/skill activate <name>`

### 5.5 SubAgent 동적 선택 (Puppeteer 논문 기반)

```
R = r(quality) - lambda * C(cost)

quality: 에이전트의 태스크 유형별 성공률 (70%) + 전체 성공률 (30%)
cost:    에이전트의 cost_weight (레이턴시 + 토큰 비용 대리)
lambda:  비용 민감도 (환경변수로 조정)
```

`SubAgentRegistry`가 사용 통계를 추적하여 선택 품질을 지속적으로 개선한다.

### 5.6 ActionValidator (Safety Envelope)

```mermaid
graph LR
    CODE["에이전트 출력"] --> AV["ActionValidator"]
    AV --> R1["시크릿 노출 방지"]
    AV --> R2["위험 명령 차단"]
    AV --> R3["대량 삭제 감지"]
    AV --> R4["파일 확장자 검사"]
    AV --> R5["디렉토리 범위 검사"]
    AV --> R6["금지 경로 차단<br/>(.claude/, .git/,<br/>__pycache__/, node_modules/)"]
    R1 & R2 & R3 & R4 & R5 & R6 --> REPORT["ValidationReport"]
    REPORT --> SAFE{is_safe?}
    SAFE -->|Yes| EXECUTE["실행"]
    SAFE -->|No| BLOCK["차단"]
```

---

## 6. A2A 프로토콜 통합

### 6.1 서버 조립 흐름

```mermaid
graph LR
    AGENT["에이전트 그래프"] --> EXE["AgentExecutor<br/>(Base/LG)"]
    EXE --> HANDLER["DefaultRequestHandler"]
    HANDLER --> APP["A2AStarletteApplication"]
    APP --> UVICORN["Uvicorn 서버"]
```

`run_server()` 한 줄로 에이전트를 A2A 서버로 노출:

```python
run_server(
    executor=LGAgentExecutor(graph=agent.graph),
    name="coding-agent",
    port=8080,
)
```

### 6.2 복원력 패턴

- **RetryPolicy**: 지수 백오프 (base_delay * 2^attempt, max_delay=30s)
- **CircuitBreaker**: 연속 5회 실패 시 OPEN, 30초 후 HALF_OPEN
- **AgentMonitor**: 에이전트별 성공률, 응답 시간, 서킷 상태 추적
- **ResilientA2AClient**: 위 패턴을 통합한 복원력 내장 클라이언트

### 6.3 라우팅 전략

`AgentRouter`는 4가지 라우팅 모드를 지원한다:

| 모드 | 전략 |
|------|------|
| `SKILL_BASED` | 스킬 매칭 점수 + 성공률 종합 평가 (기본) |
| `ROUND_ROBIN` | 순차적 분배 |
| `LEAST_LOADED` | 최소 부하 에이전트 선택 |
| `WEIGHTED` | 성공률 (70%) + 응답시간 역수 (30%) |

---

## 7. CLI 실행 흐름

```mermaid
sequenceDiagram
    participant U as 사용자
    participant CLI as CLIApp
    participant CMD as Commands
    participant S as CLISession
    participant ORC as Orchestrator
    participant A as Subagent
    participant MCP as MCP 서버

    Note over CLI: 시작 시 표시:<br/>✓ 스킬 7개 로드<br/>✓ 프로젝트 컨텍스트 로드

    U->>CLI: 입력
    CLI->>CMD: 슬래시 커맨드?
    alt /agent, /skill, /help 등
        CMD->>S: 세션 조작
        CMD-->>U: 결과 출력
    else 일반 메시지
        CLI->>S: 에이전트 조회/생성
        S->>ORC: classify (요청 분류)
        ORC-->>CLI: ⇢ 위임: coding_assistant
        ORC->>A: delegate (Subagent 호출)
        
        Note over A: ⚙ 스킬 자동 활성화
        
        loop ReAct 루프 (FAST 모델)
            A->>MCP: ⚡ 도구 호출
            MCP-->>A: 도구 결과
        end
        
        Note over A: 도구 사용 시 → STRONG 최종 생성<br/>도구 미사용 시 → FAST 출력 그대로

        A-->>CLI: on_chat_model_stream (토큰)
        CLI-->>U: 실시간 스트리밍 출력
        
        Note over CLI: ✓ 검증 통과<br/>⏱ 12.3s
        
        CLI->>S: Episodic Memory 저장
    end
```

### CLI UX 피드백

| 시점 | 표시 | 의미 |
|------|------|------|
| 시작 | `✓ 스킬 7개 로드: ...` | 사용 가능한 스킬 목록 |
| parse 후 | `⚙ 스킬 활성화: debug` | task_type 기반 자동 스킬 선택 |
| 도구 호출 | `⚡ read_file pyproject.toml` | MCP 도구 실행 |
| 스피너 | `⠋ 도구 호출 판단 (FAST)` | FAST 모델 ReAct 루프 |
| 스피너 | `⠋ 코드 생성 (STRONG)` | STRONG 모델 최종 생성 |
| Orchestrator | `⇢ 위임: coding_assistant` | Subagent 위임 |
| 검증 | `✓ 검증 통과` / `✗ 검증 실패` | 코드 품질 검증 |
| 종료 | `⏱ 12.3s` | 턴 소요시간 |

---

## 8. 평가 파이프라인 (Closed-Loop)

```mermaid
graph LR
    subgraph Loop1["Loop 1: Dataset"]
        SYN["Synthesizer<br/>합성 데이터 생성"]
        GOL["GoldenBuilder<br/>골든 데이터셋"]
        AUG["FeedbackAugmenter<br/>피드백 증강"]
        CSV["CSV Export/Import"]
    end

    subgraph Loop2["Loop 2: Evaluation"]
        MET["MetricsRegistry<br/>RAG(4) + Agent(2) + Custom(7)"]
        BAT["BatchEvaluator<br/>오프라인/온라인 평가"]
        LFB["LangfuseBridge<br/>트레이스 fetch/push"]
        CAL["CalibrationCases<br/>교정 데이터"]
    end

    subgraph Loop3["Loop 3: Remediation"]
        ANA["분석 (Analyzer)"]
        OPT["최적화 (Optimizer)"]
        REC["추천 (Recommender)"]
        REP["RecommendationReport"]
    end

    Loop1 -->|Golden Dataset| Loop2
    Loop2 -->|평가 결과| Loop3
    Loop3 -->|프롬프트 개선| Loop1

    BAT --> LFB
    LFB --> LF["Langfuse Dashboard"]
    ANA --> OPT --> REC --> REP
    REP -->|get_prompt_changes()| PROMPT["PromptRegistry<br/>프롬프트 버전 관리"]
```

---

## 9. Docker 배포 구성

```mermaid
graph TB
    subgraph Docker["Docker Compose (7 서비스)"]
        subgraph MCP_Layer["MCP 서버"]
            T["mcp-tavily<br/>:3001"]
            A["mcp-arxiv<br/>:3000"]
            S["mcp-serper<br/>:3002"]
        end

        subgraph Agent_Layer["Agent 서버 (A2A)"]
            SR["agent-simple-react<br/>:18081"]
            DR["agent-deep-research<br/>:18082"]
            DRA["agent-deep-research-a2a<br/>:18083"]
        end

        CLI_C["cli<br/>(대화형)"]
    end

    SR --> T
    DR --> T & A & S
    DRA --> T & A & S
    CLI_C --> SR & DR & DRA & T & A & S
```

---

## 10. 설계 원칙 (논문 7편 기반)

| 원칙 | 근거 논문 | 적용 |
|------|-----------|------|
| **최소 구조** | Agent-as-a-Judge | parse → execute → verify 3단계 |
| **Generator-Verifier 분리** | RubricRewards | 생성(STRONG)/검증(DEFAULT) 모델 분리 |
| **2단계 파이프라인** | 비용 최적화 | 도구 호출(FAST)/최종 생성(STRONG) 분리 |
| **Safety Envelope** | AutoHarness | ActionValidator + 금지 경로 차단 |
| **CoALA 메모리** | CoALA + Voyager | 4종 메모리 + Semantic 자동 로딩 |
| **JIT 원본 참조** | GAM | MCP 도구로 프로젝트 컨텍스트 직접 읽기 |
| **동적 오케스트레이션** | Puppeteer | Orchestrator-First + SubAgent 위임 |
| **스킬 자동 활성화** | Claude Code 패턴 | task_type → 스킬 태그 매핑 |

---

## 11. 핵심 설계 패턴

| 패턴 | 적용 위치 |
|------|-----------|
| **Template Method** | `BaseGraphAgent.init_nodes()` / `init_edges()` |
| **Factory Method** | `BaseGraphAgent.create()`, `BaseAgentConfig.get_model()` |
| **Adapter** | `LGAgentExecutor` (LangGraph ↔ A2A 프로토콜 브릿지) |
| **Orchestrator-First** | 모든 요청이 Orchestrator → Subagent 라우팅 |
| **2-Stage Pipeline** | FAST(도구 판단) → STRONG(최종 생성) 모델 분리 |
| **Progressive Loading** | Skills 3-Level (L1 메타데이터 → L2 본문 → L3 참조) |
| **Circuit Breaker** | `CircuitBreaker` (CLOSED → OPEN → HALF_OPEN 상태 전이) |
| **Subgraph Composition** | Supervisor → Researcher 서브그래프 중첩 |
| **Override Reducer** | 상태 누적/덮어쓰기 양립 |
