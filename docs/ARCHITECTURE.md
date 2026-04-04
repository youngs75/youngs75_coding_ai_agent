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
        ORC["Orchestrator<br/>요청 분류 + 위임"]
        CA["CodingAssistant<br/>코드 생성/검증"]
        DR["DeepResearch<br/>심층 연구"]
        SR["SimpleReAct<br/>MCP 도구 기반"]
    end

    subgraph Core["코어 프레임워크"]
        BGA["BaseGraphAgent<br/>그래프 에이전트 템플릿"]
        BAC["BaseAgentConfig<br/>모델 팩토리 + 설정"]
        MCL["MCPToolLoader<br/>MCP 도구 로딩/캐싱"]
        AV["ActionValidator<br/>Safety Envelope"]
        MEM["Memory<br/>CoALA 4종 메모리"]
        SKL["Skills<br/>3-Level Progressive"]
        SUB["SubAgents<br/>동적 선택 레지스트리"]
        MT["ModelTiers<br/>멀티티어 모델 해석"]
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
    CLI --> CA
    CLI --> DR
    CLI --> SR

    ORC -->|A2A 프로토콜| EXE
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

## 2. 3계층 분리 아키텍처

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

## 3. 에이전트별 그래프 흐름도

### 3.1 CodingAssistantAgent

논문 인사이트 기반 설계:
- **P1** (Agent-as-a-Judge): parse -> execute -> verify 최소 3단계
- **P2** (RubricRewards): Generator/Verifier 모델 분리
- **P5** (GAM): MCP 도구를 통한 JIT 원본 참조

```mermaid
graph TD
    START((START)) --> PARSE["parse_request<br/>요청 분석<br/>(FAST 티어)"]
    PARSE --> EXEC["execute_code<br/>코드 생성<br/>(STRONG 티어)"]
    EXEC --> ROUTE{도구 호출?}
    ROUTE -->|Yes| TOOLS["execute_tools<br/>MCP 도구 실행"]
    ROUTE -->|No| VERIFY["verify_result<br/>코드 검증<br/>(DEFAULT 티어)"]
    TOOLS --> EXEC
    VERIFY --> RETRY{검증 통과?}
    RETRY -->|passed=true| END_OK((END))
    RETRY -->|passed=false<br/>iteration < max| EXEC
    RETRY -->|passed=false<br/>iteration >= max| END_FAIL((END))
```

**상태 스키마**: `CodingState`
- `messages`: 대화 이력 (add_messages 누적)
- `semantic_context`: Semantic Memory (프로젝트 규칙/컨벤션)
- `skill_context`: Skills 메타데이터 (L1)
- `episodic_log`: Episodic Memory (이전 실행 이력)
- `procedural_skills`: Procedural Memory (학습된 코드 패턴)
- `parse_result`: 요청 분석 결과
- `generated_code`: 생성된 코드
- `verify_result`: 검증 결과 (`passed`, `issues`, `suggestions`)
- `project_context`: JIT 원본 참조 결과
- `iteration` / `max_iterations`: 반복 제어

### 3.2 DeepResearchAgent

다단계 심층 연구 워크플로우.

```mermaid
graph TD
    START((START)) --> CLARIFY["clarify_with_user<br/>질문 명확화"]
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

    subgraph Researcher["Researcher 서브그래프"]
        direction TB
        R_START((시작)) --> R_REACT["ReAct Agent<br/>(MCP 도구 사용)"]
        R_REACT --> R_END((종료))
    end

    SUPER --> REPORT["final_report_generation<br/>최종 보고서"]
    REPORT --> END((END))

    S_RESEARCH -.->|"병렬 x N"| Researcher
```

**모델 분리** (ResearchConfig):
- `research_model`: 연구 수행 (기본: gpt-5.4)
- `compression_model`: 연구 결과 압축 (기본: gpt-4o-2024-11-20)
- `final_report_model`: 최종 보고서 (기본: gpt-5.4)

### 3.3 SimpleMCPReActAgent

LangGraph `create_react_agent` 기반 단일 노드 에이전트.

```mermaid
graph LR
    START((START)) --> REACT["react_agent<br/>(create_react_agent)<br/>MCP 도구 + LLM"]
    REACT --> END((END))
```

### 3.4 OrchestratorAgent

요청을 분류하여 적합한 하위 에이전트에 A2A 프로토콜로 위임.

```mermaid
graph LR
    START((START)) --> CLS["classify<br/>요청 분류<br/>(LLM 기반)"]
    CLS --> DEL["delegate<br/>A2A 프로토콜 위임"]
    DEL --> RSP["respond<br/>응답 포맷팅"]
    RSP --> END((END))
```

---

## 4. 코어 프레임워크 구조

### 4.1 BaseGraphAgent (Template Method 패턴)

모든 에이전트의 기반 클래스. 서브클래스는 `init_nodes()`와 `init_edges()`만 구현하면 된다.

```mermaid
classDiagram
    class BaseGraphAgent {
        +graph: CompiledStateGraph
        +agent_config: BaseAgentConfig
        +model: BaseChatModel
        +checkpointer: BaseCheckpointSaver
        +store: BaseStore
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

### 4.2 BaseAgentConfig (모델 해석 체계)

두 가지 모델 해석 경로를 제공한다:

```mermaid
graph TD
    PURPOSE["get_model(purpose)"] --> LEGACY{레거시 경로?}
    LEGACY -->|Yes| RESOLVE["_resolve_model_name(purpose)"]
    RESOLVE --> INIT["init_chat_model()"]
    LEGACY -->|No, 서브클래스 오버라이드| TIER["get_tier_config(purpose)"]
    TIER --> PT["purpose_tiers[purpose]<br/>→ 'strong'/'default'/'fast'"]
    PT --> TC["model_tiers[tier]<br/>→ TierConfig"]
    TC --> CREATE["create_chat_model(tier_config)"]
```

- **레거시 경로**: `_resolve_model_name(purpose)` + `model_provider` -> `init_chat_model()`
- **티어 경로**: `purpose_tiers[purpose]` -> `model_tiers[tier]` -> `create_chat_model()`

CodingConfig는 티어 경로를 사용하되, 환경변수(`CODING_GEN_MODEL` 등)가 있으면 우선 적용한다.

### 4.3 메모리 시스템 (CoALA 논문 기반)

```mermaid
graph TD
    subgraph Memory["CoALA 4종 메모리"]
        W["Working Memory<br/>(= messages)"]
        S["Semantic Memory<br/>프로젝트 규칙/컨벤션"]
        E["Episodic Memory<br/>실행 결과 이력"]
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

| 메모리 타입 | 저장 내용 | 활성 상태 |
|-------------|-----------|-----------|
| Working | 현재 대화 (`messages`) | 항상 활성 |
| Semantic | 프로젝트 규칙/컨벤션 | 항상 활성 |
| Episodic | 실행 결과 이력 (세션 스코프) | 활성 (최대 5개, 200자 제한) |
| Procedural | 학습된 코드 패턴 | 활성 (검증 통과 시 자동 누적) |

### 4.4 Skills 시스템 (3-Level Progressive Loading)

```mermaid
graph LR
    L1["L1: 메타데이터<br/>(name, description, tags)<br/>항상 컨텍스트 주입"]
    L2["L2: 본문<br/>(prompt body)<br/>활성화 시 로드"]
    L3["L3: 참조 파일<br/>(references)<br/>실행 시 로드"]
    L1 --> L2 --> L3
```

- `SkillLoader`: YAML/JSON 파일에서 스킬 로드
- `SkillRegistry`: 스킬 등록, 검색, 활성화 관리

### 4.5 SubAgent 동적 선택 (Puppeteer 논문 기반)

```
R = r(quality) - lambda * C(cost)

quality: 에이전트의 태스크 유형별 성공률 (70%) + 전체 성공률 (30%)
cost:    에이전트의 cost_weight (레이턴시 + 토큰 비용 대리)
lambda:  비용 민감도 (환경변수로 조정)
```

`SubAgentRegistry`가 사용 통계를 추적하여 선택 품질을 지속적으로 개선한다.

### 4.6 ActionValidator (Safety Envelope)

```mermaid
graph LR
    CODE["에이전트 출력"] --> AV["ActionValidator"]
    AV --> R1["시크릿 노출 방지"]
    AV --> R2["위험 명령 차단"]
    AV --> R3["대량 삭제 감지"]
    AV --> R4["파일 확장자 검사"]
    AV --> R5["디렉토리 범위 검사"]
    R1 & R2 & R3 & R4 & R5 --> REPORT["ValidationReport"]
    REPORT --> SAFE{is_safe?}
    SAFE -->|Yes| EXECUTE["실행"]
    SAFE -->|No| BLOCK["차단"]
```

---

## 5. A2A 프로토콜 통합

### 5.1 서버 조립 흐름

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

### 5.2 복원력 패턴

```mermaid
graph TD
    REQ["요청"] --> CB{서킷 브레이커<br/>can_execute?}
    CB -->|CLOSED| RETRY["재시도 루프<br/>(지수 백오프)"]
    CB -->|OPEN| BLOCK["CircuitOpenError"]
    CB -->|HALF_OPEN| TEST["제한적 시도"]
    RETRY -->|성공| SUCCESS["record_success()"]
    RETRY -->|실패| FAIL["record_failure()"]
    FAIL -->|threshold 초과| OPEN["서킷 OPEN"]
    BLOCK --> FALLBACK["폴백 에이전트 시도"]
    SUCCESS --> RESPONSE["응답 반환"]
    FALLBACK --> RESPONSE
```

- **RetryPolicy**: 지수 백오프 (base_delay * 2^attempt, max_delay=30s)
- **CircuitBreaker**: 연속 5회 실패 시 OPEN, 30초 후 HALF_OPEN
- **AgentMonitor**: 에이전트별 성공률, 응답 시간, 서킷 상태 추적
- **ResilientA2AClient**: 위 패턴을 통합한 복원력 내장 클라이언트

### 5.3 라우팅 전략

`AgentRouter`는 4가지 라우팅 모드를 지원한다:

| 모드 | 전략 |
|------|------|
| `SKILL_BASED` | 스킬 매칭 점수 + 성공률 종합 평가 (기본) |
| `ROUND_ROBIN` | 순차적 분배 |
| `LEAST_LOADED` | 최소 부하 에이전트 선택 |
| `WEIGHTED` | 성공률 (70%) + 응답시간 역수 (30%) |

---

## 6. CLI 실행 흐름

```mermaid
sequenceDiagram
    participant U as 사용자
    participant CLI as CLIApp
    participant CMD as Commands
    participant S as CLISession
    participant A as Agent
    participant MCP as MCP 서버

    U->>CLI: 입력
    CLI->>CMD: 슬래시 커맨드?
    alt /agent, /help 등
        CMD->>S: 세션 조작
        CMD-->>U: 결과 출력
    else 일반 메시지
        CLI->>S: 에이전트 조회/생성
        S->>A: _create_agent()
        CLI->>A: astream_events(v2)
        loop 토큰 스트리밍
            A->>MCP: 도구 호출
            MCP-->>A: 도구 결과
            A-->>CLI: on_chat_model_stream (토큰)
            CLI-->>U: 실시간 출력
        end
        A-->>CLI: on_chain_end (최종 결과)
        CLI->>S: Episodic Memory 저장
    end
```

---

## 7. 평가 파이프라인 (Closed-Loop)

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

    subgraph Observability["관측성"]
        CB["CallbackHandler<br/>Langfuse 트레이싱"]
        LF["Langfuse Dashboard"]
    end

    Loop1 -->|Golden Dataset| Loop2
    Loop2 -->|평가 결과| Loop3
    Loop3 -->|프롬프트 개선| Loop1

    BAT --> LFB
    LFB --> LF
    CB --> LF

    ANA --> OPT --> REC --> REP
    REP -->|get_prompt_changes()| PROMPT["PromptRegistry<br/>프롬프트 버전 관리"]
```

### 평가 흐름 상세

| 단계 | 모듈 | 입력 | 출력 |
|------|------|------|------|
| **Loop 1** | `loop1_dataset/` | 프롬프트 템플릿 | Golden Dataset (JSON/CSV) |
| **Loop 2** | `loop2_evaluation/` | Golden Dataset + Langfuse 트레이스 | 메트릭 점수 + Langfuse Scores |
| **Loop 3** | `loop3_remediation/` | 평가 결과 | RecommendationReport + 프롬프트 개선 |

Loop 2의 **External Evaluation Pipeline** (온라인 평가):

```
1. Fetch: Langfuse SDK로 프로덕션 트레이스 조회
2. Evaluate: DeepEval 메트릭으로 각 트레이스 평가
3. Push: "deepeval.*" 접두사 스코어로 Langfuse에 기록
```

---

## 8. Docker 배포 구성

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

- 모든 서비스는 `youngs75_net` 브리지 네트워크로 연결
- MCP 서버는 헬스체크 후 Agent가 시작
- Agent 서버는 헬스체크 후 CLI가 시작
- Langfuse 인프라는 `docker-compose.langfuse.yaml`로 별도 관리

---

## 9. 설계 원칙 (논문 7편 기반)

| 원칙 | 근거 논문 | 적용 |
|------|-----------|------|
| **최소 구조** | Agent-as-a-Judge | parse -> execute -> verify 3단계 |
| **Generator-Verifier 분리** | RubricRewards | 생성/검증 모델 및 프롬프트 분리 |
| **Safety Envelope** | AutoHarness | ActionValidator (프레임워크 차원 강제) |
| **CoALA 메모리** | CoALA + Voyager | 4종 메모리 (Working/Episodic/Semantic/Procedural) |
| **JIT 원본 참조** | GAM | MCP 도구로 프로젝트 컨텍스트 직접 읽기 |
| **동적 오케스트레이션** | Puppeteer | SubAgentRegistry (R = quality - lambda*cost) |
| **졸업 Lifecycle** | Adaptation | 검증된 에이전트를 A2A 서버로 배포 |

---

## 10. 핵심 설계 패턴

| 패턴 | 적용 위치 |
|------|-----------|
| **Template Method** | `BaseGraphAgent.init_nodes()` / `init_edges()` |
| **Factory Method** | `BaseGraphAgent.create()`, `BaseAgentConfig.get_model()` |
| **Adapter** | `LGAgentExecutor` (LangGraph <-> A2A 프로토콜 브릿지) |
| **Cooperative Cancellation** | 스트림 폴링 + `asyncio.Task.cancel()` 하이브리드 |
| **Graceful Degradation** | MCP 로딩 실패 -> 도구 없이 진행 |
| **Circuit Breaker** | `CircuitBreaker` (CLOSED -> OPEN -> HALF_OPEN 상태 전이) |
| **Subgraph Composition** | Supervisor -> Researcher 서브그래프 중첩 |
| **Override Reducer** | 상태 누적/덮어쓰기 양립 |
| **Singleton** | `PromptRegistry`, `Settings` |
| **Progressive Loading** | Skills 3-Level (L1 메타데이터 -> L2 본문 -> L3 참조) |
