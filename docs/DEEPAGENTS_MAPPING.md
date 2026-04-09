# DeepAgents 기능 매핑

본 문서는 DeepAgents 레퍼런스 아키텍처의 핵심 개념이
우리 구현에서 어떤 코드로 대응되는지 매핑합니다.

> **설계 원칙**: DeepAgents를 직접 사용하지 않고 LangGraph 기반으로 자체 구현한 이유는
> [ADR-001](ARCHITECTURE_DECISIONS.md#adr-001-langgraph-기반-에이전트-오케스트레이션)을 참조하세요.

---

## 핵심 개념 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 설명 |
|----------------|---------|----------|------|
| **MemoryMiddleware** | MemoryMiddleware | `core/middleware/memory.py:83` | 요청 전 6종 메모리 검색 주입 + 응답 후 SLM 기반 자동 축적 |
| **SubAgentMiddleware** | SubAgentProcessManager + SubAgentRegistry | `core/subagents/process_manager.py:46`, `core/subagents/registry.py:41` | 프로세스 격리 + Puppeteer 점수 기반 동적 선택 |
| **task() 위임** | Orchestrator._invoke_local_agent() | `agents/orchestrator/agent.py:235` | PlannerAgent 계획 기반 SubAgent 순차 실행 |
| **미들웨어 방어** | ResilienceMiddleware | `core/middleware/resilience.py:47` | retry + fallback + abort 체크포인트 체인 |
| **AgentMiddleware 베이스** | AgentMiddleware ABC | `core/middleware/base.py:76` | wrap_model_call() 기반 before/after 훅 |
| **MiddlewareChain** | MiddlewareChain | `core/middleware/chain.py:18` | 양파(Onion) 패턴으로 미들웨어 스택 실행 |

---

## 메모리 시스템 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 차이점 |
|----------------|---------|----------|-------|
| Memory Store | MemoryStore | `core/memory/store.py:49` | 6종 CoALA 메모리 (DeepAgents 기본 3종 대비 확장) |
| Memory Search | TwoStageSearch | `core/memory/search.py:114` | Tag 필터링 → BM25 랭킹 2단계 (벡터 DB 미사용) |
| Memory Types | MemoryType Enum | `core/memory/schemas.py:24` | WORKING, EPISODIC, SEMANTIC, PROCEDURAL, USER_PROFILE, DOMAIN_KNOWLEDGE |
| Memory Injection | MemoryMiddleware._inject_store_memories() | `core/middleware/memory.py:171` | 목적 기반 메모리 검색 후 시스템 프롬프트에 주입 |
| Auto-accumulation | MemoryMiddleware._try_accumulate() | `core/middleware/memory.py:213` | SLM(FAST 티어)로 응답에서 자동 지식 추출 |
| Skill Library (Voyager) | MemoryStore.accumulate_skill() | `core/memory/store.py:189` | Procedural Memory로 성공 코드 패턴 누적 |
| Semantic Memory | SemanticMemoryLoader | `core/memory/semantic_loader.py:32` | AGENTS.md + pyproject.toml에서 프로젝트 규칙 자동 추출 |
| Memory State | MemoryAwareState | `core/memory/state.py:19` | LangGraph 상태에 6종 메모리 컨텍스트 필드 통합 |

---

## SubAgent 시스템 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 차이점 |
|----------------|---------|----------|-------|
| task() 위임 | SubAgentProcessManager.spawn() | `core/subagents/process_manager.py:62` | 프로세스 격리 (DeepAgents는 인프로세스) |
| Agent 선택 | SubAgentRegistry.select() | `core/subagents/registry.py:74` | R = quality - λ·cost 점수 기반 Puppeteer 알고리즘 |
| Agent 상태 추적 | SubAgentRegistry.transition_state() | `core/subagents/registry.py:178` | 8단계 상태 전이 (created → destroyed) |
| Agent 인스턴스 | SubAgentRegistry.create_instance() | `core/subagents/registry.py:148` | 런타임 동적 생성 + 메타데이터 (ID, 역할, 상태, 타임스탬프) |
| Agent Worker | worker.py _run_agent() | `core/subagents/worker.py:169` | 독립 프로세스에서 에이전트 초기화 + 실행 |
| Context Isolation | SubagentContextFilter | `core/subagent_context.py:37` | 필요한 정보만 명시적 전달 (전체 히스토리 상속 차단) |
| Auto-retry | spawn_and_wait() | `core/subagents/process_manager.py:266` | 실패 시 자동 재시도 + 역할 변경 가능 |
| Cleanup | cleanup_all(), _reap_zombies() | `core/subagents/process_manager.py:343, 353` | 좀비 프로세스 정리 + 임시 자원 해제 |

---

## 오케스트레이션 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 차이점 |
|----------------|---------|----------|-------|
| Main Agent Loop | Orchestrator 상태 그래프 | `agents/orchestrator/agent.py` | LangGraph StateGraph (classify → plan → delegate → respond) |
| Task Planning | PlannerAgent | `agents/planner/agent.py:85` | REASONING 티어로 아키텍처 설계 + 페이즈 분해 |
| Task Execution | CodingAssistant | `agents/coding_assistant/agent.py` | STRONG 티어로 코드 생성 + MCP 도구 기반 파일 저장 |
| Task Verification | VerificationAgent | `agents/verifier/agent.py:40` | 3-tier 검증: lint → test → LLM 리뷰 → 집계 |
| Complexity Router | Orchestrator._detect_complexity() | `agents/orchestrator/agent.py:89` | multi-file, fullstack, multi-step 키워드 감지 |

---

## 복원력 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 차이점 |
|----------------|---------|----------|-------|
| Middleware Guard | ResilienceMiddleware | `core/middleware/resilience.py:47` | 양파 패턴 미들웨어로 모든 LLM 호출 감싸기 |
| Retry Policy | RetryWithBackoff | `core/middleware/resilience.py:107` | 지수 백오프, 장애 유형별 재시도 횟수 |
| Fallback | _try_fallback() | `core/middleware/resilience.py:125` | 하위 티어 모델 자동 전환 |
| Abort / Safe Stop | abort checkpoint | `core/middleware/resilience.py:94, 121` | before/after 양측에서 중단 조건 검사 |
| Stall Detection | StallDetector | `core/stall_detector.py:39` | 슬라이딩 윈도우 + 다양성 메트릭 (CONTINUE/WARN/FORCE_EXIT) |
| Error Hierarchy | AgentError → 하위 클래스 | `core/exceptions.py:20` | SubAgentError, ToolCallError, ResilienceError 등 구조화 |
| Budget Guard | BudgetExceededError | `core/exceptions.py:111` | 비용 예산 초과 시 즉시 safe stop |
| Fallback Exhausted | ModelFallbackExhaustedError | `core/exceptions.py:137` | 모든 fallback 모델 소진 시 에러 |

---

## 모델 체계 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 차이점 |
|----------------|---------|----------|-------|
| Model Selection | ModelTier Enum (4-Tier) | `core/model_tiers.py:33` | REASONING/STRONG/DEFAULT/FAST |
| SLM 활용 | FAST 티어 (qwen-turbo) | `core/model_tiers.py:46` | 분류, 파싱, 메모리 축적에 SLM 사용 |
| Model Factory | create_chat_model() | `core/model_tiers.py:383` | LiteLLM 프록시/SDK 모드 자동 선택 |
| Purpose Mapping | recommend_tier_for_purpose() | `core/model_tiers.py:495` | 가중 점수 기반 목적별 모델 추천 |
| Cost Analysis | estimate_cost() | `core/model_tiers.py:556` | 토큰당 비용 추정 |
| Fallback Chain | build_fallback_chain() | `core/model_tiers.py:661` | 티어별 fallback 순서 자동 구성 |

---

## 도구 시스템 매핑

| DeepAgents 개념 | 우리 구현 | 코드 위치 | 차이점 |
|----------------|---------|----------|-------|
| Tool Server | MCP FastMCP Server | `mcp_servers/code_tools/server.py:26` | MCP 프로토콜 기반 (7개 도구) |
| File I/O | read_file(), write_file() | `mcp_servers/code_tools/server.py:96, 121` | workspace 경계 강제 (_safe_path) |
| Code Search | search_code() | `mcp_servers/code_tools/server.py:167` | 정규식 기반 코드 검색 |
| Workspace Guard | _safe_path() | `mcp_servers/code_tools/server.py:56` | 경로 탈출 방지 (보안) |

---

## 요약: 동등 역량 충족 여부

| DeepAgents 요구 역량 | 충족 여부 | 근거 |
|---------------------|---------|------|
| 장기 메모리 저장/조회/갱신/정정 | ✅ | 6종 CoALA 메모리 + JSONL 영속성 + CLI 정정 명령 |
| SubAgent 런타임 생성/상태 추적/종료 | ✅ | 프로세스 격리 + 8단계 상태 전이 + Puppeteer 선택 |
| retry/fallback/safe stop 방어 전략 | ✅ | ResilienceMiddleware + StallDetector + 에러 계층 |
| SLM 활용 | ✅ | FAST 티어(qwen-turbo)로 분류/파싱/메모리 축적 |
| 미들웨어 기반 횡단 관심사 | ✅ | 양파 패턴 MiddlewareChain (Resilience → Memory → Window) |
| 모델 선택 기준 명시 | ✅ | 4-Tier + 목적별 매핑 + 비용 분석 |
