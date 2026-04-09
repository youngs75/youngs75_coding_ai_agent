# 요구사항 ↔ 구현 매핑

본 문서는 `AI_CODING_AGENT_HARESS_REQUIREMENTS.md`의 핵심 요구사항 4개 축을
실제 코드 구현과 매핑합니다.

---

## 3-1. 장기 메모리와 지식 저장 체계

### 메모리 계층 매핑

| 요구사항 계층 | 구현 메모리 타입 | 저장 메서드 | 코드 위치 |
|-------------|----------------|-----------|----------|
| `user/profile` | USER_PROFILE | `MemoryStore.accumulate_user_profile()` | `core/memory/store.py:295-326` |
| `project/context` | SEMANTIC + EPISODIC | `SemanticMemoryLoader.load_all()` + `MemoryStore.put()` | `core/memory/semantic_loader.py:43`, `core/memory/store.py:70` |
| `domain/knowledge` | DOMAIN_KNOWLEDGE | `MemoryStore.accumulate_domain_knowledge()` | `core/memory/store.py:252-290` |

### 추가 메모리 (CoALA 확장)

| 메모리 타입 | 용도 | 저장 메서드 | 코드 위치 |
|------------|------|-----------|----------|
| WORKING | 현재 대화 컨텍스트 | `MemoryAwareState.messages` | `core/memory/state.py:31` |
| PROCEDURAL | 코딩 스킬/패턴 누적 | `MemoryStore.accumulate_skill()` | `core/memory/store.py:189-227` |
| EPISODIC | 과거 작업 이력 | `MemoryStore.put(type=EPISODIC)` | `core/memory/store.py:70-79` |

### 메모리 생명주기 (무엇을 / 언제 저장 / 언제 조회 / 어디에 / 정정 방법)

| 항목 | 구현 |
|------|------|
| **무엇을 저장** | 사용자 선호, 프로젝트 규칙, 도메인 지식, 코딩 스킬, 작업 이력 |
| **언제 저장** | MemoryMiddleware의 `_try_accumulate()`가 LLM 응답 후 자동 축적 (`core/middleware/memory.py:213`) |
| **언제 조회** | MemoryMiddleware의 `_inject_store_memories()`가 LLM 요청 전 자동 주입 (`core/middleware/memory.py:171`) |
| **어디에 지속** | `.ai/memory/*.jsonl` 파일 (타입별 분리), `_append_to_file()` (`core/memory/store.py:368`) |
| **정정 방법** | CLI `/correct` 명령 (`cli/commands.py:279`), `MemoryStore.update()` (`core/memory/store.py:140`) |

### 검색 메커니즘

| 구성요소 | 설명 | 코드 위치 |
|---------|------|----------|
| TwoStageSearch | CoALA 패턴: 1단계 태그 필터링 → 2단계 BM25 랭킹 | `core/memory/search.py:114-155` |
| TagBasedSearch | 메모리 타입/태그 기반 경량 필터링 | `core/memory/search.py:31-60` |
| ContentBasedSearch | BM25 점수 기반 콘텐츠 랭킹 | `core/memory/search.py:64-110` |
| Novelty Filter | Jaccard 유사도 기반 중복 방지 | `core/memory/store.py:330-358` |

### CLI 메모리 명령어

| 명령어 | 기능 | 코드 위치 |
|--------|------|----------|
| `/memory [type]` | 메모리 조회 (타입별 필터링) | `cli/commands.py:187-214` |
| `/remember <text>` | 도메인 지식 저장 | `cli/commands.py:216-232` |
| `/remember-preference <text>` | 사용자 선호 저장 | `cli/commands.py:234-252` |
| `/forget <id>` | 메모리 삭제 | `cli/commands.py:254-277` |
| `/correct <id> <text>` | 메모리 정정 | `cli/commands.py:279` |

### 로그/데모 시나리오

1. **사용자가 타입 힌트 강제 규칙 입력 → project/context 저장 → 다음 코드 생성에 반영**
   - 사용자: `/remember 우리 팀은 모든 Python 함수에 타입 힌트를 강제한다`
   - `_handle_remember()` → `accumulate_domain_knowledge()` → `.ai/memory/domain_knowledge.jsonl`에 저장
   - 다음 코드 생성 시 `MemoryMiddleware._inject_store_memories()`가 DOMAIN_KNOWLEDGE를 검색하여 시스템 프롬프트에 주입
   - LLM이 타입 힌트가 포함된 코드를 생성

2. **사용자 선호 저장 → 세션 간 유지**
   - 사용자: `/remember-preference 출력은 항상 한국어 설명 + 영어 코드`
   - `accumulate_user_profile()` → `.ai/memory/user_profile.jsonl`에 저장
   - 다음 세션에서 `_load_persisted()` (`core/memory/store.py:396`)가 자동 로드

3. **자동 지식 축적 (SLM 기반)**
   - LLM 응답 후 `MemoryMiddleware._try_accumulate()` (`core/middleware/memory.py:213`)이 FAST 티어(SLM)로 지식 추출
   - 추출된 규칙/패턴을 자동으로 적절한 메모리 타입에 저장

### 한계

- BM25 검색은 의미적 유사도가 아닌 키워드 기반이므로, 동의어/유사 표현에 취약
- 메모리 간 충돌 해소는 타임스탬프 기반 최신 우선이며, 의미적 충돌 감지는 미구현
- 메모리 TTL/만료 정책이 없어 오래된 메모리가 무한 누적될 수 있음

---

## 3-2. 동적 SubAgent 수명주기 관리

### 상태 전이 매핑

| 요구사항 상태 | 구현 상태 | 전이 메서드 | 코드 위치 |
|-------------|---------|-----------|----------|
| `created` | `CREATED` | `SubAgentRegistry.create_instance()` | `core/subagents/registry.py:148` |
| `assigned` | `ASSIGNED` | `transition_state()` (작업 할당 시) | `core/subagents/registry.py:178` |
| `running` | `RUNNING` | `SubAgentProcessManager.spawn()` | `core/subagents/process_manager.py:62` |
| `blocked` | `BLOCKED` | `transition_state()` (타임아웃/의존성) | `core/subagents/registry.py:178` |
| `completed` | `COMPLETED` | `wait()` 결과 파싱 후 전이 | `core/subagents/process_manager.py:177` |
| `failed` | `FAILED` | `wait()` 에러 감지 후 전이 | `core/subagents/process_manager.py:177` |
| `cancelled` | `CANCELLED` | `cancel()` (상위 에이전트 중단) | `core/subagents/process_manager.py:324` |
| `destroyed` | `DESTROYED` | `_destroy_instance()` (자원 정리) | `core/subagents/process_manager.py:365` |

### SubAgent 메타데이터

| 필드 | 구현 | 코드 위치 |
|------|------|----------|
| `agent_id` | UUID 기반 인스턴스 ID | `core/subagents/schemas.py` |
| `role` / `specialty` | 에이전트 유형 (coding_assistant, deep_research 등) | `core/subagents/worker.py:30` |
| `task_summary` | 작업 설명 | `core/subagents/schemas.py` |
| `parent_id` | Orchestrator 세션 ID | `core/subagents/schemas.py` |
| `state` | 8단계 상태 | `core/subagents/registry.py:178` |
| `created_at` / `updated_at` | 타임스탬프 | `core/subagents/schemas.py` |

### Puppeteer 선택 알고리즘

| 구성요소 | 설명 | 코드 위치 |
|---------|------|----------|
| 점수 공식 | `R = quality - λ·cost` | `core/subagents/registry.py:74` |
| 품질 산출 | `0.7 × task_rate + 0.3 × overall_rate` | `core/subagents/registry.py:315` |
| 이력 기록 | `record_usage()` | `core/subagents/registry.py:101` |
| 성공률 조회 | `get_success_rate()` | `core/subagents/registry.py:105` |
| 실패 사유 | `get_failure_reasons()` | `core/subagents/registry.py:117` |

### 프로세스 격리

| 구성요소 | 설명 | 코드 위치 |
|---------|------|----------|
| 프로세스 생성 | `asyncio.create_subprocess_exec` | `core/subagents/process_manager.py:62` |
| 컨텍스트 필터링 | 필요한 정보만 전달 (오염 방지) | `core/subagent_context.py:37` |
| Graceful Shutdown | SIGTERM → 대기 → SIGKILL | `core/subagents/process_manager.py:454` |
| 좀비 프로세스 정리 | `_reap_zombies()` | `core/subagents/process_manager.py:353` |
| 자원 정리 | `_finalize_resource()` | `core/subagents/process_manager.py:387` |

### 동적 생성 경로

| 트리거 | 생성 에이전트 | 코드 위치 |
|--------|-------------|----------|
| 코드 생성 요청 | CodingAssistant | `agents/orchestrator/agent.py:235` |
| 리서치 요청 | DeepResearch | `agents/orchestrator/agent.py:235` |
| 파일 조회 요청 | SimpleReAct | `agents/orchestrator/agent.py:235` |
| 복합 작업 | Planner → 다중 SubAgent | `agents/orchestrator/agent.py:182` |
| 복잡도 판별 | `_detect_complexity()` | `agents/orchestrator/agent.py:89` |

### 로그/데모 시나리오

- 사용자: "문서 요구사항 정리 후 코드 스캐폴딩도 해줘"
  1. Orchestrator `classify()` → 복합 작업 판별
  2. `_invoke_planner()` → PlannerAgent가 다단계 계획 수립
  3. 각 단계별 `_invoke_local_agent()` → SubAgent 동적 spawn
  4. 각 SubAgent: `CREATED → ASSIGNED → RUNNING → COMPLETED → DESTROYED`
  5. 실패 시: `RUNNING → FAILED`, 다른 역할의 SubAgent로 재시도

### 한계

- 병렬 SubAgent 실행은 구현되어 있으나, 의존성 기반 DAG 스케줄링은 미구현
- SubAgent 간 직접 통신 불가 (항상 Orchestrator를 경유)
- 프로세스 격리로 인한 SubAgent 시작 오버헤드 존재

---

## 3-3. Agentic Loop 복원력과 안전성

### 장애 유형별 처리 행렬

| 장애 유형 | 감지 | 재시도 | fallback | 코드 위치 |
|----------|------|--------|----------|----------|
| 모델 무응답/지연 | RetryWithBackoff 타임아웃 | 1-2회 (지수 백오프) | 하위 티어 모델 전환 | `core/middleware/resilience.py:107` |
| 반복 무진전 루프 | StallDetector (슬라이딩 윈도우 + 다양성 메트릭) | 0회 (즉시 전략 전환) | FORCE_EXIT | `core/stall_detector.py:66` |
| 잘못된 tool call | 예외 캐치 + ToolCallError | 1회 (재작성) | — | `core/exceptions.py:68` |
| SubAgent 실패 | `failed` 상태 전이 + SubAgentError | 역할별 1회 | 다른 역할 SubAgent | `core/exceptions.py:28`, `core/subagents/process_manager.py:266` |
| 외부 API 오류 | ResilienceMiddleware retry | 정책 기반 1-3회 | 대체 모델 | `core/middleware/resilience.py:125` |
| 예산 초과 | BudgetExceededError | 0회 | safe stop | `core/exceptions.py:111` |
| safe stop 필요 | 권한 부족, 위험 작업 | 0회 | 즉시 중단 | `core/exceptions.py:92` |

### 복원력 미들웨어 상세

| 구성요소 | 설명 | 코드 위치 |
|---------|------|----------|
| ResilienceMiddleware | retry + fallback + abort 체크포인트 | `core/middleware/resilience.py:47` |
| before: abort check | 요청 전 중단 조건 확인 | `core/middleware/resilience.py:94` |
| retry with backoff | 지수 백오프 재시도 | `core/middleware/resilience.py:107` |
| fallback attempt | 실패 시 대체 모델 시도 | `core/middleware/resilience.py:115` |
| after: abort check | 응답 후 중단 조건 확인 | `core/middleware/resilience.py:121` |
| `_try_fallback()` | Fallback 모델 선택 및 호출 | `core/middleware/resilience.py:125` |
| `_resolve_policy()` | 장애 유형별 정책 결정 | `core/middleware/resilience.py:154` |

### Stall Detector (무진전 루프 감지)

| 구성요소 | 설명 | 코드 위치 |
|---------|------|----------|
| StallDetector | 슬라이딩 윈도우 기반 반복 감지 | `core/stall_detector.py:39` |
| `record_and_check()` | 행동 기록 + 정체 감지 (동일 호출 + 다양성 체크) | `core/stall_detector.py:66` |
| `_check_diversity()` | 고유 액션 비율 계산 | `core/stall_detector.py:114` |
| StallAction Enum | CONTINUE / WARN / FORCE_EXIT | `core/stall_detector.py:22` |

### 에러 타입 계층

| 에러 | 설명 | 코드 위치 |
|------|------|----------|
| AgentError | 모든 에이전트 에러의 베이스 | `core/exceptions.py:20` |
| SubAgentError | SubAgent 실패 (상태, 역할 포함) | `core/exceptions.py:28` |
| MemoryError | 메모리 연산 실패 | `core/exceptions.py:55` |
| ToolCallError | 도구 호출 실패 | `core/exceptions.py:68` |
| ResilienceError | 복원력 관련 에러 베이스 | `core/exceptions.py:92` |
| StallDetectedError | 무진전 루프 감지 | `core/exceptions.py:96` |
| BudgetExceededError | 비용 예산 초과 | `core/exceptions.py:111` |
| ModelFallbackExhaustedError | 모든 fallback 모델 소진 | `core/exceptions.py:137` |

### Fallback 체인

| 구성요소 | 설명 | 코드 위치 |
|---------|------|----------|
| `build_fallback_chain()` | 티어별 fallback 모델 순서 생성 | `core/model_tiers.py:661` |
| `_try_fallback()` | ResilienceMiddleware의 fallback 시도 | `core/middleware/resilience.py:125` |
| 환경 변수 오버라이드 | `REASONING_MODEL` 등으로 모델 교체 | `core/model_tiers.py:287` |

### 로그/데모 시나리오

- **모델 무응답**: LLM 호출 30초 초과 → ResilienceMiddleware가 1회 재시도 → 실패 시 FAST 티어로 fallback → 재실패 시 `ModelFallbackExhaustedError`와 함께 safe stop
- **무진전 루프**: StallDetector가 최근 5회 행동의 다양성 0.2 미만 감지 → `StallAction.FORCE_EXIT` → 루프 종료 + 사용자에게 상태 보고
- **SubAgent 실패**: CodingAssistant가 FAILED → `spawn_and_wait()`가 자동 재시도 1회 → 재실패 시 다른 역할의 SubAgent 투입 검토

### 한계

- Fallback 체인이 동일 프로바이더(DashScope) 내에서만 동작하며, 크로스 프로바이더 fallback은 수동 설정 필요
- StallDetector의 다양성 임계값이 하드코딩되어 있으며, 작업 유형별 동적 조정은 미구현
- resume metadata 저장은 LangGraph 체크포인터에 의존하며, 별도 resume 파일은 미구현

---

## 3-4(추가). SLM 서빙 및 에이전트 내 활용

### 4-Tier 모델 체계

| 티어 | 모델 (DashScope) | 비용 (1M tok) | 용도 | 코드 위치 |
|------|-----------------|--------------|------|----------|
| REASONING | qwen-max | $1.2/$6.0 | 계획/아키텍처 | `core/model_tiers.py:43` |
| STRONG | qwen-coder-plus | $0.46/$1.38 | 코드 생성 | `core/model_tiers.py:44` |
| DEFAULT | qwen-plus | $0.4/$1.2 | 검증/분석 | `core/model_tiers.py:45` |
| FAST (SLM) | qwen-turbo | $0.04/$0.08 | 파싱/분류 | `core/model_tiers.py:46` |

### SLM 활용 지점

| 활용 지점 | 티어 | 이유 | 코드 위치 |
|----------|------|------|----------|
| 요청 분류 (classify) | FAST | JSON 구조화 출력만 필요 | `agents/orchestrator/agent.py:114` |
| 복잡도 판별 | FAST | 간단한 분류 작업 | `agents/orchestrator/agent.py:89` |
| 메모리 지식 추출 | FAST | 키워드/엔티티 추출 | `core/middleware/memory.py:213` |
| 목적별 티어 추천 | — | 가중 점수 기반 자동 선택 | `core/model_tiers.py:495` |
| 비용 추정 | — | 토큰당 비용 계산 | `core/model_tiers.py:556` |
| 트레이드오프 분석 | — | 티어 간 비교 | `core/model_tiers.py:588` |

### 로그/데모 시나리오

- 사용자 요청 "Python 계산기 만들어줘" 수신
  1. FAST(SLM): classify → `{task_type: "generate", complexity: "simple"}`
  2. REASONING: Planner가 아키텍처 설계 + 페이즈 분해
  3. STRONG: CodingAssistant가 코드 생성
  4. DEFAULT: Verifier가 코드 검증
  5. FAST(SLM): MemoryMiddleware가 응답에서 지식 추출/축적

### 한계

- SLM(qwen-turbo)의 도구 호출 정확도가 상위 티어 대비 낮을 수 있음
- 로컬 SLM 서빙(vLLM, Ollama 등)은 미구현, 클라우드 API 의존
- 티어 자동 전환 기준이 정적이며, 런타임 성능 측정 기반 동적 전환은 미구현
