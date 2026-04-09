# Coding AI Agent Harness — 현황 분석 및 보강 계획서

> 이 문서는 코드에 직접 접근할 수 없는 외부 리뷰어를 위해 작성되었습니다.
> 프로젝트의 현재 구현 상태를 코드 수준으로 상세히 설명하고, 요구사항 대비 갭과 보강 계획을 제시합니다.

---

## 1. 프로젝트 개요

### 1.1 목표
MCP(Model Context Protocol) 도구 기반의 코드 생성/검증/실행 에이전트 프레임워크.
사용자의 자연어 요청을 받아 코드를 생성하고, 테스트하고, 검증까지 자동으로 수행합니다.

### 1.2 아키텍처 개요
```
사용자 요청
    ↓
[Orchestrator] ── classify → delegate → respond
    ↓                          ↓
[PlannerAgent]          [CodingAssistantAgent]
  (계획 수립)              (코드 생성 + 테스트)
                               ↓
                        [VerificationAgent]
                          (코드 검증)
```

- **Orchestrator**: 요청 분류 → 적절한 SubAgent에 위임 → 결과 취합
- **PlannerAgent**: 복잡한 요청의 아키텍처 설계 + 태스크 분해
- **CodingAssistantAgent**: 실제 코드 생성 + 파일 저장 + 테스트 실행
- **VerificationAgent**: 생성된 코드의 품질 검증 (lint, test, LLM review)

### 1.3 기술 스택
- **프레임워크**: LangGraph (상태 그래프 기반 에이전트 오케스트레이션)
- **LLM**: OpenRouter 경유 Qwen 시리즈 (오픈소스 모델)
- **도구 연동**: MCP (Model Context Protocol) — 파일 I/O, 코드 검색, 코드 실행
- **프로토콜**: A2A (Agent-to-Agent) 프로토콜 지원
- **배포**: Docker Compose
- **관측성**: Langfuse v4
- **평가**: DeepEval 메트릭

### 1.4 4-Tier 모델 체계

| Tier | 모델 (OpenRouter) | 용도 | 비용 | SLM 여부 |
|------|-------------------|------|------|---------|
| REASONING | qwen3-max | 계획/아키텍처 설계 | 높음 | X |
| STRONG | qwen3-coder-plus | 코드 생성/도구 호출 | 중간 | X |
| DEFAULT | qwen3-coder-next | 검증/분석 | 중간 | X |
| FAST | qwen3.5-flash | 파싱/분류 | $0.04~0.07/1M tok | **O (SLM)** |

purpose → tier 자동 매핑:
- `planning` → REASONING
- `generation` → STRONG
- `verification` → DEFAULT
- `parsing`, `tool_planning` → FAST (SLM)

### 1.5 디렉토리 구조
```
coding_agent/
├── core/
│   ├── memory/          # CoALA 메모리 시스템
│   │   ├── schemas.py   # MemoryType(6종), MemoryItem
│   │   └── store.py     # MemoryStore (CRUD + 영속화 + 누적)
│   ├── skills/          # 3-Level 스킬 시스템
│   ├── middleware/       # LLM 미들웨어 체인
│   ├── subagents/       # SubAgent 동적 관리
│   │   ├── schemas.py   # 8단계 상태머신, SubAgentInstance
│   │   ├── registry.py  # Puppeteer 선택 알고리즘
│   │   ├── process_manager.py  # 프로세스 생명주기
│   │   └── worker.py    # 워커 프로세스 진입점
│   ├── model_tiers.py   # 4-Tier 모델 관리 + Fallback 체인
│   ├── resilience.py    # FailureMatrix + RetryWithBackoff + ModelFallbackChain
│   ├── stall_detector.py    # 반복 무진전 감지
│   ├── abort_controller.py  # 협력적 Safe Stop
│   └── turn_budget.py       # 턴 예산 추적
├── agents/
│   ├── orchestrator/    # Orchestrator (라우팅 + Multi-phase 실행)
│   ├── planner/         # PlannerAgent (계획 수립)
│   ├── coding_assistant/  # CodingAssistant (코드 생성)
│   └── verifier/        # VerificationAgent (코드 검증)
├── a2a/                 # A2A 프로토콜 브릿지
├── mcp_servers/         # MCP 서버 (파일 I/O, 코드 실행)
├── eval_pipeline/       # DeepEval 평가 파이프라인
└── cli/                 # 대화형 CLI
```

---

## 2. 요구사항별 현재 구현 상태

### 2.1 축 1: 장기 메모리와 지식 저장 체계

#### 요구사항 매핑

| 요구사항 3계층 | 우리 구현 | MemoryType enum |
|-------------|----------|-----------------|
| user/profile | ✅ 구현됨 | `USER_PROFILE` |
| project/context | ✅ 구현됨 | `SEMANTIC` |
| domain/knowledge | ✅ 구현됨 | `DOMAIN_KNOWLEDGE` |

추가로 `WORKING`, `EPISODIC`, `PROCEDURAL` 3개 타입이 더 있어 총 6종 (CoALA 패턴).

#### MemoryStore 핵심 구현 (store.py)

```python
class MemoryStore:
    # CRUD
    put(item: MemoryItem)           # 메모리 저장
    get(item_id, memory_type)       # 단건 조회
    search(query, memory_type, tags, limit)  # 2단계 검색 (태그필터 → BM25)
    update(item_id, memory_type, content, tags)  # 정정(갱신) + updated_at
    delete(item_id, memory_type)    # 삭제

    # 전문 누적 메서드
    accumulate_domain_knowledge(content, tags, source)
        # → Jaccard 유사도 0.8 기반 중복 판단
        # → 기존 항목이면 update, 신규면 put
    accumulate_user_profile(content, tags, source)
        # → 동일 태그 항목 갱신
    accumulate_skill(code, description, tags)
        # → Jaccard 0.7 기반 novelty 필터

    # 영속화
    _persist_path(type) → "{workspace}/.ai/memory/{type}.jsonl"
    _append_to_file(item)   # JSONL 추가
    _rewrite_file(type)     # 전체 재작성 (update/delete 시)
    _load_persisted()       # 시작 시 파일에서 로드
```

**영속화 대상**: PROCEDURAL, USER_PROFILE, DOMAIN_KNOWLEDGE, SEMANTIC (4개)

#### 메모리 활용 경로 (CodingAssistantAgent)

```
_parse_request()
    ↓
_retrieve_memory()  ← 여기서 메모리 검색
    ├── Episodic: 유사 과거 작업 검색
    ├── Procedural: 관련 코딩 스킬 검색
    ├── User Profile: 사용자 선호 검색
    └── Domain Knowledge: 도메인 지식 검색
    ↓
_generate_final()  ← 검색 결과가 프롬프트에 주입됨
```

#### 메모리 정정(Correction) 메커니즘

- `MemoryStore.update()`: 기존 항목의 content, tags, metadata 갱신
- `updated_at` 타임스탬프 자동 기록
- 영속 타입은 파일 전체 재작성으로 반영
- 중복 방지: Jaccard 유사도 기반 novelty 필터 (Voyager 패턴)

#### ✅ 구현된 것
- 3계층 메모리 타입 정의 및 분리
- JSONL 파일 영속화 (세션 간 유지)
- 2단계 검색 (태그 + BM25)
- domain_knowledge 누적 + 중복 병합
- user_profile 선호 갱신
- 정정(update) + 타임스탬프

#### ❌ 갭 (미구현)
1. **MemoryMiddleware 파일 미존재** — LangGraph 미들웨어로 메모리 자동 주입하는 레이어가 없음. 현재는 CodingAssistant의 `_retrieve_memory()` 노드에서 직접 검색.
2. **Orchestrator → SubAgent 메모리 전달 안 됨** — `_invoke_local_agent()`에서 MemoryStore 객체를 SubAgent에 전달하지 않음. SubAgent가 user_profile, domain_knowledge에 접근 불가.
3. **사용자 피드백 수집 메커니즘 없음** — `accumulate_user_profile()`은 존재하지만, 사용자 피드백을 감지하고 자동 저장하는 트리거가 없음.

---

### 2.2 축 2: 동적 SubAgent 수명주기 관리

#### 8단계 상태 머신 (schemas.py)

```
CREATED → ASSIGNED → RUNNING → COMPLETED → DESTROYED
                       ↓                        ↑
                     BLOCKED ────────────────────┘
                       ↓
                     FAILED ──→ (retry: ASSIGNED) → DESTROYED

CANCELLED → DESTROYED
```

유효한 전이만 허용하는 `VALID_TRANSITIONS` dict 정의:
```python
VALID_TRANSITIONS = {
    CREATED:   {ASSIGNED, CANCELLED, DESTROYED},
    ASSIGNED:  {RUNNING, BLOCKED, CANCELLED},
    RUNNING:   {COMPLETED, FAILED, BLOCKED, CANCELLED},
    BLOCKED:   {RUNNING, FAILED, CANCELLED},
    COMPLETED: {DESTROYED},
    FAILED:    {ASSIGNED, DESTROYED},  # ASSIGNED = 재시도
    CANCELLED: {DESTROYED},
    DESTROYED: {},  # 종료 상태
}
```

#### SubAgent 메타데이터 (SubAgentInstance)

```python
class SubAgentInstance(BaseModel):
    agent_id: str       # UUID hex (자동 생성)
    spec_name: str      # 에이전트 사양 이름
    role: str           # 역할 설명
    task_summary: str   # 작업 요약
    parent_id: str | None  # 부모 에이전트 ID
    state: SubAgentStatus  # 현재 상태
    created_at: datetime
    updated_at: datetime
    error_message: str | None
    result_summary: str | None
```

#### 동적 생성 흐름 (ProcessManager)

```
1. registry.create_instance(spec_name, task_summary, role, parent_id)
   → SubAgentInstance 생성 (CREATED)

2. 임시파일로 task_message, task_plan 전달

3. 상태: CREATED → ASSIGNED

4. asyncio.create_subprocess_exec()
   → python -m coding_agent.core.subagents.worker --agent-type ...

5. 상태: ASSIGNED → RUNNING

6. wait(agent_id, timeout_s)
   → stdout JSON 파싱 → SubAgentResult
   → 성공: COMPLETED, 실패: FAILED

7. _destroy_instance(agent_id)
   → DESTROYED + 프로세스 객체 제거 + 임시파일 정리
```

#### Puppeteer 선택 알고리즘 (registry.py)

```python
score = R(quality) - λ * C(cost_weight)
quality = 0.7 * task_success_rate + 0.3 * overall_success_rate
```
- 태스크 타입별 성공률 추적
- 비용 민감도(λ) 조절 가능

#### 워커 프로세스 (worker.py)

3개 에이전트 타입 등록:
- `coding_assistant`: 코드 생성 (SkillRegistry 초기화 O)
- `deep_research`: 심층 조사 (SkillRegistry X)
- `simple_react`: 단순 ReAct (SkillRegistry X)

stdout JSON 프로토콜로 결과 반환:
```json
{
  "status": "completed|failed",
  "result": "생성된 코드 또는 보고서",
  "written_files": ["path/to/file.py"],
  "test_passed": true,
  "exit_reason": "normal",
  "duration_s": 45.2,
  "token_usage": {},
  "error": null
}
```

#### ✅ 구현된 것
- 8단계 상태 머신 + 유효 전이 검증
- 전체 메타데이터 (agent_id, role, task_summary, parent_id, state, timestamps)
- 프로세스 기반 동적 생성/정리 (asyncio)
- 우아한 종료 (SIGTERM → 대기 → SIGKILL)
- 임시파일 자동 정리
- 타임아웃 처리 + 좀비 감지
- Puppeteer 선택 + 성공률 추적
- 이벤트 로그 기록 (SubAgentEvent)
- 동시 실행 제한 (max_concurrent=5)

#### ❌ 갭 (미구현)
1. **BLOCKED 상태 자동 복구 없음** — 의존 작업 완료 후 자동 재개 메커니즘이 없음
2. **SubAgent 간 의존성 관리 없음** — 선행 작업 완료 대기 로직 미구현
3. **결과 영속화 없음** — SubAgentResult가 메모리에만 존재, 프로세스 종료 시 손실
4. **retry_count/max_retries 필드 없음** — 재시도 횟수 추적 불가
5. **token_usage 추적 없음** — worker에서 항상 빈 dict 반환

---

### 2.3 축 3: Agentic Loop 복원력과 안전성

#### 구현된 방어 메커니즘

**1. FailureMatrix (resilience.py)**
```python
FailureType (7종):
  MODEL_TIMEOUT, STUCK_LOOP, BAD_TOOL_CALL,
  SUBAGENT_FAILURE, EXTERNAL_API_ERROR,
  MODEL_FALLBACK_NEEDED, SAFE_STOP

FailurePolicy (각 유형별):
  max_retries, backoff_base, backoff_multiplier, backoff_max,
  fallback_enabled, user_visible_status, safe_stop_condition
```

장애 유형별 정책 예시:
| 장애 유형 | 재시도 | 백오프 | Fallback | Safe Stop 조건 |
|----------|--------|--------|----------|---------------|
| MODEL_TIMEOUT | 2회 | 2s, ×2, max 30s | O | 재시도 한도 초과 |
| STUCK_LOOP | 1회 | 1s | O | 전략 전환 후 진전 없음 |
| BAD_TOOL_CALL | 1회 | 1s | X | 동일 오류 반복 |
| SUBAGENT_FAILURE | 1회 | 2s | O | 대체 경로도 실패 |
| EXTERNAL_API_ERROR | 3회 | 1s, ×2, max 60s | O | 비용 과도 또는 상태 불명 |
| SAFE_STOP | 0회 | - | X | 즉시 중단 |

**2. RetryWithBackoff (resilience.py)**
```python
class RetryWithBackoff:
    async def execute(func, *args, **kwargs):
        # 지수 백오프 재시도 wrapper
        # calculate_delay = base × multiplier^attempt (max 상한)
```

**3. ModelFallbackChain (resilience.py)**
```python
class ModelFallbackChain:
    async def invoke_with_fallback(messages):
        # STRONG → DEFAULT → FAST 순서로 시도
        # 모든 모델 실패 시 마지막 예외 raise
```

**4. StallDetector (stall_detector.py)**
```python
class StallDetector:
    record_and_check(tool_name, tool_args) → StallAction:
        # 슬라이딩 윈도우(10) 기반 반복 감지
        # 동일 (tool_name, args_hash) 패턴 추적
        # 2회 반복: WARN, 3회 반복: FORCE_EXIT
```

**5. AbortController (abort_controller.py)**
```python
class AbortController:
    abort(reason, message)    # 중단 신호 발생
    check_or_raise()          # 체크포인트에서 AbortError 발생
    # 5가지 중단 사유: USER_INTERRUPT, STALL_DETECTED,
    #   BUDGET_EXCEEDED, TURN_LIMIT, TIMEOUT
```

**6. TurnBudgetTracker (turn_budget.py)**
```python
class TurnBudgetTracker:
    record_llm_call(output_tokens) → BudgetVerdict:
        # max_llm_calls(15) 도달: STOP
        # 연속 저효율(500 tokens 미만) N회: STOP
        # 정상: OK 또는 WARN_DIMINISHING
```

#### 실제 적용 상태 (Critical!)

| 컴포넌트 | 코드 정의 | 에이전트에서 실제 호출 |
|---------|----------|-------------------|
| FailureMatrix | ✅ resilience.py | **❌ 미적용** — 정책만 정의, 오류 처리에서 사용 안 함 |
| RetryWithBackoff | ✅ resilience.py | **❌ 미사용** — LLM 호출 실패 시 자동 재시도 없음 |
| ModelFallbackChain | ✅ resilience.py | **❌ 미사용** — 단일 모델만 사용 |
| build_fallback_chain() | ✅ model_tiers.py | **❌ 미호출** — 함수 존재하지만 호출처 없음 |
| StallDetector | ✅ stall_detector.py | **⚠️ 초기화만** — CodingAssistant에서 생성+리셋하지만 record_and_check() 호출 없음 |
| AbortController | ✅ abort_controller.py | **❌ 미사용** — 인스턴스 생성조차 안 됨 |
| TurnBudgetTracker | ✅ turn_budget.py | **✅ 부분 적용** — CodingAssistant._execute_code()에서만 사용 |

#### ✅ 구현된 것 (실제 동작)
- TurnBudgetTracker: _execute_code 노드에서 LLM 호출 예산 추적 + STOP 판정
- CodingAssistant: 테스트 실패 시 최대 반복(max_iterations=11) 후 종료
- ProcessManager: 타임아웃(300s) + 좀비 프로세스 정리
- 실패 패턴 학습: 최대 반복 도달 시 에러 유형을 Procedural Memory에 저장

#### ❌ 갭 (미구현 — 가장 심각한 문제)
1. **FailureMatrix가 실제 코드 경로에 적용되지 않음** — 정책은 완벽히 정의되어 있지만, 어떤 에이전트의 어떤 try-except에서도 참조하지 않음
2. **ModelFallbackChain이 호출되지 않음** — build_fallback_chain() 함수와 invoke_with_fallback() 메서드가 있지만 실제 LLM 호출에서 사용되지 않음
3. **StallDetector가 도구 호출 후 체크하지 않음** — 초기화와 리셋만 되고, 실제 도구 호출 후 record_and_check() 호출이 없음
4. **AbortController가 아예 인스턴스화되지 않음** — 코드는 완성되어 있지만 사용처가 없음

---

### 2.4 축 4: SLM 서빙 및 활용

#### 에이전트별 모델 Tier 사용 현황

| 에이전트 | 노드 | 사용 Tier | SLM? |
|---------|------|----------|------|
| CodingAssistant | parse_request | DEFAULT | X |
| | execute_code | **FAST** (tool_planning) | **O** |
| | generate_final | STRONG | X |
| | verify | DEFAULT | X |
| Planner | analyze_task | REASONING | X |
| | research | REASONING | X |
| Verifier | llm_review | DEFAULT | X |
| Orchestrator | classify | DEFAULT | X |

#### ✅ 구현된 것
- FAST tier = qwen3.5-flash (SLM)로 정의됨
- CodingAssistant의 tool_planning 노드에서 FAST 사용
- purpose → tier 자동 매핑 시스템 존재
- 비용/성능 분석 유틸리티 (estimate_cost, analyze_tier_tradeoffs)

#### ❌ 갭
1. **Orchestrator classify가 DEFAULT 사용** — 단순 라우팅인데 DEFAULT(중간 비용) 모델 사용. FAST(SLM)로 충분함
2. **Planner가 모든 노드에서 REASONING 사용** — analyze_task는 STRONG 또는 DEFAULT로도 가능
3. **Verifier가 DEFAULT 고정** — lint/test 결과 집계는 FAST로 가능
4. **"모든 Agent에서 SLM 최적화"를 충족 못함** — 채점 기준에서 "SLM으로 Main 및 SubAgent 등 모든 Agent 로직을 최적화"를 요구하는데, 현재 CodingAssistant의 1개 노드에서만 SLM 사용

---

## 3. 추가 발견 사항

### 3.1 Orchestrator → Planner 미연동
- `_invoke_planner()` 함수가 정의만 되어 있고 실제 호출되지 않음
- delegate 노드에서 PlannerAgent를 거치지 않고 바로 CodingAssistant 호출
- task_plan_structured가 없으면 multi-phase 감지 실패 → 항상 단일 phase로 실행

### 3.2 MemoryMiddleware 파일 부재
- `schemas.py`에서 코멘트로 언급되지만 `middleware.py` 파일 자체가 없음
- 메모리를 LangGraph 미들웨어로 자동 주입하는 레이어 부재

### 3.3 skill_registry 전달 경로
- worker.py: coding_assistant에 대해서만 SkillRegistry 초기화 ✅
- orchestrator._execute_phases_sequentially: 직접 CodingAssistant 생성 시 SkillRegistry 초기화 ✅
- orchestrator._invoke_local_agent → ProcessManager.spawn_and_wait: 정상 전달 확인 필요

---

## 4. 보강 계획

### Phase 1: 핵심 갭 해소 (채점 직결, 병렬 수행 가능)

#### 1-1. Orchestrator → SubAgent 메모리 전달

**현재**: `_invoke_local_agent()`에서 MemoryStore 미전달 → SubAgent가 격리 환경에서 작업
**목표**: SubAgent가 user_profile, domain_knowledge를 참조하여 코드 생성

**수정 범위**:
- `orchestrator/agent.py`: _invoke_local_agent()에 memory_store 파라미터 추가
- `subagents/process_manager.py`: spawn()에 메모리 직렬화 전달 (JSONL 임시파일)
- `subagents/worker.py`: 메모리 로드 후 에이전트에 전달
- 또는: workspace의 `.ai/memory/` 디렉토리를 SubAgent가 직접 읽도록 경로 전달 (더 단순)

**검증**: "사용자가 도메인 규칙 입력 → 저장 → SubAgent가 코드 생성 시 참조" E2E 시나리오

#### 1-2. 복원력 메커니즘 실제 적용

**현재**: FailureMatrix, RetryWithBackoff, ModelFallbackChain, StallDetector, AbortController 모두 코드는 완성되어 있지만 에이전트 루프에서 호출되지 않음
**목표**: 정의된 정책을 실제 에이전트 루프에 통합

**수정 범위**:
- CodingAssistant의 LLM 호출을 RetryWithBackoff로 래핑
- CodingAssistant의 도구 호출 후 StallDetector.record_and_check() 호출 + WARN/FORCE_EXIT 처리
- LLM 호출 경로에 ModelFallbackChain 적용 (STRONG 실패 → DEFAULT → FAST)
- AbortController를 에이전트 메인 루프에 체크포인트로 삽입
- Orchestrator에서 SubAgent 실패 시 FailureMatrix 정책 참조

**검증**: 
- 모델 타임아웃 시 재시도 → fallback 모델 전환 로그
- 반복 무진전 3회 시 FORCE_EXIT 로그
- safe stop 시 현재 상태 + 실패 원인 + 다음 행동 기록

#### 1-3. SLM 전 에이전트 활용 확대

**현재**: CodingAssistant의 tool_planning 1개 노드에서만 FAST(SLM) 사용
**목표**: 모든 에이전트의 적합한 노드에서 SLM 활용

**수정 대상**:
| 에이전트 | 노드 | 현재 | 목표 |
|---------|------|------|------|
| Orchestrator | classify | DEFAULT | **FAST** |
| CodingAssistant | parse_request | DEFAULT | **FAST** |
| Verifier | aggregate_results | DEFAULT | **FAST** |
| Planner | analyze_task (복잡도 판단) | REASONING | **STRONG** (또는 FAST) |

**검증**: 각 에이전트의 모델 tier 변경 후 기능 정상 동작 확인 + 비용 절감 측정

### Phase 2: 증빙 강화 (채점 가시성)

#### 2-1. SubAgent 상태 전이 로그 완성
- 전 구간 상태 전이 로그 기록 확인
- blocked, cancelled 시나리오 데모용 테스트 추가
- SubAgentResult를 JSONL로 영속화

#### 2-2. 메모리 E2E 시나리오 데모/테스트
- user_profile: "타입 힌트 강제" → 다음 코드 생성에 반영
- domain_knowledge: "Silver 등급 환불 수수료 0%" → 결제 로직에 참조
- project_context: "Flask + Vue 스택" → 프레임워크 선택에 반영

#### 2-3. PEP8 + Google Style DocString 정비
- ruff로 PEP8 준수 검증
- 주요 public 클래스/함수에 Google Style docstring 추가

### Phase 3: 문서화

#### 3-1. 요구사항 매핑 문서
각 요구사항별:
- 설계 의도
- 코드 위치 (파일:줄)
- 로그 또는 데모 시나리오
- 아직 남아 있는 한계

---

## 5. 평가 기준별 예상 점수 (보강 전/후)

| 평가 기준 (각 10점) | 보강 전 | 보강 후 목표 | 핵심 개선 |
|---|---|---|---|
| 요구사항 구현 (DeepAgents 개념) | 6~7 | 8~9 | PEP8/DocString + 매핑 문서 |
| 메모리 시스템 활용 | 5~6 | 8~9 | SubAgent 메모리 전달 + E2E 데모 |
| SubAgent 동적 활용 | 5~6 | 8~9 | 복원력 통합 + 상태 전이 로그 |
| SLM 서빙 및 활용 | 4~5 | 7~8 | 전 에이전트 FAST tier 활용 |
| **합계** | **20~24/40** | **31~35/40** | |

---

## 6. 리뷰어에게 드리는 질문

1. **복원력 통합 전략**: FailureMatrix 정책을 각 에이전트에 분산 적용할지, 공통 미들웨어 레이어로 한 곳에서 관리할지 — 어느 접근이 더 적절한가?

2. **메모리 전달 방식**: SubAgent에 메모리를 전달할 때, (A) MemoryStore 객체를 직렬화하여 전달 vs (B) workspace의 .ai/memory/ JSONL 파일을 직접 읽도록 경로만 전달 — 어느 방식이 더 견고한가?

3. **SLM 활용 범위**: Orchestrator의 classify를 FAST(SLM)로 변경하면 라우팅 정확도가 떨어질 수 있음. 채점 기준의 "모든 Agent 로직을 SLM으로 최적화"를 어느 수준까지 해석해야 하는가?

4. **우선순위**: Phase 1의 3개 작업 중 채점 영향이 가장 큰 것은 무엇이라 판단하는가?

5. **누락된 보강 항목**: 요구사항 문서 대비 이 보강 계획에서 빠진 중요한 항목이 있는가?
