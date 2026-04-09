# Architecture Decision Records (ADR)

본 문서는 AI Coding Agent Harness의 핵심 설계 결정을 ADR 형식으로 기록합니다.

---

## ADR-001: LangGraph 기반 에이전트 오케스트레이션

### 상황

AI Coding Agent를 구현하기 위한 에이전트 프레임워크를 선택해야 했습니다.
요구사항은 다음과 같습니다:

- 에이전트 간 상태 전이를 명시적으로 관리
- Human-in-the-Loop (HITL) 지원
- 미들웨어 체인을 통한 횡단 관심사 처리
- 체크포인트 기반 복구

### 결정

**LangGraph 상태 그래프 + interrupt/Command 기반 HITL 패턴을 채택합니다.**

- `BaseGraphAgent`를 Template Method 패턴으로 구현하여, 각 에이전트가 노드와 엣지를 선언적으로 정의
- Orchestrator는 `classify → plan → delegate → respond` 상태 머신으로 동작
- Planner는 `analyze_task → route → research/explore → create_plan` 그래프로 동작

### 이유

1. **상태 전이 명시성**: 각 노드가 명확한 책임을 가지며, 엣지 조건으로 분기를 제어
2. **체크포인트 지원**: `MemorySaver`를 통한 상태 저장/복구로 장애 시 재개 가능
3. **미들웨어 통합**: 그래프 노드 실행 전후에 미들웨어 체인을 삽입하여 메모리 주입, 복원력 처리를 에이전트 코드와 분리
4. **LangChain 생태계**: MCP 어댑터, LiteLLM 프록시 등 도구 통합이 자연스러움

### 대안 검토

| 대안 | 기각 이유 |
|------|----------|
| DeepAgents 직접 사용 | 상태 관리 투명성 부족, 내부 상태 전이를 커스텀하기 어려움 |
| CrewAI | 역할 기반 고정 구조로 동적 SubAgent 생성에 부적합 |
| 순수 asyncio 루프 | 체크포인트, 미들웨어 등을 모두 직접 구현해야 하는 부담 |

### 코드 위치

- `coding_agent/core/base_agent.py` — BaseGraphAgent (Template Method)
- `coding_agent/agents/orchestrator/agent.py` — Orchestrator 상태 그래프
- `coding_agent/agents/planner/agent.py:85` — PlannerAgent 그래프

---

## ADR-002: 프로세스 격리 기반 SubAgent

### 상황

Orchestrator가 SubAgent에 작업을 위임할 때, 실행 방식을 결정해야 했습니다.
SubAgent 실패가 메인 에이전트에 전파되지 않아야 하고, 메모리/컨텍스트가 격리되어야 합니다.

### 결정

**`asyncio.create_subprocess_exec`로 SubAgent를 별도 프로세스로 실행합니다.**

- `SubAgentProcessManager`가 프로세스 생성, 상태 추적, 종료/정리를 관리
- 8단계 상태 전이: `created → assigned → running → blocked → completed/failed → cancelled → destroyed`
- SIGTERM → 대기 → SIGKILL 순서의 graceful shutdown

### 이유

1. **메모리 격리**: 각 SubAgent가 독립 프로세스로 실행되어 메모리 누수가 전파되지 않음
2. **장애 전파 차단**: SubAgent 크래시가 Orchestrator에 영향을 주지 않음
3. **독립적 종료**: 타임아웃 시 프로세스 킬로 확실한 자원 회수
4. **컨텍스트 격리**: `SubagentContextFilter`가 필요한 정보만 명시적으로 전달

### 대안 검토

| 대안 | 기각 이유 |
|------|----------|
| 쓰레드 기반 | GIL 제약 + 메모리 격리 불가 + 장애 전파 |
| 인프로세스 코루틴 | 장애 전파 위험 + 메모리 공유로 인한 오염 |
| 컨테이너 격리 | 과도한 오버헤드, 개발 환경에서 비실용적 |

### 코드 위치

- `coding_agent/core/subagents/process_manager.py:46` — SubAgentProcessManager
- `coding_agent/core/subagents/process_manager.py:62` — `spawn()` (프로세스 생성)
- `coding_agent/core/subagents/process_manager.py:454` — `_kill_process()` (SIGTERM → SIGKILL)
- `coding_agent/core/subagent_context.py:37` — SubagentContextFilter (컨텍스트 격리)
- `coding_agent/core/subagents/schemas.py` — SubAgent 상태/메타데이터 스키마

---

## ADR-003: CoALA 6종 메모리

### 상황

요구사항은 최소 3계층 메모리(`user/profile`, `project/context`, `domain/knowledge`)를 요구합니다.
코딩 에이전트 특성상 더 세분화할지 결정이 필요했습니다.

### 결정

**CoALA(Cognitive Architectures for Language Agents) 패턴으로 6종 메모리를 구현합니다.**

| 메모리 타입 | 용도 | 요구사항 매핑 |
|------------|------|-------------|
| Working | 현재 대화 컨텍스트 (messages) | — |
| Episodic | 과거 작업 이력/결과 참조 | project/context |
| Semantic | 프로젝트 규칙, 기술스택, 커밋 규칙 | project/context |
| Procedural | 코딩 스킬/패턴 누적 (Voyager 패턴) | domain/knowledge |
| User Profile | 사용자 선호, 습관, 피드백 | user/profile |
| Domain Knowledge | 비즈니스 규칙, API 계약, 도메인 용어 | domain/knowledge |

### 이유

1. **Procedural Memory**: 코딩 에이전트는 반복 작업에서 스킬을 누적해야 함. Voyager 패턴의 `accumulate_skill()`로 성공한 코드 패턴을 저장하고 재활용
2. **Episodic Memory**: 과거 작업 결과를 참조하여 유사 작업에서 실수를 반복하지 않음
3. **2-Stage Search**: CoALA의 Tag 필터링 → BM25 랭킹 검색으로 관련 메모리를 효율적으로 조회
4. **JSONL 영속성**: 각 메모리 타입별 `.ai/memory/*.jsonl` 파일로 세션 간 영속성 보장

### 대안 검토

| 대안 | 기각 이유 |
|------|----------|
| 3계층만 구현 | Procedural/Episodic이 코딩 에이전트에서 필수적 |
| 벡터 DB 기반 | 로컬 실행 환경에서 외부 의존성 부담, BM25로 충분 |
| LLM 체크포인터만 사용 | 장기 메모리가 아닌 세션 상태일 뿐 |

### 코드 위치

- `coding_agent/core/memory/schemas.py:24` — MemoryType Enum (6종)
- `coding_agent/core/memory/store.py:49` — MemoryStore 클래스
- `coding_agent/core/memory/store.py:189` — `accumulate_skill()` (Procedural)
- `coding_agent/core/memory/store.py:252` — `accumulate_domain_knowledge()`
- `coding_agent/core/memory/store.py:295` — `accumulate_user_profile()`
- `coding_agent/core/memory/search.py:114` — TwoStageSearch (CoALA 패턴)
- `coding_agent/core/memory/state.py:19` — MemoryAwareState (그래프 상태 통합)
- `coding_agent/core/memory/semantic_loader.py:32` — SemanticMemoryLoader

---

## ADR-004: Puppeteer 선택 알고리즘

### 상황

Orchestrator가 작업을 위임할 SubAgent를 선택할 때, 고정 매핑이 아닌 동적 기준이 필요했습니다.
품질과 비용의 균형, 실행 이력 기반 학습이 요구되었습니다.

### 결정

**R = quality - λ·cost 점수 기반 Puppeteer 선택 알고리즘을 구현합니다.**

```
R = r(quality) - λ · C(cost)
quality = 0.7 × task_success_rate + 0.3 × overall_success_rate
```

- `SubAgentRegistry`가 에이전트별 실행 이력(성공률, 비용)을 추적
- 작업 유형과 에이전트 역량을 매칭하여 후보 목록 생성
- R 점수가 가장 높은 에이전트를 선택
- 실패 시 다른 역할/모델의 SubAgent로 대체 가능

### 이유

1. **비용 최적화**: λ 파라미터로 품질-비용 트레이드오프를 조절
2. **이력 기반 학습**: 실행 결과가 누적되어 점수가 자동으로 보정됨
3. **동적 선택**: 고정된 2개 역할이 아닌, 작업 성격에 따른 최적 에이전트 선택
4. **실패 대응**: 실패한 에이전트 대신 다른 후보를 자동으로 시도

### 코드 위치

- `coding_agent/core/subagents/registry.py:41` — SubAgentRegistry 클래스
- `coding_agent/core/subagents/registry.py:74` — `select()` (Puppeteer 알고리즘)
- `coding_agent/core/subagents/registry.py:315` — `_compute_quality()` (품질 점수 산출)
- `coding_agent/core/subagents/registry.py:101` — `record_usage()` (이력 기록)
- `coding_agent/core/subagents/registry.py:148` — `create_instance()` (동적 인스턴스 생성)
- `coding_agent/core/subagents/registry.py:178` — `transition_state()` (상태 전이)

---

## ADR-005: 양파 패턴 미들웨어 체인

### 상황

복원력(retry/fallback), 메모리 주입/축적, 메시지 윈도우 관리 등의 횡단 관심사를 처리해야 했습니다.
이를 각 에이전트에 분산하면 코드 중복과 누락이 발생합니다.

### 결정

**AgentMiddleware의 `wrap_model_call()` 기반 양파(Onion) 패턴 미들웨어 체인을 구현합니다.**

```
Request → [Resilience.before] → [Memory.before] → [Window.before]
                                                        ↓
                                                    LLM Call
                                                        ↓
Response ← [Resilience.after] ← [Memory.after] ← [Window.after]
```

- `MiddlewareChain`이 미들웨어를 역순으로 감싸 양파 구조 형성
- 각 미들웨어는 `before`(요청 변환) → `inner` → `after`(응답 처리) 구조

### 이유

1. **Safety Envelope**: 복원력, 메모리, 관측성을 프레임워크 수준에서 보장
2. **관심사 분리**: 에이전트 코드는 비즈니스 로직에만 집중
3. **조합 가능성**: 미들웨어를 자유롭게 추가/제거/순서 변경 가능
4. **테스트 용이성**: 각 미들웨어를 독립적으로 단위 테스트 가능

### 현재 미들웨어 스택

| 순서 | 미들웨어 | 역할 |
|------|---------|------|
| 1 | ResilienceMiddleware | retry + fallback + abort 체크포인트 |
| 2 | MemoryMiddleware | 요청 전 메모리 검색 주입 + 응답 후 자동 지식 축적 |
| 3 | MessageWindowMiddleware | 슬라이딩 윈도우로 메시지 폭발 방지 |

### 코드 위치

- `coding_agent/core/middleware/base.py:76` — AgentMiddleware 추상 베이스 클래스
- `coding_agent/core/middleware/base.py:94` — `wrap_model_call()` (before/after 패턴)
- `coding_agent/core/middleware/chain.py:18` — MiddlewareChain (양파 패턴 구현)
- `coding_agent/core/middleware/chain.py:43` — `invoke()` (역순 감싸기)
- `coding_agent/core/middleware/memory.py:83` — MemoryMiddleware
- `coding_agent/core/middleware/resilience.py:47` — ResilienceMiddleware
- `coding_agent/core/middleware/message_window.py:23` — MessageWindowMiddleware

---

## ADR-006: 4-Tier 모델 체계 + SLM 활용

### 상황

다양한 작업(계획, 코드 생성, 검증, 파싱)에 단일 모델을 사용하면 비용이 과도하거나 품질이 부족합니다.
작업별 적정 성능의 모델을 자동으로 선택해야 했습니다.

### 결정

**REASONING / STRONG / DEFAULT / FAST 4티어 체계를 구현하고, FAST 티어에 SLM을 활용합니다.**

| 티어 | 용도 | DashScope 모델 | 비용 (1M tok) |
|------|------|---------------|--------------|
| REASONING | 계획/아키텍처 설계 | qwen-max | $1.2 / $6.0 |
| STRONG | 코드 생성 | qwen-coder-plus | $0.46 / $1.38 |
| DEFAULT | 검증/분석 | qwen-plus | $0.4 / $1.2 |
| FAST | 파싱/분류/메모리 축적 | qwen-turbo | $0.04 / $0.08 |

### 이유

1. **비용 최적화**: 파싱/분류 같은 간단한 작업에 FAST(SLM)을 사용하여 비용을 75배 절감
2. **작업별 적정 성능**: 코드 생성에는 STRONG, 아키텍처 설계에는 REASONING 티어 사용
3. **Fallback 체인**: 상위 티어 실패 시 하위 티어로 자동 전환 (`build_fallback_chain()`)
4. **환경 변수 오버라이드**: `REASONING_MODEL`, `STRONG_MODEL` 등으로 모델 교체 가능

### SLM 활용 지점

| 작업 | 티어 | 이유 |
|------|------|------|
| 요청 분류 (classify) | FAST | JSON 구조화 출력만 필요 |
| 메모리 지식 추출 | FAST | 키워드/엔티티 추출 수준 |
| 도구 호출 판단 | FAST | 도구 선택은 패턴 매칭 수준 |
| 복잡도 판별 | FAST | 간단한 분류 작업 |

### 코드 위치

- `coding_agent/core/model_tiers.py:33` — ModelTier Enum (4티어)
- `coding_agent/core/model_tiers.py:287` — `build_default_tiers()` (환경 변수 오버라이드)
- `coding_agent/core/model_tiers.py:383` — `create_chat_model()` (LiteLLM 통합)
- `coding_agent/core/model_tiers.py:495` — `recommend_tier_for_purpose()` (목적별 추천)
- `coding_agent/core/model_tiers.py:661` — `build_fallback_chain()` (Fallback 체인)
- `coding_agent/core/model_tiers.py:556` — `estimate_cost()` (비용 추정)
