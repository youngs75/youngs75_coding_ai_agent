# Coding Agent 설계 가이드 — 논문 인사이트 기반

**작성일**: 2026-04-02  
**목적**: youngs75_a2a 프레임워크에 Coding Agent를 추가할 때 참조할 설계 원칙과 구현 가이드  
**근거**: CoALA, AutoHarness, Puppeteer, Adaptation, Agent-as-a-Judge, RubricRewards, GAM 7편 논문 분석 결과

---

## 1. 최소 구조로 시작 (Agent-as-a-Judge ablation 교훈)

> **핵심**: 8개 모듈 중 Memory/Planning을 제거했을 때 오히려 성능이 향상됨.  
> Memory는 이전 판단 오류가 연쇄 전파, Planning은 불안정한 계획이 노이즈 유발.

### 시작 구조: 3노드

```
parse_request → execute_code → verify_result
```

| 노드 | 역할 | Agent-as-a-Judge 대응 모듈 |
|------|------|---|
| `parse_request` | 사용자 요청 분석, 변경 범위 식별 | Graph + Locate |
| `execute_code` | 코드 생성/수정 실행 | Read + Retrieve |
| `verify_result` | 결과 검증 (테스트, 린트) | Ask |

### 추가 금지 (ablation 결과 해로움)

- ~~`plan_changes`~~ — 초기에 넣지 않음. 성능 측정 후 ablation으로 추가 여부 판단
- ~~`memory`~~ — 이전 세션 기억하지 않음. 매 세션 독립 실행이 기본

### 추가 기준

노드를 추가할 때는 반드시 **with/without 비교 실험**을 수행한다.  
추가했을 때 성공률이 올라가지 않으면 넣지 않는다.

---

## 2. Generator-Verifier 분리 (RubricRewards 교훈)

> **핵심**: 생성자와 검증자를 분리하고, 검증자에게만 특권 정보를 부여해야 과적합을 탐지할 수 있다.

### Config 설계

```python
class CodingConfig(BaseAgentConfig):
    generation_model: str = "gpt-5.4"
    verification_model: str = "gpt-5.4"

    def _resolve_model_name(self, purpose: str) -> str:
        if purpose == "generation":
            return self.generation_model
        if purpose == "verification":
            return self.verification_model
        return self.default_model
```

### 검증자 특권 정보

`verify_result` 노드의 시스템 프롬프트에만 제공하는 정보:
- 과거 이 프로젝트에서 자주 발생한 버그 패턴
- 코드 변경이 영향을 미칠 수 있는 의존성 목록
- 테스트 커버리지 기준치

`execute_code` 노드에는 이 정보를 **주지 않는다**. 코드 생성은 요구사항만 보고, 검증은 더 넓은 맥락에서 판단한다.

### 주의

- 같은 모델이라도 **시스템 프롬프트 분리**가 핵심
- SFT(좋은 코드 복사) < RL(왜 좋은지 이해) — 단순 예제 주입보다 판단 기준을 제공

---

## 3. Safety Envelope (AutoHarness 교훈)

> **핵심**: is_legal_action() 함수가 LLM 제안을 실행 전에 검증. 평균 14.5회 반복만에 100% 합법 행동 달성.  
> **패턴 4**: "심판이 선수를 겸하면 안 된다" — 검증기는 에이전트 바깥에 있어야 한다.

### 배치 위치: core/ 레벨

```
agents/coding_assistant/  ← 여기가 아니라
core/action_validator.py  ← 여기에 배치
```

### 구현 방향

```python
# core/action_validator.py

class ActionValidator:
    """에이전트 출력을 실행 전에 검증하는 Safety Envelope.
    
    개별 에이전트가 아닌 프레임워크 차원에서 강제한다.
    """

    def __init__(self, rules: list[ValidationRule]):
        self.rules = rules

    async def validate(self, action: AgentAction) -> ValidationResult:
        """모든 규칙을 검사하여 통과/거부/수정 판정."""
        ...

    def is_safe(self, action: AgentAction) -> bool:
        """빠른 검증 (동기)."""
        ...
```

### 검증 규칙 예시 (Coding Agent용)

| 규칙 | 설명 |
|------|------|
| 파일 범위 제한 | 허용된 디렉토리 밖의 파일 수정 금지 |
| 삭제 상한 | 한 번에 삭제하는 줄 수 제한 |
| 의존성 변경 감지 | package.json/requirements.txt 변경 시 경고 |
| 시크릿 노출 방지 | API 키, 비밀번호 패턴 감지 |
| 실행 명령 제한 | rm -rf, DROP TABLE 등 위험 명령 차단 |

### 적용 방식

`BaseGraphAgent.build_graph()` 시 검증 노드를 자동 삽입하는 것이 가장 깔끔하다:

```python
# 에이전트 코드 변경 없이 프레임워크가 검증을 강제
def build_graph(self):
    ...
    if self.action_validator:
        # execute 노드 뒤에 자동으로 검증 노드 삽입
```

---

## 4. CoALA Memory 체계 확장 (CoALA + Voyager 교훈)

> **핵심**: 에이전트 메모리를 Working/Episodic/Semantic/Procedural 4종으로 분류.  
> Voyager는 procedural memory에 학습된 스킬을 저장하고 재사용.

### 확장 클래스 (BaseGraphState 수정 불필요)

```python
# core/memory_state.py

class MemoryAwareState(BaseGraphState):
    """CoALA 메모리 체계를 반영한 확장 상태."""
    
    # Working Memory — 현재 대화 (= BaseGraphState.messages)
    # 아래는 추가 메모리 계층
    
    episodic: list[str]     # 실행 결과 이력 (이번 세션)
    semantic: dict           # 도메인 지식/규칙 (프로젝트 컨벤션 등)
    procedural: list[str]   # 학습된 스킬 패턴 (Voyager식 누적)
```

### Coding Agent에서의 활용

| 메모리 타입 | 저장 내용 | 예시 |
|---|---|---|
| Working | 현재 대화, 현재 코드 변경 | `messages` |
| Episodic | 이번 세션의 실행/테스트 결과 | "test_auth.py 3개 통과, 1개 실패" |
| Semantic | 프로젝트 규칙 | "이 프로젝트는 snake_case 사용" |
| Procedural | 검증된 코드 패턴 | "이 프로젝트에서 DB 접근 시 항상 context manager 사용" |

### 주의: Agent-as-a-Judge 교훈과의 균형

- Episodic/Procedural Memory는 **오류 전파 위험**이 있음
- 초기에는 Working + Semantic만 사용
- Episodic/Procedural은 성능 측정 후 도입 여부 판단

---

## 5. JIT 원본 참조 (GAM 교훈)

> **핵심**: "요약만으로는 부족하고, 필요 시 원본에 돌아갈 수 있어야 한다."

### 현재 상태

DeepResearchAgent의 `compress_research` 노드는 이미 `raw_notes`를 보존하고 있고,
`final_report_generation`에서 `notes + raw_notes` 양쪽을 사용한다.

### Coding Agent 적용

코드 분석 → 요약 → 변경 계획 흐름에서:

```
코드베이스 읽기 → 요약(compress) → 변경 실행
                      │
                      └── 요약이 부족하면 원본 코드 재참조 가능해야 함
```

`execute_code` 노드가 요약만 보고 코드를 생성하면 맥락을 놓칠 수 있다.
원본 파일 경로를 상태에 보존하여 필요 시 직접 읽을 수 있는 경로를 열어둔다.

### 검색 전략 (GAM 실험 결과)

> BM25 + Embedding + Page-ID 3종 조합이 최고 성능

Coding Agent에서는:
- **키워드 검색** (grep/ripgrep) — 함수명, 클래스명 정확 매칭
- **의미 검색** (embedding) — "인증 관련 코드" 같은 자연어 검색
- **구조 검색** (AST/파일 경로) — import 관계, 호출 그래프

---

## 6. 동적 오케스트레이션 (Puppeteer 교훈)

> **핵심**: 정적 위임보다 현재 상태 기반 동적 에이전트 선택이 효과적.  
> 진화 후 활성 에이전트 수가 오히려 감소 — 필요한 것만 쓰는 것이 낫다.

### 도입 시점: 에이전트 3개 이상일 때

현재 에이전트: SimpleReAct, DeepResearch, DeepResearchA2A → 아직 불필요  
Coding Agent 추가 후: 4개 → 검토 시작  
특화 에이전트(테스트, 린트, 리뷰) 추가 후: 6개+ → 도입 가치 있음

### 구현 방향

```python
# a2a/orchestrator.py

class AgentOrchestrator:
    """태스크 상태에 따라 최적 에이전트를 동적 선택."""
    
    async def select(self, task_state, available_agents) -> AgentEndpoint:
        # R = r(품질) − λ·C(비용)
        # 품질: 에이전트의 태스크 유형별 성공률
        # 비용: 레이턴시 + 토큰 사용량
        ...
```

### 비용 함수

```
R = r(quality) − λ · C(cost)

quality: 해당 에이전트의 유사 태스크 성공률
cost: API 토큰 비용 + 응답 시간
λ: 비용 민감도 (환경변수로 조정)
```

---

## 7. 졸업 Lifecycle (Adaptation 교훈)

> **핵심**: 검증된 에이전트 설정을 고정하고 "졸업된 도구"로 재배포.

### 적용 시점: 운영 안정화 후

### 졸업 프로세스

```
1. 에이전트가 특정 도메인에서 충분히 검증됨
2. 해당 설정(Config)을 고정
3. A2A 서버로 배포 (run_server() 한 줄)
4. 다른 에이전트가 A2A 프로토콜로 "도구"처럼 호출
```

### 예시

```python
# 금융 리서치에 특화된 DeepResearch가 검증 완료 후:
run_server(
    executor=LGAgentExecutor(graph=finance_research_agent.graph),
    name="finance-researcher",
    description="금융 도메인 전문 리서치 (졸업됨)",
    port=8092,
)

# 다른 에이전트가 A2A로 호출:
# call_supervisor_a2a() → http://localhost:8092
```

---

## 구현 우선순위 체크리스트

Coding Agent 구현 시 아래 순서로 진행:

- [ ] **P1**: 최소 3노드 구조 (parse → execute → verify)
- [ ] **P2**: CodingConfig에 generation/verification 모델 분리
- [ ] **P3**: core/action_validator.py Safety Envelope
- [ ] **P4**: core/memory_state.py MemoryAwareState (Working + Semantic만)
- [ ] **P5**: verify 노드에 JIT 원본 참조 경로
- [ ] **P6**: a2a/orchestrator.py (에이전트 4개 이상 시)
- [ ] **P7**: 졸업 lifecycle 프로세스 (운영 안정화 후)

---

*논문 7편 인사이트 + 학습 노트 5개 패턴 기반 | 2026-04-02 작성*
