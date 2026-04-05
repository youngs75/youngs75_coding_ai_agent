# 7개 논문 구현 매핑

본 프로젝트는 7개 학술 논문의 핵심 개념을 선별하여 실제 코드로 구현하였다.
이 문서는 각 논문이 **어떻게 구현되었고, 어디서 동작하는지**를 코드 레벨로 추적한다.

---

## 1. Agent-as-a-Judge — `parse → execute → verify` 3단계

CodingAssistant의 그래프가 **생성자(Generator)와 판정자(Judge)를 분리**한다.

```
parse_request → execute_code → generate_final(STRONG) → verify_result(DEFAULT) → apply_code
                                                              ↓ (실패 시)
                                                         execute_code로 재시도
```

- `verify_result` 노드가 Judge 역할 — 정확성, 안전성, 스타일, 완전성 등 7개 항목 검사
- 검증자는 생성자가 모르는 **특권 정보**(시크릿 패턴, 확장자 화이트리스트)를 갖고 검사
- 실패 시 `_should_retry`로 자동 재시도 (최대 3회)

**구현 위치**: `agents/coding_assistant/agent.py` 250~784행

---

## 2. RubricRewards — Generator/Verifier 모델 분리

같은 모델을 쓰면 자기 오류를 못 잡기 때문에, 생성과 검증에 **다른 티어 모델**을 사용한다.

| 단계 | Purpose | 티어 | 모델 (DashScope) |
|------|---------|------|-----------------|
| 분류/파싱 | `parsing` | FAST | qwen-turbo |
| 도구 호출 판단 | `tool_planning` | DEFAULT | qwen-plus |
| **코드 생성** | `generation` | **STRONG** | qwen-coder-plus |
| **코드 검증** | `verification` | **DEFAULT** | qwen-plus |

```python
# agents/coding_assistant/config.py — purpose → tier 매핑
purpose_tiers = {
    "generation": ModelTier.STRONG,
    "verification": ModelTier.DEFAULT,  # 생성자와 다른 모델
    "tool_planning": ModelTier.DEFAULT,
    "parsing": ModelTier.FAST,
}
```

**구현 위치**: `core/model_tiers.py`, `agents/coding_assistant/config.py` 75~114행

---

## 3. CoALA + Voyager — 4종 메모리 시스템

| 메모리 | 유형 | 저장 시점 | 사용 시점 |
|--------|------|----------|----------|
| **Working** | 현재 대화 `messages` | 매 턴 | 항상 |
| **Semantic** | 프로젝트 규칙 (AGENTS.md 등) | 시작 시 자동 로드 | 시스템 프롬프트 주입 |
| **Episodic** | 실행 결과 이력 | verify 후 저장 | 다음 턴 컨텍스트 |
| **Procedural** | 성공한 코드 패턴 | verify 통과 시 | 유사 태스크 참조 |

### Voyager 패턴: 스킬 누적 + 중복 방지

verify 통과한 코드를 Procedural Memory에 누적 저장한다.
이때 **Jaccard 유사도 >= 0.7이면 중복으로 판단하여 저장을 거부**한다(novelty filter).

```python
# core/memory/store.py — Voyager식 스킬 누적
def accumulate_skill(self, code, description, *, novelty_threshold=0.7):
    # 기존 Procedural 항목과 토큰 유사도 비교
    # 중복이면 reject, 신규면 저장
```

### 시스템 프롬프트 주입 순서

```
1. Semantic Memory  → 프로젝트 규칙/컨벤션
2. Skill Context    → 활성 스킬 L2 본문
3. Episodic Memory  → 최근 실행 결과 (세션 스코프, 최대 5개)
4. Procedural Memory → 유사 태스크의 성공 코드 패턴
```

**구현 위치**: `core/memory/store.py` 123~207행, `core/memory/schemas.py`

---

## 4. GAM (Grounded Action Model) — JIT 원본 참조 (MCP)

LLM이 **환각 없이 실제 데이터를 참조**하도록 MCP 도구를 통해 실시간(JIT) 원본을 가져온다.

```
LLM "파일 읽어야 해" → tool_call: read_file("app.py")
                          ↓
                     MCP 서버 실행
                          ↓
                     실제 파일 내용 반환
                          ↓
                     ToolMessage로 LLM에 피드백 → 다음 추론
```

### MCPToolLoader 동작 방식

1. MCP 서버별 health check (2회 재시도)
2. 사용 가능한 서버에서 도구 목록 로드 + 캐싱
3. ReAct 루프에서 도구 실행 결과가 즉시 LLM 컨텍스트에 주입
4. 도구 결과는 `ToolMessage`로 래핑되어 다음 추론의 입력이 됨

**구현 위치**: `core/mcp_loader.py`, `agents/coding_assistant/agent.py` 540~690행

---

## 5. Puppeteer — 동적 오케스트레이션 `R = r(q) - lambda * C`

Orchestrator가 어떤 서브에이전트에 위임할지 **점수 기반으로 동적 선택**한다.

### 점수 공식

```
R = 0.7 * task_specific_success_rate + 0.3 * overall_rate - lambda * cost_weight
```

| 항목 | 설명 |
|------|------|
| `r(quality)` | 태스크 유형별 성공률 (70%) + 전체 성공률 (30%) |
| `C(cost)` | 에이전트 비용 가중치 (레이턴시 + 토큰 비용 프록시) |
| `lambda` | cost sensitivity 파라미터 (0~1, 높을수록 비용 민감) |

### 동적 선택 흐름

1. 후보 에이전트를 `capabilities` 기반으로 필터링
2. 각 후보에 대해 `R = quality - lambda * cost` 계산
3. 최고 점수 에이전트 선택 + 상세 이유 문자열 반환
4. 실행 후 `record_usage(success/fail)`로 통계 갱신 → 다음 선택에 반영

**구현 위치**: `core/subagents/registry.py` 52~77행, `core/subagents/schemas.py`

---

## 6. AutoHarness — Safety Envelope

생성된 코드가 실행되기 전에 **5가지 안전 검사**를 통과해야 한다.

### ActionValidator 검사 항목

| 검사 | 차단 대상 | 결과 |
|------|----------|------|
| **시크릿 노출** | `API_KEY="sk-..."`, `PASSWORD="..."` | BLOCK |
| **위험 명령** | `rm -rf /`, `DROP TABLE`, `FORMAT C:` | BLOCK |
| **대량 삭제** | 100개 이상 삭제 지시자 | WARN |
| **확장자 제한** | `.py, .js, .ts, .json, .yaml, .md` 외 | BLOCK |
| **경로 탈출** | workspace 외부 접근 | BLOCK |

### ToolPermissionManager 3단계 권한 계층

| 레벨 | 소스 | 우선순위 |
|------|------|---------|
| Level 1 | `DEFAULT_PERMISSIONS` (하드코딩) | 최저 |
| Level 2 | `.agent/permissions.yaml` 파일 | 중간 |
| Level 3 | 환경변수 `TOOL_PERM_{TOOL_NAME}` | 최고 |

도구별 기본 권한:
- **읽기 도구** (read_file, search_code, list_directory) → ALLOW
- **쓰기 도구** (write_file, str_replace) → ALLOW (workspace 내)
- **실행/삭제 도구** (execute_python, bash, delete_file) → ASK (사용자 확인 필요)

**구현 위치**: `core/action_validator.py`, `core/tool_permissions.py`

---

## 7. Claude Code 패턴 — 스킬 자동활성화 + 3-Level Progressive Loading

### 3단계 점진적 로딩

컨텍스트 비용을 최소화하기 위해 스킬을 단계적으로 로드한다.

| 레벨 | 내용 | 로딩 시점 | 컨텍스트 비용 |
|------|------|----------|-------------|
| **L1** | 이름, 설명, 태그 | 항상 | 최소 |
| **L2** | 프롬프트 본문 | 활성화 시 (on-demand) | 중간 |
| **L3** | 외부 참조 파일 | 실행 시 (lazy) | 최대 |

### task_type 기반 자동활성화

`parse_request`에서 추출한 `task_type`에 따라 관련 스킬을 자동으로 L2까지 로드한다.

```python
# core/skills/registry.py — task_type → 태그 매핑
TASK_TYPE_TAGS = {
    "generate": ["quality"],
    "fix": ["fix", "debug"],
    "refactor": ["refactor"],
    "explain": ["explain"],
    "analyze": ["review", "security"],
}
# → search_by_tags(["fix", "debug"]) → 매칭 스킬 자동 activate → L2 본문 로드
```

### 활성화 흐름

```
parse_request에서 task_type 추출 (예: "fix")
  ↓
TASK_TYPE_TAGS["fix"] → ["fix", "debug"] 태그 조회
  ↓
search_by_tags(["fix", "debug"]) → 매칭 스킬 검색
  ↓
activate(skill_name) → L2 본문 로드
  ↓
skill_context에 주입 → 시스템 프롬프트에 포함
```

**구현 위치**: `core/skills/registry.py` 108~127행, `core/skills/schemas.py`, `core/skills/loader.py`

---

## 전체 데이터 흐름 (7개 논문 통합)

7개 논문이 단독으로 동작하는 것이 아니라 **하나의 파이프라인에서 유기적으로 연결**된다.

```
사용자 입력
  ↓
Parse (FAST) ─────────────── [7] Claude Code: 스킬 자동활성화
  ↓
Memory Retrieval ─────────── [3] CoALA+Voyager: Episodic + Procedural 검색
  ↓
Execute ReAct (DEFAULT) ──── [4] GAM: MCP 도구로 JIT 원본 참조
  ↓
Generate Final (STRONG) ──── [2] RubricRewards: 생성 전용 모델
  ↓
Verify (DEFAULT) ─────────── [1] Agent-as-a-Judge: 판정 + [6] AutoHarness: Safety Envelope
  ↓ (통과 시)
Save Memory ──────────────── [3] Voyager: 스킬 누적 (Jaccard novelty filter)
  ↓
Apply Code ───────────────── [6] AutoHarness: 경로 탈출/금지 경로 차단
  ↓
Orchestrator 위임 결정 ───── [5] Puppeteer: R = r(q) - lambda * C 동적 선택
```

## 구현 상태 요약

| # | 논문 | 핵심 개념 | 구현 상태 | 주요 파일 |
|---|------|----------|----------|----------|
| 1 | Agent-as-a-Judge | 3단계 파이프라인 + Judge 검증 | **완전 구현** | `agents/coding_assistant/agent.py` |
| 2 | RubricRewards | Generator/Verifier 모델 분리 | **완전 구현** | `core/model_tiers.py`, `config.py` |
| 3 | CoALA + Voyager | 4종 메모리 + 스킬 누적 | **완전 구현** | `core/memory/` |
| 4 | GAM | JIT 원본 참조 (MCP) | **완전 구현** | `core/mcp_loader.py` |
| 5 | Puppeteer | 동적 오케스트레이션 R=r-lambda*C | **완전 구현** | `core/subagents/registry.py` |
| 6 | AutoHarness | Safety Envelope + 권한 계층 | **완전 구현** | `core/action_validator.py`, `core/tool_permissions.py` |
| 7 | Claude Code | 3-Level 스킬 + 자동활성화 | **완전 구현** | `core/skills/` |
