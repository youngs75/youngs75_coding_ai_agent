# Claude Code & Codex 소스코드 분석 — AI Coding Agent Harness 개선 참고

**분석일**: 2026-04-04  
**분석 대상**: claude-code-haha (Claude Code 유출 소스), codex (OpenAI Codex CLI)  
**목적**: youngs75_coding_ai_agent 프로젝트의 아키텍처, 기능, 안정성 개선을 위한 참고 자료

---

## 1. 전체 아키텍처 비교

### 1.1 Claude Code (TypeScript/Bun)

```
entrypoints/cli.tsx (부트스트랩)
    ↓
main.ts (CLI 로더)
    ↓
QueryEngine (세션 상태 관리 — 13,000+ 줄)
    ├─ submitMessage() → 1턴 실행
    └─ query() → 비동기 제너레이터 기반 에이전트 루프
        ├─ StreamingToolExecutor (병렬 도구 실행)
        ├─ CanUseToolFn (도구 권한 판정)
        └─ compact (자동 컨텍스트 압축)
```

- **단일 바이너리 CLI** — Feature-gated dead code elimination으로 빌드타임에 불필요 코드 제거
- **AsyncGenerator 기반 루프** — 선언적 상태 그래프가 아닌 명령형 제너레이터 루프로 유연성 극대화
- **다중 엔트리포인트** — CLI / bridge / daemon / remote-control 런타임 분기

### 1.2 Codex (Rust)

```
codex-rs/
├── cli/           → CLI 엔트리포인트
├── core/          → 핵심 실행 엔진
│   ├── agent/     → 에이전트 수명 관리 (spawn, mailbox, registry)
│   ├── tools/     → 도구 핸들러 (shell, apply_patch, MCP, exec)
│   └── codex.rs   → 메인 세션 루프
├── tui/           → ratatui 기반 터미널 UI (app.rs 444KB)
├── exec/          → 프로세스 실행 관리
├── sandboxing/    → 플랫폼별 샌드박싱 (Linux/macOS/Windows)
├── execpolicy/    → Starlark 기반 실행 정책 엔진
├── apply-patch/   → Unified diff 패치 파서 및 적용
└── protocol/      → 타입 정의 및 IPC 프로토콜
```

- **Rust 모놀리식 바이너리** — 성능과 안전성 극대화
- **Mailbox 기반 에이전트 통신** — mpsc + sequence number로 순서 보장
- **플랫폼별 샌드박싱** — Seatbelt(macOS), Landlock/Bubblewrap(Linux)

### 1.3 우리 프로젝트 (Python/LangGraph)

```
youngs75_coding_ai_agent/
├── core/           → BaseGraphAgent, MCP Loader, model_tiers
├── agents/         → coding_assistant, deep_research, simple_react, orchestrator
├── a2a_local/      → A2A 프로토콜 통합
├── mcp_servers/    → code_tools MCP 서버
├── cli/            → prompt-toolkit + rich 기반 대화형 CLI
└── eval_pipeline/  → DeepEval + Langfuse 관측성
```

---

## 2. 에이전트 루프 & 도구 사용 패턴

### 2.1 Claude Code — StreamingToolExecutor

**파일**: `src/services/tools/StreamingToolExecutor.ts`

Claude Code의 도구 실행기는 **병렬성과 순서 보장을 동시에 달성**합니다:

```
도구 큐: [read_file(A), read_file(B), bash(C), read_file(D)]

실행 흐름:
  read_file(A) ──┐
  read_file(B) ──┤── 병렬 실행 (isConcurrencySafe=true)
                 ├── 결과 대기
  bash(C)     ───┤── 독점 실행 (isConcurrencySafe=false)
                 ├── 완료 후
  read_file(D) ──┘── 다음 실행
```

**핵심 메커니즘**:
- 각 도구 정의에 `isConcurrencySafe()` 플래그 존재
- Concurrent-safe 도구끼리만 병렬 실행 가능 (예: 여러 파일 동시 읽기)
- Non-concurrent 도구 (Bash, Agent)는 독점 실행
- 결과는 큐 순서대로 emit (순서 보장)
- Bash 에러 시 `siblingAbortController` 발동 → 형제 프로세스 즉시 중단

**우리와의 차이**: 우리는 MCP 도구를 순차 실행. 여러 파일 읽기도 하나씩 처리.

### 2.2 Claude Code — max_output_tokens 복구

**파일**: `src/query.ts`

```
API 응답 stop_reason 분석:
  ├─ "end_turn"    → 정상 종료
  ├─ "tool_use"    → 도구 결과 대기 후 계속
  ├─ "max_tokens"  → 복구 시도 (최대 3회)
  │     ├─ 1차: reactiveCompact (캐시 재구성)
  │     ├─ 2차: 재시도
  │     └─ 3차: 에러 반환
  └─ "stop_sequence" → custom stop (드물음)
```

**우리와의 차이**: 우리는 max_tokens 발생 시 그대로 종료. 복구 로직 없음.

### 2.3 Codex — Mailbox 기반 에이전트 통신

**파일**: `codex-rs/core/src/agent/mailbox.rs`

```rust
pub struct Mailbox {
    tx: mpsc::UnboundedSender<InterAgentCommunication>,
    next_seq: AtomicU64,        // 메시지 순번
    seq_tx: watch::Sender<u64>, // 변경 감지 (subscriber 알림)
}
```

- 각 에이전트 spawn 시 전용 mailbox 생성 (thread_id 단위)
- sequence number로 메시지 순서 보장
- `watch::Sender<u64>`로 subscriber들에게 새 메시지 도착 알림

### 2.4 Codex — 에이전트 포크 모드

**파일**: `codex-rs/core/src/agent/mod.rs`

```rust
pub enum SpawnAgentForkMode {
    FullHistory,         // 부모 전체 히스토리 상속
    LastNTurns(usize),   // 마지막 N개 턴만 상속
}

const DEFAULT_AGENT_MAX_DEPTH: i32 = 1;
const DEFAULT_AGENT_MAX_THREADS: Option<usize> = Some(6);
```

- 자식 에이전트에 전체 히스토리 전달 vs 최근 N턴만 전달 선택 가능
- 깊이 제한(max_depth=1)과 스레드 제한(max_threads=6)으로 무한 확장 방지

**우리와의 차이**: 우리는 서브에이전트에 항상 전체 히스토리 전달. 토큰 낭비.

---

## 3. MCP 통합

### 3.1 Claude Code — 고급 MCP 클라이언트

**파일**: `src/services/mcp/client.ts` (5,100+ 줄)

**연결 방식 (우선순위 순)**:
1. stdio (스탠드얼론 서버 프로세스)
2. SSE (HTTP Server-Sent Events)
3. streamableHttp (스트리밍 가능한 HTTP)
4. WebSocket
5. in-process (테스트용)

**고급 기능**:
- **OAuth 인증 지원** (XAA protocol)
- **동적 리소스 풀링**: `ListResources` → `ReadResource` 도구로 MCP 리소스를 동적 탐색
- **도구별 프로그레스 콜백**: 실행 중 진행률 UI 업데이트
- **이미지 자동 리사이징**: MCP에서 반환된 이미지를 컨텍스트에 맞게 조정
- **타임아웃 처리**: 기본 30초, 도구별 커스텀 가능
- **결과 크기 자동 절단**: 컨텍스트 오버플로우 방지

### 3.2 Codex — MCP 서버 자동 발견

**파일**: `codex-rs/codex-mcp/`

- 설정 파일에서 MCP 서버 자동 발견
- stdio / SSE / WebSocket 지원
- 발견 후 권한 확인 과정 통합

### 3.3 우리 프로젝트와의 차이

| 항목 | Claude Code | Codex | 우리 |
|------|------------|-------|-----|
| MCP 라이브러리 | 자체 SDK 클라이언트 | Rust MCP client | langchain-mcp-adapters |
| 서버 발견 | 런타임 config + dynamic | 자동 발견 | 정적 서버 레지스트리 |
| 스트리밍 | SSE, WebSocket | SSE, WebSocket | 없음 |
| 프로그레스 콜백 | 있음 (도구 실행 중) | 있음 | 없음 |
| 리소스 풀링 | 있음 (동적 탐색) | 있음 | 없음 |
| OAuth | 지원 | 미지원 | 미지원 |

---

## 4. 컨텍스트 관리 (Context Window Management)

### 4.1 Claude Code — 자동 컴팩션 시스템

**파일**: `src/services/compact/compact.ts`, `autoCompact.ts`

```
컨텍스트 사용량 모니터링:
  ├─ calculateTokenWarningState()
  │     ├─ 80% 이상 → 자동 컴팩션 트리거
  │     └─ 90% 이상 → 긴급 컴팩션
  │
  ├─ 컴팩션 전략:
  │     ├─ reactive compact: API 실패(max_tokens) 후 캐시 재구성
  │     ├─ snip compact: 히스토리 일부만 요약 (GAM 패턴)
  │     └─ history snip: 상위 N턴 유지 + 나머지 요약
  │
  └─ microCompact: API 응답 힌트 기반 미세 조정
```

**Prompt Cache 최적화**:
- 정적 시스템 프롬프트 캐싱 (1시간 유효)
- 도구 정의 캐싱
- 메모리 파일 캐싱
- cache_creation_tokens vs cache_read_tokens 분리 추적
- 비용 50% 감소 + 레이턴시 개선

### 4.2 Codex — 선택적 히스토리 전파

**파일**: `codex-rs/core/src/thread_rollout_truncation.rs`

```
히스토리 필터링 규칙:
  ├─ system/developer 메시지 → 항상 포함
  ├─ user 메시지 → 항상 포함
  └─ assistant 메시지 → FinalAnswer phase만 포함
      (중간 reasoning/tool_call은 제외)
```

- `truncate_rollout_to_last_n_fork_turns()`: 자식 에이전트에 전달할 히스토리를 N턴으로 제한
- 시스템/개발자 메시지는 truncation에서 제외

### 4.3 우리와의 차이

우리는 **컨텍스트 관리가 전혀 없음**. 긴 대화 시 토큰 초과로 실패.

---

## 5. 스킬(Slash 커맨드) & 플러그인 시스템

### 5.1 Claude Code — Markdown 기반 스킬

**파일**: `src/skills/loadSkillsDir.ts` (800+ 줄)

**스킬 소스 (우선순위 순)**:
1. 조직 정책 → `~/.claude/.skills/`
2. 사용자 설정 → `~/.claude/skills/`
3. 프로젝트 설정 → `.claude/skills/`
4. 플러그인
5. 빌트인
6. MCP 프롬프트에서 자동 생성

**스킬 파일 형식 (Markdown + Frontmatter)**:
```markdown
---
name: "commit"
description: "Stage and commit changes"
whenToUse: "After making code changes"
allowedTools: ["bash", "file_read"]
effort: "low"
---

# Commit Skill

1. Run `git status` to see changes
2. Stage relevant files
3. Create a descriptive commit message
...
```

**특징**:
- 심볼릭 링크 처리 (중복 방지)
- 런타임 .claude/skills/ 디렉토리 감시 (동적 로드)
- 토큰 추정 (frontmatter 기반)
- 시스템 프롬프트에는 요약만 포함 (전체는 호출 시 로드)

### 5.2 Codex — 패키지형 스킬

**파일**: `codex-rs/skills/`

- 스킬 = 패키지된 에이전트 설정
- `.codex/skills/` 디렉토리에서 로드
- 동적 로드 + 버전 관리

### 5.3 우리와의 차이

우리는 `core/skills/` 모듈이 있으나 기본 구현만 존재. Claude Code 수준의 동적 발견/로드는 없음.

---

## 6. 권한 모델 & 안전성

### 6.1 Claude Code — 계층화된 권한 시스템

**파일**: `src/hooks/useCanUseTool.tsx`

```
설정 레벨:
  └─ Tool Permission Rules (alwaysAllow, alwaysDeny, alwaysAsk)
       ├─ Policy Settings (조직 — 최우선)
       ├─ User Settings (~/.claude/)
       └─ Project Settings (.claude/)

런타임 판정:
  ├─ hasPermissionsToUseTool() → config/rules 기반 빠른 판정
  ├─ "ask" 결과 시:
  │     ├─ Coordinator permission (자동 분류기 검사 대기)
  │     ├─ Swarm worker permission (백그라운드 에이전트용)
  │     └─ Interactive permission (사용자 프롬프트)
  ├─ Speculative classifier check → 비차단 분류 (2초 타임아웃)
  └─ 거부 기록 → permissionDenials 배열에 수집 → SDK 보고
```

**Auto Mode**: ML 기반 `TRANSCRIPT_CLASSIFIER`로 Bash 명령 안전성 자동 판정

### 6.2 Codex — Starlark 기반 실행 정책 엔진

**파일**: `codex-rs/execpolicy/`

```starlark
# 정책 규칙 예시
prefix_rule(
    pattern = ["git", ["status", "log", "diff"]],
    decision = "allow",
    justification = "읽기 전용 git 명령은 안전",
    match = [["git", "status"], "git log --oneline"],
    not_match = [["git", "push"]],
)

host_executable(
    name = "git",
    paths = ["/opt/homebrew/bin/git", "/usr/bin/git"],
)
```

**평가 결과**:
```json
{
  "matchedRules": [{
    "prefixRuleMatch": {
      "matchedPrefix": ["git", "status"],
      "decision": "allow",
      "resolvedProgram": "/usr/bin/git"
    }
  }],
  "decision": "allow"
}
```

### 6.3 Codex — 다층 샌드박싱

**파일**: `codex-rs/sandboxing/src/manager.rs`

```
SandboxType:
  ├─ MacosSeatbelt    → sandbox-exec (Apple)
  ├─ LinuxSeccomp     → Landlock LSM + Bubblewrap
  ├─ WindowsRestrictedToken
  └─ None

정책 분리:
  ├─ FileSystemSandboxPolicy (읽기/쓰기 경로 제한)
  └─ NetworkSandboxPolicy (네트워크 접근 제한)
```

### 6.4 우리와의 차이

우리는 **권한 시스템이 없음**. MCP code_tools에 workspace 경로 제한(`_safe_path`)만 존재.

---

## 7. 메모리 시스템

### 7.1 Claude Code — CLAUDE.md + 팀 메모리

**파일**: `src/memdir/memdir.ts` (650+ 줄)

```
메모리 검색 경로:
  ├─ 프로젝트 .claude/CLAUDE.md
  ├─ 상위 디렉토리 .claude/CLAUDE.md (최대 홈까지 재귀)
  └─ 팀 메모리 .claude/teams/{team}/...

메모리 첨부 방식:
  ├─ nested_memory (다른 CLAUDE.md 링크)
  ├─ 마지막 변경 시각 추적
  ├─ 자동 프리페치 (관련성 기반)
  └─ 가중치 기반 선택 (관련성 점수)
```

**특징**:
- 팀 메모리 (세션 간 공유)
- 동적 로드 (쿼리 시 필요한 것만)
- 토큰 예산 내에서 관련 메모리만 선택

### 7.2 우리와의 차이

우리는 `core/memory/` 모듈에 Episodic/Semantic/Procedural 메모리가 있으나, 프로젝트 메타정보 자동 주입이나 팀 메모리 공유는 없음.

---

## 8. 파일 작업 & Diff 기반 패치

### 8.1 Codex — apply_patch 모듈

**파일**: `codex-rs/apply-patch/src/lib.rs`

```rust
pub enum ApplyPatchFileChange {
    Add { content: String },          // 새 파일 생성
    Delete { content: String },        // 파일 삭제
    Update {
        unified_diff: String,          // unified diff 형식
        move_path: Option<PathBuf>,    // 파일 이동 시 대상 경로
        new_content: String,           // 최종 결과 내용
    },
}
```

**Safety Assessment**:
```
패치 안전성 검사:
  ├─ SafetyCheck::AutoApprove → 자동 승인 (안전한 변경)
  ├─ SafetyCheck::AskUser     → 사용자 확인 필요
  └─ SafetyCheck::Reject      → 거부 (위험한 변경)
```

- Unified diff 형식으로 변경 검증 (similarity crate 사용)
- 상대 경로 → 절대 경로 정규화
- 파일 이동(rename) 지원

### 8.2 우리와의 차이

우리의 MCP code_tools는 `write_file(path, content)`로 **전체 파일 덮어쓰기만 지원**. diff 기반 수정, 안전성 검증 없음.

---

## 9. 스트리밍 & UX

### 9.1 Claude Code — Ink (React 터미널 렌더링)

**파일**: `src/ink/`, `src/components/`

- **Ink 6.8.0**: React Reconciler 기반 터미널 렌더링
- **Yoga**: 터미널 레이아웃 엔진
- **가상 스크롤**: `useVirtualScroll.ts` (45KB) — 긴 출력 효율적 렌더링
- **음성 통합**: `useVoice.ts` (99KB)
- **토큰 스트리밍**: `text_delta` 이벤트별 실시간 출력, `input_json_delta`로 도구 인자 스트리밍
- **조건부 import**: lazy loading으로 시작 시간 최적화
- **useMinDisplayTime**: 너무 빠른 UI 업데이트 방지

### 9.2 Codex — ratatui 기반 TUI

**파일**: `codex-rs/tui/`

- `app.rs` (444KB): 상태 머신 & 이벤트 루프
- `chatwidget.rs` (445KB): 대화 렌더링 엔진
- `markdown_render.rs`: Markdown → ratatui 위젯 변환
- `diff_render.rs` (94KB): Git diff 시각화
- 스트림 파서: UTF-8 검증, 태그 파싱, Markdown 스트리밍, LLM 응답 분할

### 9.3 우리와의 차이

| 항목 | Claude Code | Codex | 우리 |
|------|------------|-------|-----|
| 렌더링 | Ink (React) | ratatui | prompt-toolkit + rich |
| 토큰 스트리밍 | text_delta 실시간 | 스트림 파서 | astream_events 기반 |
| Markdown 렌더링 | 있음 | 있음 (diff 포함) | 기본 코드 하이라이팅만 |
| 가상 스크롤 | 있음 | 있음 | 없음 |
| 도구 진행률 | progress 콜백 | 있음 | 없음 |

---

## 10. 훅 시스템 (Extensibility)

### 10.1 Claude Code — 사용자 정의 훅

**파일**: `src/utils/hooks/`, `src/utils/hooks.ts`

```
훅 소스:
  ├─ settings.json hooks (사용자 정의)
  ├─ plugins (플러그인 등록)
  └─ 내장 (coordinator, auto-mode 등)

훅 타입:
  ├─ pre_tool_call    → 도구 실행 전
  ├─ post_tool_call   → 도구 실행 후
  ├─ pre_compact      → 컨텍스트 컴팩션 전
  ├─ post_compact     → 컨텍스트 컴팩션 후
  ├─ session_start    → 세션 시작
  └─ user_prompt_submit → 사용자 입력 제출 시
```

- 쉘 명령어 실행 (`shell` 타입) 가능
- 결과를 시스템 메시지로 주입
- 블로킹 (훅 완료까지 대기)

### 10.2 우리와의 차이

우리는 훅 시스템 없음. LangGraph의 callback 패턴을 활용 가능하나 사용자 정의 훅은 미구현.

---

## 11. 프롬프트 엔지니어링

### 11.1 Claude Code — 정적/동적 분리 시스템 프롬프트

**파일**: `src/constants/prompts.ts` (1,000+ 줄)

```
시스템 프롬프트 구성:
  ├─ Static Prefix (캐시 가능, 모든 사용자 동일)
  │     ├─ SYSTEM_PROMPT_DYNAMIC_BOUNDARY 마커
  │     ├─ 도구 목록 + 사용 가이드
  │     ├─ 주요 지침 (안전성, 코드 품질 등)
  │     └─ 환경 정보 (OS, shell, git 상태)
  │
  └─ Dynamic Content (사용자/세션 특화)
        ├─ 메모리 (CLAUDE.md)
        ├─ 언어 설정
        ├─ 프로젝트 컨텍스트
        └─ 워커 컨텍스트 (coordinator mode)
```

**Coordinator Mode 프롬프트**:
```
워커 지시사항 (coordinator → worker):
  - "Workers spawned via the Agent tool have access to these tools: ..."
  - "Workers also have access to MCP tools from connected MCP servers: ..."
  - "Scratchpad directory: ... (read/write without prompts)"

워커 결과 형식 (XML):
  <task-notification>
    <task-id>{agentId}</task-id>
    <status>completed|failed|killed</status>
    <summary>{status summary}</summary>
    <result>{agent final text}</result>
  </task-notification>
```

### 11.2 우리와의 차이

우리는 단순 시스템 프롬프트. 정적/동적 분리, 캐시 최적화, 도구 사용 가이드 없음.

---

## 12. Coordinator Mode (멀티 에이전트 오케스트레이션)

### 12.1 Claude Code

```
활성화: CLAUDE_CODE_COORDINATOR_MODE=1

역할 분리:
  ├─ Coordinator: 사용자 요청 분석 → 워커 배분 → 결과 종합
  └─ Worker: 실제 작업 수행 (독립적 환경)
       ├─ 병렬 실행 (여러 워커 동시)
       ├─ 격리 (워커 오류 → 다른 워커 영향 없음)
       └─ 구조화된 결과 반환 (task-notification XML)
```

### 12.2 Codex — Multi-Agent v2 Protocol

**파일**: `codex-rs/core/src/tools/handlers/multi_agents_v2/`

```
도구 구성:
  ├─ spawn.rs         → Agent spawn
  ├─ send_message.rs  → A2A 메시징
  ├─ wait.rs          → 비동기 대기
  ├─ list_agents.rs   → 활성 agents 조회
  └─ followup_task.rs → 후속 작업 지정
```

### 12.3 우리와의 차이

- 우리: A2A SDK 기반 분산 (TCP, HTTP, 무상태)
- Claude Code: 동일 프로세스 내 격리 (메시지 기반)
- Codex: 프로토콜 기반 에이전트 관리

---

## 13. 기타 혁신적 기능

### 13.1 Claude Code — Tool Search (동적 도구 발견)

- 사용자 쿼리에 맞는 도구 동적 검색
- "deferred tools" — 처음에 숨기고 필요 시에만 제공
- 쿼리당 1회 검색 (캐싱)

### 13.2 Claude Code — Thinking (적응형 추론)

```
모드:
  ├─ adaptive: 필요 시 자동 활성화
  ├─ disabled: 비활성화
  └─ enabled: 항상 활성화

특징:
  ├─ redacted_thinking (결과에 포함하지 않음)
  ├─ max_thinking_length (모델별 다름)
  └─ interleaved thinking (도구 결과 분석 전)
```

### 13.3 Claude Code — File History (변경 이력 추적)

- 각 파일 변경마다 스냅샷
- 증분 저장 (diff만)
- LRU 캐시 (메모리 효율)
- 롤백 가능

### 13.4 Codex — 계층적 설정 체계

```
설정 우선순위 (낮음 → 높음):
  1. Built-in defaults
  2. Global config.toml (~/.codex/config.toml)
  3. Project config.toml (.codex/config.toml)
  4. Environment variables
  5. CLI flags

역할 기반 설정:
  agent_roles:
    research:
      nickname_candidates: [...]
      tools: [...]
      system_prompt_override: ...
```

---

## 14. 우리 프로젝트 개선 우선순위 로드맵

### P0: 즉시 도입 (1-2주, 가장 큰 효과)

| # | 기능 | 출처 | 현재 상태 | 기대 효과 |
|---|------|------|----------|----------|
| 1 | **스트리밍 도구 병렬 실행** | Claude Code | 순차 실행 | 2-3배 속도 향상 |
| 2 | **자동 컨텍스트 컴팩션** | Claude Code | 관리 없음 | 긴 대화 안정성 |
| 3 | **Unified Diff 파일 수정** | Codex | 전체 덮어쓰기 | 안전한 코드 수정 |
| 4 | **선택적 히스토리 전파** | Codex | 전체 전달 | 토큰 50%+ 절약 |

### P1: 중기 계획 (1개월)

| # | 기능 | 출처 | 현재 상태 | 기대 효과 |
|---|------|------|----------|----------|
| 5 | **프로젝트 메타정보 (CLAUDE.md 패턴)** | Claude Code | 없음 | 프로젝트별 규칙 자동 적용 |
| 6 | **도구 권한 모델** | 양쪽 모두 | 없음 | 안전한 도구 실행 |
| 7 | **Markdown 기반 스킬 확장** | Claude Code | 기본 구현만 | 사용자 커스터마이징 |
| 8 | **max_output_tokens 복구** | Claude Code | 없음 | 응답 실패 감소 |

### P2: 장기 비전 (3개월+)

| # | 기능 | 출처 | 현재 상태 | 기대 효과 |
|---|------|------|----------|----------|
| 9 | **Coordinator Mode** | Claude Code | A2A만 | 로컬 병렬 워커 |
| 10 | **Starlark 실행 정책** | Codex | 없음 | 세밀한 도구 제어 |
| 11 | **Prompt Cache 최적화** | Claude Code | 없음 | 비용 50% 절감 |
| 12 | **Tool Search (동적 도구 발견)** | Claude Code | 없음 | 대규모 도구셋 관리 |

---

## 15. 요약

### Claude Code의 강점
- **극도로 최적화된 성능** (1ms 단위 프로파일링, lazy loading)
- **확장성 극대화** (feature flag, 동적 로드, 플러그인)
- **복합 도구 조율** (병렬 + 순서 보장 + 권한)
- **프로덕션 안정성** (에러 복구, 자동 컴팩션, 안전성)

### Codex의 강점
- **안전성 최우선** (다층 샌드박싱, Starlark 정책 엔진)
- **구조적 파일 수정** (unified diff + safety assessment)
- **효율적 히스토리 관리** (선택적 전파, truncation)
- **선언적 설정** (계층적 TOML 설정, 역할 기반)

### 우리가 가져가야 할 핵심

1. **도구 병렬 실행** — 즉시 속도 개선
2. **컨텍스트 자동 관리** — 안정성의 핵심
3. **diff 기반 파일 수정** — 안전한 코드 생성
4. **계층적 권한 모델** — 프로덕션 필수
