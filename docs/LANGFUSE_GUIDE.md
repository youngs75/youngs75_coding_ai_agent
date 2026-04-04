# Langfuse 대시보드 설정 및 사용 가이드

Langfuse는 LLM 애플리케이션을 위한 오픈소스 관측성 플랫폼입니다.
이 프로젝트에서는 에이전트 실행 추적, LLM 호출 모니터링, 평가 점수 관리를 위해 Langfuse v3를 사용합니다.

## 목차

1. [Docker Compose로 Langfuse 기동](#1-docker-compose로-langfuse-기동)
2. [초기 설정](#2-초기-설정)
3. [환경변수 설정](#3-환경변수-설정)
4. [CLI에서 Langfuse 활성화 확인](#4-cli에서-langfuse-활성화-확인)
5. [대시보드 주요 메트릭](#5-대시보드-주요-메트릭)
6. [데모 트레이스 생성](#6-데모-트레이스-생성)
7. [커스텀 대시보드 뷰 구성](#7-커스텀-대시보드-뷰-구성)
8. [아키텍처 개요](#8-아키텍처-개요)
9. [문제 해결](#9-문제-해결)

---

## 1. Docker Compose로 Langfuse 기동

이 프로젝트는 Langfuse v3의 전체 스택을 Docker Compose로 제공합니다.

### 서비스 구성

| 서비스 | 이미지 | 역할 | 기본 포트 |
|--------|--------|------|-----------|
| `langfuse-web` | `langfuse/langfuse:3` | 웹 대시보드 + API 서버 | 3100 |
| `langfuse-worker` | `langfuse/langfuse-worker:3` | 비동기 이벤트 처리 워커 | 3030 |
| `postgres` | `postgres:17` | 메인 데이터베이스 | 5432 |
| `clickhouse` | `clickhouse-server:25.8` | 분석용 OLAP 데이터베이스 | 8123, 9005 |
| `redis` | `redis:7.2.4` | 캐시 및 큐 | 6379 |
| `minio` | `minio` | S3 호환 오브젝트 스토리지 | 9090, 9091 |

### 기동 명령

```bash
# Langfuse 스택 시작
docker compose -f docker/docker-compose.langfuse.yaml up -d

# 상태 확인
docker compose -f docker/docker-compose.langfuse.yaml ps

# 로그 확인
docker compose -f docker/docker-compose.langfuse.yaml logs -f langfuse-web

# 중지
docker compose -f docker/docker-compose.langfuse.yaml down

# 중지 + 데이터 삭제 (주의: 모든 트레이스 데이터 삭제)
docker compose -f docker/docker-compose.langfuse.yaml down -v
```

### 자동 설정 스크립트

```bash
# Docker Compose 기동 후 헬스체크 및 설정 검증
python scripts/setup_langfuse.py --wait --verbose

# 설정 가이드만 출력
python scripts/setup_langfuse.py --guide
```

---

## 2. 초기 설정

Docker Compose로 서버를 기동한 후 브라우저에서 초기 설정을 진행합니다.

### 2-1. 회원가입

1. 브라우저에서 `http://localhost:3100` 접속
2. "Sign Up" 클릭
3. 이메일, 이름, 비밀번호 입력
4. 첫 번째 가입 사용자가 자동으로 관리자(Admin) 권한을 갖습니다

### 2-2. 프로젝트 생성

1. 로그인 후 "New Project" 클릭
2. 프로젝트 이름 입력 (예: `youngs75-a2a`)
3. "Create" 클릭

### 2-3. API 키 발급

1. 프로젝트 대시보드에서 좌측 메뉴 > "Settings"
2. "API Keys" 탭 선택
3. "Create API Key" 클릭
4. 생성된 키를 복사:
   - **Public Key**: `pk-lf-xxxxxxxx` 형식
   - **Secret Key**: `sk-lf-xxxxxxxx` 형식

> **주의**: Secret Key는 생성 시에만 표시됩니다. 반드시 즉시 복사하세요.

---

## 3. 환경변수 설정

### .env 파일에 추가

프로젝트 루트의 `.env` 파일에 다음을 추가합니다:

```bash
# ── Langfuse (관측성) ──
LANGFUSE_HOST=http://localhost:3100
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_TRACING_ENABLED=1
LANGFUSE_SAMPLE_RATE=1.0
```

### Langfuse Docker 전용 환경변수

Langfuse 서버 자체의 설정은 `docker/.env.langfuse.example` 파일을 참고하여 `docker/.env.langfuse` 파일을 생성합니다:

```bash
cp docker/.env.langfuse.example docker/.env.langfuse
# 필요한 값을 수정
```

### 환경변수 설명

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `LANGFUSE_HOST` | Langfuse 서버 URL | `http://localhost:3100` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 프로젝트 공개 키 | (필수) |
| `LANGFUSE_SECRET_KEY` | Langfuse 프로젝트 비밀 키 | (필수) |
| `LANGFUSE_TRACING_ENABLED` | 트레이싱 활성화 여부 (`1`/`0`) | `1` |
| `LANGFUSE_SAMPLE_RATE` | 트레이스 샘플링 비율 (0.0~1.0) | `1.0` |
| `CLI_LANGFUSE_ENABLED` | CLI 레벨 Langfuse 토글 | `1` |

### 설정 검증

```bash
python scripts/setup_langfuse.py --verbose
```

---

## 4. CLI에서 Langfuse 활성화 확인

CLI 실행 시 Langfuse가 올바르게 설정되면 다음 메시지가 표시됩니다:

```
Langfuse 관측성 활성화됨
```

### 활성화 조건

CLI에서 Langfuse가 활성화되려면 다음 조건이 모두 충족되어야 합니다:

1. `CLI_LANGFUSE_ENABLED=1` (cli/config.py의 CLIConfig)
2. `LANGFUSE_TRACING_ENABLED=1` (eval_pipeline/settings.py의 Settings)
3. `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`가 모두 설정됨
4. `langfuse` 파이썬 패키지가 설치됨

### 비활성화 방법

```bash
# 방법 1: CLI 레벨에서 비활성화
export CLI_LANGFUSE_ENABLED=0

# 방법 2: 전체 트레이싱 비활성화
export LANGFUSE_TRACING_ENABLED=0
```

---

## 5. 대시보드 주요 메트릭

Langfuse 대시보드(`http://localhost:3100`)에서 확인할 수 있는 주요 정보입니다.

### 5-1. Traces (에이전트 실행 추적)

각 에이전트 실행이 하나의 Trace로 기록됩니다.

- **Trace 이름**: 에이전트 이름 (예: `coding_assistant`, `deep_research`)
- **Session ID**: 멀티턴 대화의 세션 식별자
- **User ID**: 사용자 식별자
- **Tags**: 환경(`env:local`), 서비스(`service:youngs75-a2a`), 에이전트 태그
- **Metadata**: 앱 버전, 환경 정보

**활용법**:
- 태그 기반 필터링으로 특정 에이전트/환경의 실행만 조회
- 세션 뷰에서 멀티턴 대화의 전체 흐름 추적
- 시간 범위 지정으로 특정 기간의 실행 분석

### 5-2. Spans & Generations (노드별 실행 상세)

각 에이전트 그래프 노드가 Span으로, LLM 호출이 Generation으로 기록됩니다.

**coding_assistant 노드**:
| 노드 | 설명 | 평균 소요 |
|------|------|-----------|
| `parse_request` | 요청 분석 | ~100ms |
| `execute_code` | 코드 생성 | ~1.5s |
| `verify_result` | 코드 검증 | ~500ms |

**deep_research 노드**:
| 노드 | 설명 | 평균 소요 |
|------|------|-----------|
| `clarify_with_user` | 질문 명확화 | ~200ms |
| `write_research_brief` | 연구 브리프 | ~350ms |
| `research_supervisor` | 연구 수행 | ~5s |
| `final_report_generation` | 보고서 작성 | ~2.5s |

**Generation 상세 정보**:
- 모델명 (gpt-4o, gpt-4o-mini 등)
- 프롬프트/완료 토큰 수
- 입력/출력 내용
- 소요 시간

### 5-3. Scores (평가 점수)

에이전트 메트릭 수집기(`AgentMetricsCollector`)가 기록하는 스코어:

| 스코어 이름 | 타입 | 설명 |
|-------------|------|------|
| `agent.total_tokens` | NUMERIC | 총 토큰 사용량 |
| `agent.prompt_tokens` | NUMERIC | 프롬프트 토큰 |
| `agent.completion_tokens` | NUMERIC | 완료 토큰 |
| `agent.duration_ms` | NUMERIC | 전체 소요 시간 |
| `agent.error_count` | NUMERIC | 에러 수 |

평가 파이프라인(`eval_pipeline`)이 기록하는 스코어:

| 스코어 이름 | 타입 | 설명 |
|-------------|------|------|
| `deepeval.faithfulness` | NUMERIC | 응답 충실도 |
| `deepeval.answer_relevancy` | NUMERIC | 응답 관련성 |
| `quality.passed` | BOOLEAN | 검증 통과 여부 |
| `quality.risk_level` | CATEGORICAL | 위험도 수준 |

**활용법**:
- 스코어 기반 필터링 — 실패한 트레이스만 조회
- 시계열 그래프 — 품질 메트릭 추이 모니터링
- 히스토그램 — 스코어 분포 분석

### 5-4. Cost (토큰 비용 추적)

Generation에 기록된 토큰 사용량을 기반으로 비용을 자동 계산합니다.

- **모델별 비용**: 각 LLM 모델의 토큰 비용 집계
- **시간별 비용**: 일별/주별 비용 추이
- **사용자별 비용**: 사용자 ID 기반 비용 분석
- **세션별 비용**: 대화 세션별 총 비용

---

## 6. 데모 트레이스 생성

대시보드 확인을 위한 샘플 데이터를 생성할 수 있습니다:

```bash
# 기본 실행 (에이전트별 3개 시나리오)
python scripts/langfuse_demo.py

# 더 많은 시나리오 생성
python scripts/langfuse_demo.py --count 10

# 특정 에이전트만
python scripts/langfuse_demo.py --agent coding_assistant

# 드라이런 (실제 전송 없이 확인)
python scripts/langfuse_demo.py --dry-run --count 5

# 재현 가능한 결과
python scripts/langfuse_demo.py --seed 42
```

데모 실행 후 대시보드에서 `demo` 태그로 필터링하면 생성된 트레이스를 확인할 수 있습니다.

---

## 7. 커스텀 대시보드 뷰 구성

Langfuse v3 대시보드에서 프로젝트에 맞는 뷰를 구성하는 방법입니다.

### 7-1. 태그 기반 필터 프리셋

Traces 목록에서 자주 사용하는 필터 조합을 저장합니다:

- **에이전트별**: `Tags contains "agent:coding_assistant"`
- **환경별**: `Tags contains "env:prod"`
- **실패만**: `Scores: quality.passed < 0.5`
- **고비용**: `Total Cost > 0.01`

### 7-2. 세션 뷰 활용

멀티턴 대화를 세션 단위로 분석합니다:
1. Sessions 탭 선택
2. 특정 세션 클릭
3. 해당 세션의 모든 트레이스를 시간순으로 확인
4. 대화 흐름과 각 턴의 품질을 추적

### 7-3. 사용자 뷰 활용

사용자별 사용 패턴을 분석합니다:
1. Users 탭 선택
2. 특정 사용자 클릭
3. 해당 사용자의 세션 목록, 총 비용, 평균 품질 확인

### 7-4. 평가 대시보드

평가 파이프라인의 결과를 모니터링합니다:
1. Scores 탭에서 시계열 그래프 확인
2. 특정 스코어(예: `deepeval.faithfulness`)의 추이 관찰
3. 임계값 미달 트레이스를 즉시 조회하여 원인 분석

### 7-5. 프롬프트 관리

Langfuse의 프롬프트 관리 기능을 활용할 수 있습니다:
1. Prompts 탭에서 프롬프트 버전 관리
2. A/B 테스트를 위한 프롬프트 변형 생성
3. 프롬프트 버전별 성능 비교

---

## 8. 아키텍처 개요

```
CLI (cli/app.py)
 |
 |-- CLIConfig.langfuse_enabled (CLI 레벨 토글)
 |
 |-- create_langfuse_handler()     # 콜백 핸들러 생성
 |-- build_observed_config()       # 실행 config 구성
 |
 |-- agent.graph.astream_events()  # 에이전트 실행
 |     |
 |     |-- LangChain CallbackHandler  # 자동 수집
 |     |     |-- Trace 생성
 |     |     |-- Span (노드별)
 |     |     |-- Generation (LLM 호출)
 |     |
 |     |-- AgentMetricsCollector      # 메트릭 수집
 |           |-- 토큰 사용량
 |           |-- 노드별 소요 시간
 |           |-- 에러 수
 |
 |-- safe_flush()                  # 버퍼 전송
 |
 v
Langfuse Server (docker/docker-compose.langfuse.yaml)
 |-- langfuse-web (대시보드 + API)
 |-- langfuse-worker (비동기 처리)
 |-- postgres (메인 DB)
 |-- clickhouse (분석 DB)
 |-- redis (캐시/큐)
 |-- minio (오브젝트 스토리지)
```

### 코드 위치

| 파일 | 역할 |
|------|------|
| `eval_pipeline/observability/langfuse.py` | 코어 SDK 유틸리티 (enabled, client, score_trace, enrich_trace) |
| `eval_pipeline/observability/callback_handler.py` | 콜백 핸들러 팩토리, 메트릭 수집기 |
| `eval_pipeline/settings.py` | Langfuse 관련 환경변수 설정 |
| `cli/config.py` | CLI 레벨 Langfuse 토글 |
| `cli/app.py` | CLI 메인 루프 (콜백 핸들러 주입) |
| `docker/docker-compose.langfuse.yaml` | Langfuse 서버 Docker Compose |
| `scripts/setup_langfuse.py` | 초기 설정 및 헬스체크 |
| `scripts/langfuse_demo.py` | 데모 트레이스 생성 |

---

## 9. 문제 해결

### Langfuse 서버에 연결할 수 없음

```bash
# 1. Docker 서비스 상태 확인
docker compose -f docker/docker-compose.langfuse.yaml ps

# 2. 서버 로그 확인
docker compose -f docker/docker-compose.langfuse.yaml logs langfuse-web

# 3. 헬스체크
curl http://localhost:3100/api/public/health

# 4. 포트 충돌 확인
lsof -i :3100
```

### "Langfuse 비활성화 상태" 메시지

환경변수가 올바르게 설정되었는지 확인합니다:

```bash
python scripts/setup_langfuse.py --verbose
```

필수 항목:
- `LANGFUSE_HOST`가 설정되어 있는지
- `LANGFUSE_PUBLIC_KEY`가 `pk-lf-` 형식인지
- `LANGFUSE_SECRET_KEY`가 `sk-lf-` 형식인지
- `LANGFUSE_TRACING_ENABLED=1`인지

### 트레이스가 대시보드에 표시되지 않음

1. `safe_flush()`가 호출되고 있는지 확인 (cli/app.py의 finally 블록)
2. Langfuse 서버의 Worker 로그 확인:
   ```bash
   docker compose -f docker/docker-compose.langfuse.yaml logs langfuse-worker
   ```
3. 네트워크 연결 확인 (방화벽, 프록시 등)
4. 샘플 레이트 확인: `LANGFUSE_SAMPLE_RATE=1.0` (100% 수집)

### 메모리/디스크 이슈

ClickHouse와 PostgreSQL의 볼륨이 증가할 수 있습니다:

```bash
# 볼륨 사용량 확인
docker system df -v | grep langfuse

# 오래된 데이터 정리 (Langfuse 대시보드의 Settings에서 데이터 보존 정책 설정)
```
