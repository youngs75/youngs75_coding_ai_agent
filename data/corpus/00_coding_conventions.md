# Coding Agent 코딩 컨벤션

## Python 코딩 규칙

### 명명 규칙
- 함수/변수: `snake_case`
- 클래스: `PascalCase`
- 상수: `UPPER_SNAKE_CASE`
- 비공개 멤버: `_leading_underscore`

### 타입 힌트
- 모든 함수 시그니처에 타입 힌트 필수
- `from __future__ import annotations` 사용
- Optional 대신 `X | None` 사용 (Python 3.10+)

### 에러 처리
- 빈 except 금지 — 항상 구체적 예외 타입 명시
- 사용자 입력은 반드시 검증
- 외부 API 호출은 try/except로 감싸고 재시도 로직 포함

### 보안 규칙
- 하드코딩된 시크릿 금지 — 환경변수 또는 시크릿 매니저 사용
- SQL 쿼리는 파라미터 바인딩 필수
- 사용자 입력을 직접 eval/exec에 전달 금지

### 테스트 규칙
- 모든 공개 함수에 단위 테스트 작성
- pytest 사용, fixture로 공통 설정 관리
- 외부 의존성은 mock 처리

### 프로젝트 구조
- 에이전트는 `agents/` 하위에 배치
- 공통 로직은 `core/`에 배치
- 설정은 Pydantic BaseModel/BaseSettings 사용
