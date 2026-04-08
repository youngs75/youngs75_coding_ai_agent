# ═══════════════════════════════════════════════════════════════
# youngs75-coding-ai-agent — 개발 편의 Makefile
# ═══════════════════════════════════════════════════════════════
# 사용법:
#   make help       — 사용 가능한 명령 목록
#   make lint       — ruff 린트 + 포맷 검사
#   make test       — pytest 실행 (eval 제외)
#   make build      — Docker 이미지 빌드
#   make up         — Docker Compose 전체 기동
#   make down       — Docker Compose 종료
# ═══════════════════════════════════════════════════════════════

.DEFAULT_GOAL := help
.PHONY: help setup lint format test test-eval build up down logs clean mcp-up mcp-down up-harness down-harness logs-harness ps-harness health-check

# ── 초기 설정 ──

setup: ## 원커맨드 초기 설정 (Python + 의존성 + .env + MCP)
	@bash scripts/setup.sh

# ── 린트 & 포맷 ──

lint: ## ruff 린트 + 포맷 검사
	uv run ruff check .
	uv run ruff format --check .

format: ## ruff 자동 포맷 적용
	uv run ruff check --fix .
	uv run ruff format .

# ── 테스트 ──

test: ## pytest 실행 (eval 테스트 제외)
	uv run pytest tests/ --ignore=tests/eval/ -v

test-eval: ## eval 파이프라인 테스트 실행
	uv run pytest tests/eval/ -v

test-all: ## 전체 테스트 실행
	uv run pytest tests/ -v

# ── Docker ──

build: ## Docker Compose 이미지 빌드
	cd docker && docker compose build

up: ## Docker Compose 전체 기동 (백그라운드)
	cd docker && docker compose up -d

down: ## Docker Compose 종료 및 컨테이너 제거
	cd docker && docker compose down

logs: ## Docker Compose 로그 스트리밍
	cd docker && docker compose logs -f

ps: ## Docker Compose 서비스 상태 확인
	cd docker && docker compose ps

# ── MCP 서버 ──

mcp-up: ## MCP 도구 서버 기동
	cd docker && docker compose -f docker-compose.mcp.yml up -d

mcp-down: ## MCP 도구 서버 종료
	cd docker && docker compose -f docker-compose.mcp.yml down

# ── Harness (전체 에이전트 하니스) ──

up-harness: ## 전체 하니스 기동 (MCP + 에이전트 + 오케스트레이터)
	cd docker && docker compose -f docker-compose.harness.yml up -d --build

down-harness: ## 전체 하니스 종료
	cd docker && docker compose -f docker-compose.harness.yml down

logs-harness: ## 하니스 로그 스트리밍
	cd docker && docker compose -f docker-compose.harness.yml logs -f

ps-harness: ## 하니스 서비스 상태 확인
	cd docker && docker compose -f docker-compose.harness.yml ps

health-check: ## 모든 하니스 서비스 헬스체크
	@bash scripts/health_check.sh

# ── 환경별 Docker 기동 ──

up-dev: ## 개발 환경으로 Docker 기동
	cd docker && docker compose --env-file ../config/settings.dev.env up -d

up-staging: ## 스테이징 환경으로 Docker 기동
	cd docker && docker compose --env-file ../config/settings.staging.env up -d

up-prod: ## 프로덕션 환경으로 Docker 기동
	cd docker && docker compose --env-file ../config/settings.prod.env up -d

# ── 유틸리티 ──

sync: ## uv 의존성 동기화
	uv sync --frozen

clean: ## 캐시 및 빌드 산출물 정리
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage

# ── 도움말 ──

help: ## 사용 가능한 명령 목록 표시
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
