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
.PHONY: help setup lint format test test-eval build up down logs ps health-check cli sync clean

# ── 초기 설정 ──

setup: ## 원커맨드 초기 설정 (Python + 의존성 + .env + MCP)
	@bash coding_agent/scripts/setup.sh

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

# ── Docker (Harness 통합) ──

build: ## Docker 이미지 빌드
	cd docker && docker compose build

up: ## 전체 Harness 기동 (MCP + LiteLLM + Agent + Orchestrator)
	cd docker && docker compose up -d --build

down: ## 전체 Harness 종료 및 컨테이너 제거
	cd docker && docker compose down

logs: ## Docker 로그 스트리밍
	cd docker && docker compose logs -f

ps: ## Docker 서비스 상태 확인
	cd docker && docker compose ps

health-check: ## 모든 서비스 헬스체크
	@bash coding_agent/scripts/health_check.sh

cli: ## 대화형 CLI 실행 (Docker 우선, 로컬 폴백)
	@bash youngs75-agent.sh

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
