#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# youngs75-coding-ai-agent — 원커맨드 초기 설정
#
# 사용법:
#   make setup          (권장)
#   bash scripts/setup.sh
#
# 이 스크립트가 하는 일:
#   1. Python 3.13 확인 (없으면 uv로 자동 설치)
#   2. uv 확인 (없으면 자동 설치)
#   3. 의존성 설치 (uv sync)
#   4. .env 파일 생성 + 필수 API 키 안내
#   5. Docker MCP 서버 기동 (선택)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── 색상 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── 프로젝트 루트로 이동 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ── 헬퍼 함수 ──
info()  { echo -e "${CYAN}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}!${NC} $*"; }
fail()  { echo -e "${RED}✗${NC} $*"; exit 1; }

# 대화형 여부 감지
if [ -t 0 ]; then
    INTERACTIVE=true
else
    INTERACTIVE=false
fi

ask_yn() {
    # $1: 프롬프트, 기본값 Y
    if [ "$INTERACTIVE" = true ]; then
        read -rp "  $1 [Y/n] " REPLY
        REPLY=${REPLY:-Y}
    else
        REPLY=Y
        info "$1 → 자동 수락 (비대화형 모드)"
    fi
    [[ "$REPLY" =~ ^[Yy]$ ]]
}

step() {
    echo ""
    echo -e "${BOLD}[$1/6] $2${NC}"
    echo "────────────────────────────────────────"
}

# API 키 입력 헬퍼 (입력 중 * 표시 + 글자 수 피드백)
read_secret() {
    local prompt="$1"
    if [ "$INTERACTIVE" = true ]; then
        REPLY_SECRET=""
        local char=""
        printf "  %s: " "$prompt"
        while IFS= read -rsn1 char; do
            # Enter → 입력 완료
            if [[ -z "$char" ]]; then
                break
            fi
            # Backspace 처리
            if [[ "$char" == $'\x7f' ]] || [[ "$char" == $'\b' ]]; then
                if [ ${#REPLY_SECRET} -gt 0 ]; then
                    REPLY_SECRET="${REPLY_SECRET%?}"
                    printf '\b \b'
                fi
            else
                REPLY_SECRET+="$char"
                printf '*'
            fi
        done
        echo ""
        # 입력 확인 피드백
        if [ -n "$REPLY_SECRET" ]; then
            local len=${#REPLY_SECRET}
            local masked="${REPLY_SECRET:0:3}***${REPLY_SECRET: -3}"
            echo -e "  ${GREEN}→${NC} ${len}자 입력됨 (${masked})"
        else
            echo -e "  ${YELLOW}→${NC} 입력 없음"
        fi
    else
        REPLY_SECRET=""
    fi
}

# ═══════════════════════════════════════════════════════════════
# Step 1: uv 확인/설치
# ═══════════════════════════════════════════════════════════════
step 1 "패키지 매니저 (uv) 확인"

if command -v uv &>/dev/null; then
    ok "uv $(uv --version 2>/dev/null | head -1) 설치됨"
else
    info "uv가 없습니다. 설치합니다..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # PATH에 추가 (현재 세션)
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv 설치 완료"
    else
        fail "uv 설치 실패. https://docs.astral.sh/uv/getting-started/installation/ 참고"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Step 2: Python 3.13 확인
# ═══════════════════════════════════════════════════════════════
step 2 "Python 3.13 확인"

# uv가 관리하는 Python 또는 시스템 Python 확인
PYTHON_OK=false

# uv python list에서 3.13 확인
if uv python find 3.13 &>/dev/null; then
    PY_PATH=$(uv python find 3.13 2>/dev/null)
    PY_VER=$("$PY_PATH" --version 2>/dev/null || echo "unknown")
    ok "Python 3.13 발견: $PY_VER ($PY_PATH)"
    PYTHON_OK=true
fi

if [ "$PYTHON_OK" = false ]; then
    info "Python 3.13이 없습니다. uv로 설치합니다..."
    uv python install 3.13
    if uv python find 3.13 &>/dev/null; then
        PY_PATH=$(uv python find 3.13 2>/dev/null)
        PY_VER=$("$PY_PATH" --version 2>/dev/null || echo "unknown")
        ok "Python 3.13 설치 완료: $PY_VER"
    else
        fail "Python 3.13 설치 실패. 수동 설치 필요: uv python install 3.13"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Step 3: 의존성 설치
# ═══════════════════════════════════════════════════════════════
step 3 "의존성 설치 (uv sync)"

if [ -f "uv.lock" ]; then
    info "uv.lock 발견 — frozen 모드로 설치"
    uv sync --frozen
else
    info "의존성 해석 및 설치 중..."
    uv sync
fi

# 설치 확인
if uv run python -c "import youngs75_a2a" 2>/dev/null; then
    ok "패키지 설치 완료 (youngs75_a2a)"
else
    warn "패키지 임포트 확인 실패 — uv pip install -e . 시도"
    uv pip install -e .
fi

# ═══════════════════════════════════════════════════════════════
# Step 4: .env 설정
# ═══════════════════════════════════════════════════════════════
step 4 "환경변수 설정 (.env)"

SKIP_ENV=false

if [ -f ".env" ]; then
    warn ".env 파일이 이미 존재합니다"
    if [ "$INTERACTIVE" = true ]; then
        if ask_yn "기존 .env를 새로 설정할까요?"; then
            info "기존 .env를 백업합니다 → .env.backup"
            cp .env .env.backup
        else
            ok "기존 .env를 유지합니다"
            SKIP_ENV=true
        fi
    else
        ok "기존 .env를 유지합니다 (비대화형 모드)"
        SKIP_ENV=true
    fi
fi

if [ "$SKIP_ENV" = false ]; then
    if [ "$INTERACTIVE" = true ]; then
        # ── 메인 프로바이더 선택 ──
        echo ""
        echo -e "  ${BOLD}메인 LLM 프로바이더를 선택하세요:${NC}"
        echo -e "    ${CYAN}1)${NC} DashScope (Qwen3 직접 API — 권장, 낮은 레이턴시)"
        echo -e "    ${CYAN}2)${NC} OpenRouter (다중 모델 라우터)"
        echo ""
        read -rp "  선택 [1]: " PROVIDER_CHOICE
        PROVIDER_CHOICE=${PROVIDER_CHOICE:-1}

        DASHSCOPE_KEY=""
        OPENROUTER_KEY=""

        if [ "$PROVIDER_CHOICE" = "2" ]; then
            PROVIDER="openrouter"
            MAIN_LABEL="OpenRouter"
            BACKUP_LABEL="DashScope"
        else
            PROVIDER="dashscope"
            MAIN_LABEL="DashScope"
            BACKUP_LABEL="OpenRouter"
        fi

        # ── 메인 프로바이더 API 키 (필수) ──
        echo ""
        echo -e "  ${BOLD}[메인] ${MAIN_LABEL} API 키${NC} (필수)"
        if [ "$PROVIDER" = "dashscope" ]; then
            echo -e "  발급: ${CYAN}https://bailian.console.alibabacloud.com/${NC}"
            read_secret "DASHSCOPE_API_KEY"
            DASHSCOPE_KEY="$REPLY_SECRET"
            if [ -z "$DASHSCOPE_KEY" ]; then
                fail "메인 프로바이더 API 키는 필수입니다"
            fi
        else
            echo -e "  발급: ${CYAN}https://openrouter.ai/keys${NC}"
            read_secret "OPENROUTER_API_KEY"
            OPENROUTER_KEY="$REPLY_SECRET"
            if [ -z "$OPENROUTER_KEY" ]; then
                fail "메인 프로바이더 API 키는 필수입니다"
            fi
        fi
        ok "${MAIN_LABEL} API 키 설정 완료 (메인)"

        # ── 백업 프로바이더 API 키 (권장) ──
        echo ""
        echo -e "  ${BOLD}[백업] ${BACKUP_LABEL} API 키${NC} (권장 — DR 구성)"
        echo -e "  메인 프로바이더 장애 시 자동 전환됩니다. 없으면 Enter로 건너뛰세요."
        if [ "$PROVIDER" = "dashscope" ]; then
            echo -e "  발급: ${CYAN}https://openrouter.ai/keys${NC}"
            read_secret "OPENROUTER_API_KEY (백업)"
            OPENROUTER_KEY="$REPLY_SECRET"
        else
            echo -e "  발급: ${CYAN}https://bailian.console.alibabacloud.com/${NC}"
            read_secret "DASHSCOPE_API_KEY (백업)"
            DASHSCOPE_KEY="$REPLY_SECRET"
        fi
        if [ -n "$REPLY_SECRET" ]; then
            ok "${BACKUP_LABEL} API 키 설정 완료 (백업)"
        else
            info "${BACKUP_LABEL} 건너뜀 — 단일 프로바이더로 운영"
        fi

        # ── Tavily API 키 (선택) ──
        echo ""
        echo -e "  ${BOLD}Tavily 웹 검색 API${NC} (선택사항)"
        echo -e "  DeepResearch 에이전트에 필요합니다. 없으면 Enter로 건너뛰세요."
        echo -e "  발급: ${CYAN}https://app.tavily.com/home${NC}"
        read_secret "TAVILY_API_KEY"
        TAVILY_KEY="$REPLY_SECRET"
        if [ -n "$TAVILY_KEY" ]; then
            ok "Tavily API 키 설정 완료"
        else
            info "Tavily 건너뜀 — 나중에 .env에서 설정 가능"
        fi

        # ── Langfuse 관측성 (선택) ──
        LF_PUB_KEY=""
        LF_SEC_KEY=""
        LF_ENABLED="0"
        echo ""
        if ask_yn "Langfuse 관측성을 설정할까요? (선택사항)"; then
            echo -e "  발급: ${CYAN}https://cloud.langfuse.com${NC}"
            read_secret "LANGFUSE_PUBLIC_KEY"
            LF_PUB_KEY="$REPLY_SECRET"
            read_secret "LANGFUSE_SECRET_KEY"
            LF_SEC_KEY="$REPLY_SECRET"
            if [ -n "$LF_PUB_KEY" ] && [ -n "$LF_SEC_KEY" ]; then
                LF_ENABLED="1"
                ok "Langfuse 키 설정 완료"
            else
                warn "Langfuse 키가 비어있어 비활성화됩니다"
            fi
        else
            info "Langfuse 건너뜀 — LiteLLM Proxy에서 나중에 설정 가능"
        fi

        # ── .env 생성 (sed 치환) ──
        info ".env.example 기반으로 .env 생성 중..."
        sed \
            -e "s|^LLM_PROVIDER=.*|LLM_PROVIDER=${PROVIDER}|" \
            -e "s|^DASHSCOPE_API_KEY=.*|DASHSCOPE_API_KEY=${DASHSCOPE_KEY}|" \
            -e "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=${OPENROUTER_KEY}|" \
            -e "s|^TAVILY_API_KEY=.*|TAVILY_API_KEY=${TAVILY_KEY}|" \
            -e "s|^LANGFUSE_PUBLIC_KEY=.*|LANGFUSE_PUBLIC_KEY=${LF_PUB_KEY}|" \
            -e "s|^LANGFUSE_SECRET_KEY=.*|LANGFUSE_SECRET_KEY=${LF_SEC_KEY}|" \
            -e "s|^LANGFUSE_TRACING_ENABLED=.*|LANGFUSE_TRACING_ENABLED=${LF_ENABLED}|" \
            .env.example > .env

        ok ".env 파일 생성 완료"
        echo ""

        # 백업 키 상태 판별
        if [ "$PROVIDER" = "dashscope" ]; then
            BACKUP_STATUS=$([ -n "$OPENROUTER_KEY" ] && echo "✓" || echo "미설정")
        else
            BACKUP_STATUS=$([ -n "$DASHSCOPE_KEY" ] && echo "✓" || echo "미설정")
        fi
        TAVILY_STATUS=$([ -n "$TAVILY_KEY" ] && echo "✓" || echo "미설정")
        LF_STATUS=$([ "$LF_ENABLED" = "1" ] && echo "✓" || echo "미설정")

        echo -e "  ${BOLD}메인:${NC}     ${MAIN_LABEL} ✓"
        echo -e "  ${BOLD}백업:${NC}     ${BACKUP_LABEL} ${BACKUP_STATUS}"
        echo -e "  ${BOLD}Tavily:${NC}   ${TAVILY_STATUS}"
        echo -e "  ${BOLD}Langfuse:${NC} ${LF_STATUS}"
        echo ""
    else
        # 비대화형 모드: 환경변수에서 읽거나 템플릿 복사
        info "비대화형 모드 — 환경변수에서 .env 생성"
        if [ -n "${DASHSCOPE_API_KEY:-}" ] || [ -n "${OPENROUTER_API_KEY:-}" ]; then
            sed \
                -e "s|^LLM_PROVIDER=.*|LLM_PROVIDER=${LLM_PROVIDER:-dashscope}|" \
                -e "s|^DASHSCOPE_API_KEY=.*|DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-}|" \
                -e "s|^OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}|" \
                -e "s|^TAVILY_API_KEY=.*|TAVILY_API_KEY=${TAVILY_API_KEY:-}|" \
                -e "s|^LANGFUSE_PUBLIC_KEY=.*|LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-}|" \
                -e "s|^LANGFUSE_SECRET_KEY=.*|LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-}|" \
                .env.example > .env
            ok ".env 생성 완료 (환경변수 기반)"
        else
            cp .env.example .env
            warn ".env 생성됨 — API 키를 수동으로 입력해야 합니다"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Step 5: Docker MCP 서버 (선택)
# ═══════════════════════════════════════════════════════════════
step 5 "Docker 하니스 기동"

DOCKER_STARTED=false

if ! command -v docker &>/dev/null; then
    warn "Docker가 설치되어 있지 않습니다"
    echo ""
    echo "  Docker 설치 후: make up-harness"
    echo ""
elif ! docker info &>/dev/null 2>&1; then
    warn "Docker 데몬이 실행 중이 아닙니다"
    echo ""
    echo "  Docker Desktop을 시작한 뒤: make up-harness"
    echo ""
else
    # 이미 하니스 컨테이너가 실행 중인지 확인
    HARNESS_RUNNING=$(docker ps --filter "name=harness_" --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$HARNESS_RUNNING" -gt 0 ]; then
        ok "하니스 컨테이너 ${HARNESS_RUNNING}개 실행 중"
        docker ps --filter "name=harness_" --format "  {{.Names}}\t{{.Status}}" 2>/dev/null
        echo ""
        if ask_yn "하니스를 재빌드할까요?"; then
            info "하니스 재빌드 및 재기동 중... (수 분 소요될 수 있습니다)"
            cd docker && docker compose -f docker-compose.harness.yml up -d --build && cd ..
            DOCKER_STARTED=true
            ok "하니스 재빌드 완료"
        fi
    else
        echo ""
        if ask_yn "전체 AI Coding Agent 하니스를 빌드 및 기동할까요?"; then
            info "하니스 빌드 및 기동 중... (첫 빌드는 수 분 걸릴 수 있습니다)"
            cd docker && docker compose -f docker-compose.harness.yml up -d --build && cd ..
            DOCKER_STARTED=true
            ok "하니스 기동 완료"
        else
            info "나중에 실행하려면: make up-harness"
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════
# Step 6: 헬스체크
# ═══════════════════════════════════════════════════════════════
step 6 "서비스 헬스체크"

if [ "$DOCKER_STARTED" = true ]; then
    info "서비스 초기화 대기 중... (최대 120초)"
    bash "$ROOT_DIR/scripts/health_check.sh" --wait 120 || true
else
    # Docker를 기동하지 않았으면 실행 중인 서비스만 확인
    if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        HARNESS_RUNNING=$(docker ps --filter "name=harness_" --format "{{.Names}}" 2>/dev/null | wc -l)
        if [ "$HARNESS_RUNNING" -gt 0 ]; then
            bash "$ROOT_DIR/scripts/health_check.sh" || true
        else
            info "실행 중인 하니스 서비스 없음 — 헬스체크 건너뜀"
        fi
    else
        info "Docker 미사용 — 헬스체크 건너뜀"
    fi
fi

# ═══════════════════════════════════════════════════════════════
# 완료
# ═══════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}${BOLD}  설정 완료!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  CLI 실행:     youngs75-agent"
echo "  하니스 기동:  make up-harness"
echo "  하니스 종료:  make down-harness"
echo "  로그 확인:    make logs-harness"
echo "  헬스체크:     make health-check"
echo "  도움말:       make help"
echo ""
