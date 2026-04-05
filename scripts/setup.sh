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
    echo -e "${BOLD}[$1/5] $2${NC}"
    echo "────────────────────────────────────────"
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

if [ -f ".env" ]; then
    # .env가 이미 있으면 필수 키만 확인
    ok ".env 파일 존재"

    # LLM 프로바이더 확인
    LLM_PROVIDER=$(grep '^LLM_PROVIDER=' .env 2>/dev/null | cut -d= -f2)
    LLM_PROVIDER=${LLM_PROVIDER:-openrouter}

    if [ "$LLM_PROVIDER" = "dashscope" ]; then
        # DashScope (기본 프로바이더) API 키 확인
        if grep -q '^DASHSCOPE_API_KEY=sk-' .env 2>/dev/null; then
            ok "DashScope API 키 설정됨 (LLM_PROVIDER=dashscope)"
        else
            warn "DASHSCOPE_API_KEY가 설정되지 않았습니다"
            echo ""
            echo -e "  ${BOLD}필수:${NC} .env 파일에 DashScope API 키를 입력하세요"
            echo -e "  발급: ${CYAN}https://bailian.console.alibabacloud.com/${NC}"
            echo ""
        fi
    else
        # OpenRouter API 키 확인
        if grep -q '^OPENROUTER_API_KEY=sk-or-' .env 2>/dev/null; then
            ok "OpenRouter API 키 설정됨 (LLM_PROVIDER=openrouter)"
        else
            warn "OPENROUTER_API_KEY가 비어있습니다"
            echo ""
            echo -e "  ${BOLD}필수:${NC} .env 파일에 OpenRouter API 키를 입력하세요"
            echo -e "  발급: ${CYAN}https://openrouter.ai/keys${NC}"
            echo ""
        fi
    fi
else
    info ".env.example에서 .env 생성"
    cp .env.example .env
    ok ".env 파일 생성 완료"
    echo ""
    echo -e "  ${BOLD}═══ 필수 설정 ═══${NC}"
    echo ""
    echo -e "  .env 파일을 열어서 LLM 프로바이더를 설정하세요:"
    echo ""
    echo -e "  ${BOLD}DashScope (권장):${NC}"
    echo -e "    LLM_PROVIDER=dashscope"
    echo -e "    DASHSCOPE_API_KEY=sk-..."
    echo -e "    발급: ${CYAN}https://bailian.console.alibabacloud.com/${NC}"
    echo ""
    echo -e "  ${BOLD}OpenRouter (대안):${NC}"
    echo -e "    LLM_PROVIDER=openrouter"
    echo -e "    OPENROUTER_API_KEY=sk-or-..."
    echo -e "    발급: ${CYAN}https://openrouter.ai/keys${NC}"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════
# Step 5: Docker MCP 서버 (선택)
# ═══════════════════════════════════════════════════════════════
step 5 "Docker MCP 서버"

if ! command -v docker &>/dev/null; then
    warn "Docker가 설치되어 있지 않습니다"
    echo ""
    echo "  MCP 도구 서버 없이도 CLI는 실행 가능하지만,"
    echo "  코드 생성/파일 읽기 등 도구 기능을 쓰려면 Docker가 필요합니다."
    echo ""
    echo "  Docker 설치 후: make mcp-up"
    echo ""
elif ! docker info &>/dev/null 2>&1; then
    warn "Docker 데몬이 실행 중이 아닙니다"
    echo ""
    echo "  Docker Desktop을 시작한 뒤: make mcp-up"
    echo ""
else
    # Docker가 가용한 경우
    MCP_RUNNING=$(docker ps --filter "name=mcp_" --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$MCP_RUNNING" -gt 0 ]; then
        ok "MCP 서버 ${MCP_RUNNING}개 실행 중"
        docker ps --filter "name=mcp_" --format "  {{.Names}}\t{{.Status}}" 2>/dev/null

        # 이미지가 오래됐는지 확인 — 소스 코드가 이미지보다 새로우면 재빌드 안내
        NEWEST_SRC=$(find mcp_servers/ tests/run_*_mcp.py -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
        OLDEST_IMG=$(docker inspect --format '{{.Created}}' mcp_code_tools_server 2>/dev/null)
        NEEDS_REBUILD=false

        if [ -n "$NEWEST_SRC" ] && [ -n "$OLDEST_IMG" ]; then
            # epoch 비교
            SRC_EPOCH=$(printf '%.0f' "$NEWEST_SRC")
            IMG_EPOCH=$(date -d "$OLDEST_IMG" +%s 2>/dev/null || echo 0)
            if [ "$SRC_EPOCH" -gt "$IMG_EPOCH" ]; then
                NEEDS_REBUILD=true
            fi
        fi

        if [ "$NEEDS_REBUILD" = true ]; then
            echo ""
            warn "MCP 소스 코드가 컨테이너 이미지보다 새롭습니다"
            if ask_yn "이미지를 재빌드할까요?"; then
                info "MCP 서버 재빌드 및 재기동 중..."
                cd docker && docker compose -f docker-compose.mcp.yml up -d --build && cd ..
                ok "MCP 서버 재빌드 완료"
            else
                info "나중에 재빌드: cd docker && docker compose -f docker-compose.mcp.yml up -d --build"
            fi
        fi
    else
        echo ""
        if ask_yn "MCP 도구 서버를 지금 기동할까요?"; then
            info "MCP 서버 빌드 및 기동 중..."
            cd docker && docker compose -f docker-compose.mcp.yml up -d && cd ..
            ok "MCP 서버 기동 완료"
        else
            info "나중에 실행하려면: make mcp-up"
        fi
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
echo "  실행:  youngs75-agent"
echo "  또는:  uv run python -m youngs75_a2a.cli.app"
echo ""
echo "  도움말:  make help"
echo ""
