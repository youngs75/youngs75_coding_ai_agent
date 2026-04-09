#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# youngs75-coding-ai-agent — 하니스 서비스 헬스체크
#
# 사용법:
#   bash scripts/health_check.sh              # 1회 점검
#   bash scripts/health_check.sh --wait 120   # 최대 120초 폴링
#   make health-check                         # Makefile 숏컷
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── 색상 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
info() { echo -e "${CYAN}▸${NC} $*"; }

# ── 서비스 정의 (이름:포트:엔드포인트) ──
SERVICES=(
    "litellm-proxy:4000:/health"
    "mcp-server:3003:/mcp"
    "agent-coding-assistant:18084:/health"
    "orchestrator:18080:/health"
)

# ── 인자 파싱 ──
WAIT_SECS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait)
            WAIT_SECS="${2:-120}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# ── 헬스체크 함수 ──
check_all() {
    local all_ok=true
    local litellm_key="${LITELLM_MASTER_KEY:-sk-harness-local-dev}"

    for entry in "${SERVICES[@]}"; do
        IFS=: read -r name port endpoint <<< "$entry"

        if [ "$name" = "litellm-proxy" ]; then
            # LiteLLM Proxy는 인증 헤더 필요
            http_code=$(curl -so /dev/null -w "%{http_code}" \
                -H "Authorization: Bearer ${litellm_key}" \
                "http://localhost:${port}${endpoint}" 2>/dev/null || echo "000")
        elif [ "$endpoint" = "/mcp" ]; then
            # MCP streamable-http는 Accept 헤더 + JSON-RPC initialize 필요
            http_code=$(curl -so /dev/null -w "%{http_code}" \
                -X POST \
                -H "Content-Type: application/json" \
                -H "Accept: application/json, text/event-stream" \
                -d '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"hc","version":"1"}}}' \
                "http://localhost:${port}${endpoint}" 2>/dev/null || echo "000")
        else
            http_code=$(curl -so /dev/null -w "%{http_code}" \
                "http://localhost:${port}${endpoint}" 2>/dev/null || echo "000")
        fi

        if [[ "$http_code" =~ ^(200|405)$ ]]; then
            ok "${name} (:${port}) — healthy (${http_code})"
        else
            fail "${name} (:${port}) — not responding (${http_code})"
            all_ok=false
        fi
    done
    $all_ok
}

# ── 실행 ──
echo ""
echo -e "${BOLD}  하니스 서비스 헬스체크${NC}"
echo "────────────────────────────────────────"

if [ "$WAIT_SECS" -gt 0 ]; then
    INTERVAL=10
    ELAPSED=0
    while [ "$ELAPSED" -lt "$WAIT_SECS" ]; do
        if check_all 2>/dev/null; then
            echo ""
            echo -e "${GREEN}${BOLD}  모든 서비스 정상!${NC}"
            echo ""
            exit 0
        fi
        ELAPSED=$((ELAPSED + INTERVAL))
        if [ "$ELAPSED" -lt "$WAIT_SECS" ]; then
            info "서비스 기동 대기 중... (${ELAPSED}s/${WAIT_SECS}s)"
            sleep "$INTERVAL"
        fi
    done
    # 최종 리포트
    echo ""
    echo -e "${BOLD}  최종 상태 (${WAIT_SECS}초 후):${NC}"
    echo "────────────────────────────────────────"
    if check_all; then
        echo ""
        echo -e "${GREEN}${BOLD}  모든 서비스 정상!${NC}"
        echo ""
        exit 0
    else
        echo ""
        echo -e "${RED}${BOLD}  일부 서비스가 응답하지 않습니다${NC}"
        echo "  로그 확인: make logs-harness"
        echo ""
        exit 1
    fi
else
    # 1회 점검
    if check_all; then
        echo ""
        echo -e "${GREEN}${BOLD}  모든 서비스 정상!${NC}"
        echo ""
        exit 0
    else
        echo ""
        echo -e "${RED}${BOLD}  일부 서비스가 응답하지 않습니다${NC}"
        echo "  로그 확인: make logs-harness"
        echo ""
        exit 1
    fi
fi
