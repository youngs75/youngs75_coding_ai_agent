#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# youngs75-agent — AI Coding Agent CLI (샌드박스 모드)
# ═══════════════════════════════════════════════════════════════
#
# 사용법:
#   ./youngs75-agent.sh                # CLI 접속 (Docker 샌드박스)
#   ./youngs75-agent.sh --agent NAME   # 기본 에이전트 지정
#   ./youngs75-agent.sh --export       # 결과물을 현재 폴더로 추출
#
# Docker Harness 안에서 코드를 생성/테스트하고,
# 완료 후 --export 또는 /export로 호스트에 결과물을 가져옵니다.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="harness_agent_coding"

# ── 색상 ──
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── 인자 파싱 ──
MODE="cli"
AGENT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --export)
            MODE="export"
            shift
            ;;
        --clean)
            MODE="clean"
            shift
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        -h|--help)
            echo -e "${BOLD}youngs75-agent${NC} — AI Coding Agent CLI"
            echo ""
            echo "사용법:"
            echo "  ./youngs75-agent.sh              CLI 접속 (Docker 샌드박스)"
            echo "  ./youngs75-agent.sh --agent NAME 기본 에이전트 지정"
            echo "  ./youngs75-agent.sh --export     결과물을 현재 폴더로 추출"
            echo "  ./youngs75-agent.sh --clean      워크스페이스 초기화 (이전 산출물 삭제)"
            echo ""
            echo "에이전트: orchestrator, coding_assistant, deep_research, simple_react"
            echo ""
            echo "CLI 내부 커맨드:"
            echo "  /export    — 결과물 목록 확인 + 추출 준비"
            echo "  /help      — 전체 커맨드 목록"
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            echo "도움말: ./youngs75-agent.sh --help"
            exit 1
            ;;
    esac
done

# ── Docker 상태 확인 ──
is_container_healthy() {
    local status
    status="$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "not_found")"
    [[ "$status" == "healthy" ]]
}

if ! is_container_healthy; then
    echo -e "${RED}오류: Docker 컨테이너($CONTAINER)가 실행 중이 아닙니다.${NC}"
    echo -e "${YELLOW}   'make up-harness'로 Harness를 먼저 기동하세요.${NC}"
    exit 1
fi

# ── Export 모드 ──
if [[ "$MODE" == "export" ]]; then
    DEST="$(pwd)"
    echo -e "${GREEN}📦 Export${NC} — 샌드박스 결과물을 현재 폴더로 추출"
    echo -e "${CYAN}   대상: ${DEST}${NC}"
    echo ""

    # workspace 전체를 tar로 묶어 호스트로 복사
    # .venv, node_modules 등도 포함 (프로젝트 그대로 재현 가능하도록)
    docker exec "$CONTAINER" tar czf /tmp/workspace_export.tar.gz \
        -C /workspace .

    # 호스트로 복사 + 압축 해제
    docker cp "$CONTAINER":/tmp/workspace_export.tar.gz /tmp/workspace_export.tar.gz
    tar xzf /tmp/workspace_export.tar.gz -C "$DEST"
    rm -f /tmp/workspace_export.tar.gz

    # 결과 표시
    FILE_COUNT=$(find "$DEST" -type f | wc -l)
    DIR_COUNT=$(find "$DEST" -type d | wc -l)
    echo -e "${GREEN}✓ 추출 완료${NC} (파일 ${FILE_COUNT}개, 디렉토리 ${DIR_COUNT}개)"
    echo ""
    # 주요 파일만 표시 (.venv, node_modules 등 상세 생략)
    find "$DEST" -type f -not -path '*/.venv/*' -not -path '*/node_modules/*' \
        -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' \
        -not -name '*.pyc' | sort | while read -r f; do
        rel="${f#$DEST/}"
        echo -e "  ${DIM}${rel}${NC}"
    done
    echo ""
    echo -e "${CYAN}📂 ${DEST}${NC}"
    echo ""
    echo -e "${YELLOW}💡 다음 작업 전 워크스페이스를 초기화하려면:${NC}"
    echo -e "   ./youngs75-agent.sh --clean"
    exit 0
fi

# ── Clean 모드 ──
if [[ "$MODE" == "clean" ]]; then
    echo -e "${YELLOW}🧹 Workspace 초기화${NC} — 이전 산출물을 삭제합니다"
    CONTAINER="harness_mcp_server"
    docker exec "$CONTAINER" sh -c 'find /workspace -mindepth 1 -not -name ".gitkeep" -exec rm -rf {} + 2>/dev/null; echo done' || true
    echo -e "${GREEN}✓ /workspace 초기화 완료${NC}"
    exit 0
fi

# ── CLI 모드 ──
echo -e "${GREEN}🔒 샌드박스 모드${NC} — Docker 컨테이너에서 실행"
echo -e "${CYAN}   코드 생성/테스트는 격리된 환경에서 수행됩니다${NC}"
echo -e "${DIM}   완료 후: ./youngs75-agent.sh --export 또는 CLI에서 /export${NC}"
echo ""

env_args=()
if [[ -n "$AGENT" ]]; then
    env_args+=(-e "CLI_DEFAULT_AGENT=$AGENT")
fi

docker exec -it "${env_args[@]}" "$CONTAINER" \
    python -m coding_agent.cli
