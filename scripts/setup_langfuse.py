#!/usr/bin/env python3
"""Langfuse 초기 설정 및 헬스체크 스크립트.

Docker Compose로 Langfuse 서버를 기동한 후, 이 스크립트를 실행하여
서버 상태를 확인하고 필요한 환경변수가 올바르게 설정되었는지 검증합니다.

사용법:
    # 1. Langfuse Docker Compose 기동
    docker compose -f docker/docker-compose.langfuse.yaml up -d

    # 2. 헬스체크 및 환경변수 검증
    python scripts/setup_langfuse.py

    # 3. 상세 정보 출력
    python scripts/setup_langfuse.py --verbose

    # 4. 커스텀 호스트 지정
    python scripts/setup_langfuse.py --host http://localhost:3100

종료 코드:
    0 — 모든 검증 통과
    1 — 환경변수 누락 또는 서버 연결 실패
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ── 상수 ──────────────────────────────────────────────────

# Langfuse v3 헬스체크 엔드포인트
_HEALTH_PATH = "/api/public/health"

# 필수 환경변수 목록
_REQUIRED_ENV_VARS = [
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]

# 선택적 환경변수 (경고만 표시)
_OPTIONAL_ENV_VARS = [
    "LANGFUSE_TRACING_ENABLED",
    "LANGFUSE_SAMPLE_RATE",
]

# 기본 호스트
_DEFAULT_HOST = "http://localhost:3100"

# 헬스체크 재시도 설정
_MAX_RETRIES = 30
_RETRY_INTERVAL_SEC = 2


# ── 유틸리티 ──────────────────────────────────────────────


def _print_header(title: str) -> None:
    """섹션 제목을 출력합니다."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_status(label: str, ok: bool, detail: str = "") -> None:
    """상태를 체크/크로스 마크와 함께 출력합니다."""
    mark = "[OK]" if ok else "[FAIL]"
    msg = f"  {mark} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ── 환경변수 검증 ─────────────────────────────────────────


def validate_env_vars(*, verbose: bool = False) -> bool:
    """필수 환경변수가 설정되었는지 확인합니다.

    Args:
        verbose: True이면 각 변수의 값(마스킹)을 출력

    Returns:
        bool: 모든 필수 변수가 설정되었으면 True
    """
    _print_header("환경변수 검증")

    all_ok = True

    # 필수 환경변수 확인
    for var in _REQUIRED_ENV_VARS:
        value = os.environ.get(var, "")
        is_set = bool(value.strip())
        detail = ""
        if verbose and is_set:
            # 비밀 키는 앞 8자만 표시
            if "KEY" in var or "SECRET" in var:
                detail = f"{value[:8]}..."
            else:
                detail = value
        elif not is_set:
            detail = "미설정 — .env 파일 또는 환경변수를 확인하세요"
        _print_status(var, is_set, detail)
        if not is_set:
            all_ok = False

    # 선택적 환경변수 확인 (경고만)
    if verbose:
        print()
        print("  [선택적 환경변수]")
        for var in _OPTIONAL_ENV_VARS:
            value = os.environ.get(var, "")
            is_set = bool(value.strip())
            detail = value if is_set else "(기본값 사용)"
            _print_status(var, True, detail)

    return all_ok


# ── .env 파일 로드 ────────────────────────────────────────


def load_dotenv_if_available() -> None:
    """프로젝트 루트의 .env 파일이 있으면 로드합니다."""
    # scripts/ 디렉토리의 상위 = 프로젝트 루트
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        return

    # dotenv 라이브러리 없이 간단하게 파싱
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 기존 환경변수가 없을 때만 설정 (환경변수 우선)
            if key and key not in os.environ:
                os.environ[key] = value


# ── 헬스체크 ──────────────────────────────────────────────


def check_health(
    host: str,
    *,
    max_retries: int = _MAX_RETRIES,
    retry_interval: float = _RETRY_INTERVAL_SEC,
    verbose: bool = False,
) -> bool:
    """Langfuse 서버의 헬스체크 엔드포인트를 확인합니다.

    Docker 컨테이너가 기동 중일 수 있으므로 재시도 로직을 포함합니다.

    Args:
        host: Langfuse 서버 URL (예: http://localhost:3100)
        max_retries: 최대 재시도 횟수
        retry_interval: 재시도 간격 (초)
        verbose: True이면 각 시도 상태를 출력

    Returns:
        bool: 서버가 정상이면 True
    """
    _print_header("Langfuse 서버 헬스체크")

    url = f"{host.rstrip('/')}{_HEALTH_PATH}"
    print(f"  대상: {url}")
    print(
        f"  최대 대기: {max_retries * retry_interval}초 ({max_retries}회 x {retry_interval}초)"
    )
    print()

    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="replace")

            if status == 200:
                _print_status("헬스체크", True, f"HTTP {status}")
                if verbose:
                    print(f"  응답: {body[:200]}")
                return True
            else:
                if verbose:
                    print(f"  시도 {attempt}/{max_retries}: HTTP {status}")

        except (URLError, OSError, TimeoutError) as e:
            if verbose:
                print(f"  시도 {attempt}/{max_retries}: {e}")

        if attempt < max_retries:
            time.sleep(retry_interval)

    _print_status("헬스체크", False, "서버에 연결할 수 없습니다")
    print()
    print("  해결 방법:")
    print("    1. Docker Compose가 실행 중인지 확인하세요:")
    print("       docker compose -f docker/docker-compose.langfuse.yaml ps")
    print("    2. 서버 로그를 확인하세요:")
    print(
        "       docker compose -f docker/docker-compose.langfuse.yaml logs langfuse-web"
    )
    print("    3. 포트가 올바른지 확인하세요 (기본: 3100)")
    return False


# ── API 연결 검증 ─────────────────────────────────────────


def verify_api_connection(host: str, *, verbose: bool = False) -> bool:
    """Langfuse API 키가 유효한지 간단히 확인합니다.

    /api/public/health 엔드포인트는 인증 없이 접근 가능하므로,
    실제 API 키 검증은 SDK를 통해 수행합니다.

    Args:
        host: Langfuse 서버 URL
        verbose: True이면 상세 정보 출력

    Returns:
        bool: API 연결 성공 여부
    """
    _print_header("Langfuse SDK 연결 검증")

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        _print_status("SDK 연결", False, "API 키가 설정되지 않았습니다")
        return False

    try:
        from langfuse import Langfuse

        lf = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        # auth_check()는 Langfuse SDK v3에서 API 키 유효성을 확인하는 메서드
        lf.auth_check()
        _print_status("SDK 연결", True, "API 키 인증 성공")

        if verbose:
            print(f"  호스트: {host}")
            print(f"  공개 키: {public_key[:12]}...")

        lf.flush()
        return True

    except ImportError:
        _print_status("SDK 연결", False, "langfuse 패키지가 설치되지 않았습니다")
        print("  설치: pip install langfuse")
        return False

    except Exception as e:
        _print_status("SDK 연결", False, str(e))
        print()
        print("  해결 방법:")
        print("    1. Langfuse 웹 대시보드에 로그인하여 프로젝트 API 키를 확인하세요")
        print(f"       {host}")
        print(
            "    2. .env 파일의 LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY를 업데이트하세요"
        )
        return False


# ── 설정 요약 ─────────────────────────────────────────────


def print_setup_summary(host: str) -> None:
    """초기 설정 가이드를 출력합니다."""
    _print_header("Langfuse 초기 설정 가이드")

    print(f"""
  Langfuse 대시보드에 처음 접속하면 초기 설정이 필요합니다:

  1. 브라우저에서 {host} 에 접속
  2. 회원가입 (첫 사용자가 관리자가 됩니다)
  3. 새 프로젝트 생성
  4. 프로젝트 설정 > API Keys 에서 키 발급
  5. 발급받은 키를 .env 파일에 설정:

     LANGFUSE_HOST={host}
     LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
     LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
     LANGFUSE_TRACING_ENABLED=1

  6. 이 스크립트를 다시 실행하여 연결을 확인하세요:
     python scripts/setup_langfuse.py --verbose
""")


# ── Docker Compose 상태 확인 ──────────────────────────────


def check_docker_compose(*, verbose: bool = False) -> bool:
    """Docker Compose 서비스 상태를 확인합니다.

    Returns:
        bool: 모든 서비스가 실행 중이면 True
    """
    import subprocess

    _print_header("Docker Compose 서비스 상태")

    compose_file = (
        Path(__file__).resolve().parent.parent
        / "docker"
        / "docker-compose.langfuse.yaml"
    )

    if not compose_file.exists():
        _print_status("Compose 파일", False, f"{compose_file} 을 찾을 수 없습니다")
        return False

    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "ps", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            _print_status("Docker Compose", False, "docker compose ps 실행 실패")
            if verbose:
                print(f"  stderr: {result.stderr[:200]}")
            return False

        # JSON 출력 파싱 — docker compose v2는 각 줄이 JSON 객체
        import json

        services_running = 0
        services_total = 0
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                svc = json.loads(line)
                name = svc.get("Name", svc.get("name", "unknown"))
                state = svc.get("State", svc.get("state", "unknown"))
                health = svc.get("Health", svc.get("health", ""))
                services_total += 1
                is_running = state.lower() == "running"
                if is_running:
                    services_running += 1
                detail = f"state={state}"
                if health:
                    detail += f", health={health}"
                _print_status(name, is_running, detail)
            except json.JSONDecodeError:
                continue

        if services_total == 0:
            _print_status("서비스", False, "실행 중인 서비스가 없습니다")
            print("  시작: docker compose -f docker/docker-compose.langfuse.yaml up -d")
            return False

        all_running = services_running == services_total
        print()
        print(f"  총 {services_total}개 서비스 중 {services_running}개 실행 중")
        return all_running

    except FileNotFoundError:
        _print_status("Docker", False, "docker 명령어를 찾을 수 없습니다")
        return False
    except subprocess.TimeoutExpired:
        _print_status("Docker Compose", False, "타임아웃")
        return False


# ── 메인 ──────────────────────────────────────────────────


def main() -> int:
    """메인 진입점.

    Returns:
        int: 종료 코드 (0=성공, 1=실패)
    """
    parser = argparse.ArgumentParser(
        description="Langfuse 초기 설정 및 헬스체크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scripts/setup_langfuse.py                    # 기본 검증
  python scripts/setup_langfuse.py --verbose           # 상세 출력
  python scripts/setup_langfuse.py --host http://myhost:3100  # 커스텀 호스트
  python scripts/setup_langfuse.py --wait              # 서버 대기 후 검증
  python scripts/setup_langfuse.py --guide             # 설정 가이드만 출력
        """,
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Langfuse 서버 URL (기본: LANGFUSE_HOST 환경변수 또는 {_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="상세 정보 출력",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="서버가 준비될 때까지 대기 (Docker Compose 기동 직후 사용)",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="초기 설정 가이드만 출력",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Docker Compose 상태 확인 건너뛰기",
    )

    args = parser.parse_args()

    # .env 파일 로드
    load_dotenv_if_available()

    # 호스트 결정: 명령줄 인자 > 환경변수 > 기본값
    host = args.host or os.environ.get("LANGFUSE_HOST", "") or _DEFAULT_HOST

    # 가이드 모드
    if args.guide:
        print_setup_summary(host)
        return 0

    print()
    print("  Langfuse 설정 검증을 시작합니다...")
    print(f"  대상 호스트: {host}")

    results: list[bool] = []

    # 1. 환경변수 검증
    env_ok = validate_env_vars(verbose=args.verbose)
    results.append(env_ok)

    # 2. Docker Compose 상태 확인 (선택적)
    if not args.skip_docker:
        docker_ok = check_docker_compose(verbose=args.verbose)
        results.append(docker_ok)

    # 3. 서버 헬스체크
    retries = _MAX_RETRIES if args.wait else 5
    health_ok = check_health(host, max_retries=retries, verbose=args.verbose)
    results.append(health_ok)

    # 4. SDK 연결 검증 (헬스체크 통과 시에만)
    if health_ok and env_ok:
        api_ok = verify_api_connection(host, verbose=args.verbose)
        results.append(api_ok)
    else:
        if not health_ok:
            print("\n  서버 연결 실패 — SDK 연결 검증을 건너뜁니다.")
        if not env_ok:
            print("\n  환경변수 미설정 — SDK 연결 검증을 건너뜁니다.")

    # 결과 요약
    _print_header("검증 결과 요약")
    all_ok = all(results)
    if all_ok:
        print("  모든 검증을 통과했습니다!")
        print(f"  Langfuse 대시보드: {host}")
    else:
        print("  일부 검증에 실패했습니다.")
        print_setup_summary(host)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
