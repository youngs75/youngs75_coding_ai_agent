"""환경변수 헬퍼."""

import os


def require_env(key: str, message: str | None = None) -> str:
    """필수 환경변수를 반환한다. 없으면 ValueError."""
    value = os.getenv(key)
    if not value:
        raise ValueError(message or f"환경변수 '{key}'가 설정되지 않았습니다.")
    return value


def get_env(key: str, default: str = "") -> str:
    """환경변수를 반환한다. 없으면 기본값."""
    return os.getenv(key, default)
