"""coding_agent 전체 테스트 공통 설정."""

from __future__ import annotations

import os

# 테스트 환경에서는 Langfuse 트레이싱 비활성화
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "0")


def pytest_configure(config):
    """pytest 커스텀 마커 등록."""
    config.addinivalue_line("markers", "flaky: mark test as flaky with reruns")
