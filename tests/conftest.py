"""youngs75_a2a 전체 테스트 공통 설정."""

from __future__ import annotations

import os

# 테스트 환경에서는 Langfuse 트레이싱 비활성화
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "0")
