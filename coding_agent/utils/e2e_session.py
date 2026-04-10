"""E2E 세션 ID 관리.

프로세스 수명 동안 단일 세션 ID를 유지하여
LiteLLM Proxy → Langfuse 트레이싱에서 하나의 E2E 실행을 묶는다.

- 환경변수 ``HARNESS_SESSION_ID``가 있으면 재사용 (Docker/CI 외부 주입용)
- 없으면 자동 생성하여 환경변수에 저장 (프로세스 내 일관성 보장)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime


def get_or_create_session_id() -> str:
    """환경변수에 있으면 재사용, 없으면 생성하여 환경변수에 저장."""
    sid = os.environ.get("HARNESS_SESSION_ID")
    if not sid:
        sid = f"harness-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        os.environ["HARNESS_SESSION_ID"] = sid
    return sid
