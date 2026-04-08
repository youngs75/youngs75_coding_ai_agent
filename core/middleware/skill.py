"""스킬 시스템 프롬프트 주입 미들웨어.

활성화된 스킬의 바디를 시스템 프롬프트에 주입한다.
중복 주입을 방지하기 위해 이미 주입된 스킬은 스킵한다.
"""

from __future__ import annotations

import logging
from typing import Any

from youngs75_a2a.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
    append_to_system_message,
)

logger = logging.getLogger(__name__)


class SkillMiddleware(AgentMiddleware):
    """활성 스킬을 시스템 프롬프트에 주입하는 미들웨어."""

    def __init__(self, skill_registry: Any | None = None) -> None:
        self._skill_registry = skill_registry

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        # 레지스트리 없거나 활성 스킬 없으면 패스스루
        if not self._skill_registry:
            return await handler(request)

        active_bodies: list[str] = self._skill_registry.get_active_skill_bodies()
        if not active_bodies:
            return await handler(request)

        # 중복 주입 방지: 이미 시스템 메시지에 포함된 스킬은 스킵
        new_bodies = [
            body for body in active_bodies
            if body not in (request.system_message or "")
        ]

        if not new_bodies:
            return await handler(request)

        # 스킬 본문을 시스템 프롬프트에 추가
        skill_section = "## 활성 스킬\n" + "\n".join(
            f"- {body}" for body in new_bodies
        )
        new_system = append_to_system_message(request.system_message, skill_section)

        logger.debug("[SkillMiddleware] %d개 스킬 주입", len(new_bodies))
        return await handler(request.override(system_message=new_system))
