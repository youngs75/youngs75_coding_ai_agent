"""내장 훅 — 기본 제공 훅 핸들러 모음.

로깅, 타이밍, 감사(audit) 등 일반적으로 유용한 훅을 제공한다.

사용 예시:
    from youngs75_a2a.core.hooks import HookManager, HookEvent
    from youngs75_a2a.core.builtin_hooks import logging_hook, timing_hook

    manager = HookManager()
    manager.register(HookEvent.PRE_TOOL_CALL, logging_hook)
    manager.register(HookEvent.PRE_TOOL_CALL, timing_hook)
    manager.register(HookEvent.POST_TOOL_CALL, timing_hook)
"""

from __future__ import annotations

import logging
import time

from youngs75_a2a.core.hooks import HookContext, HookEvent

logger = logging.getLogger(__name__)

# 감사 대상 민감 도구 목록
SENSITIVE_TOOLS: set[str] = {
    "bash",
    "execute_python",
    "delete_file",
    "apply_patch",
    "write_file",
    "str_replace",
}


async def logging_hook(ctx: HookContext) -> HookContext:
    """로깅 훅: 모든 도구 호출/결과를 로깅한다.

    PRE 이벤트에서는 도구 이름과 인자를, POST 이벤트에서는 결과 요약을 기록한다.
    """
    if ctx.event == HookEvent.PRE_TOOL_CALL:
        logger.info(
            "[Hook:logging] PRE_TOOL_CALL tool=%s args=%s",
            ctx.tool_name,
            _truncate(str(ctx.tool_args), 200),
        )
    elif ctx.event == HookEvent.POST_TOOL_CALL:
        result_summary = _truncate(str(ctx.tool_result), 200)
        logger.info(
            "[Hook:logging] POST_TOOL_CALL tool=%s result=%s",
            ctx.tool_name,
            result_summary,
        )
    elif ctx.event == HookEvent.PRE_NODE:
        logger.info("[Hook:logging] PRE_NODE node=%s", ctx.node_name)
    elif ctx.event == HookEvent.POST_NODE:
        logger.info("[Hook:logging] POST_NODE node=%s", ctx.node_name)
    elif ctx.event == HookEvent.ON_ERROR:
        logger.error(
            "[Hook:logging] ON_ERROR tool=%s node=%s error=%s",
            ctx.tool_name,
            ctx.node_name,
            ctx.error,
        )
    elif ctx.event == HookEvent.PRE_LLM_CALL:
        logger.info("[Hook:logging] PRE_LLM_CALL")
    elif ctx.event == HookEvent.POST_LLM_CALL:
        logger.info("[Hook:logging] POST_LLM_CALL")

    return ctx


async def timing_hook(ctx: HookContext) -> HookContext:
    """타이밍 훅: 실행 시간을 측정한다.

    PRE 이벤트에서 시작 시간을 기록하고, POST 이벤트에서 경과 시간을 계산한다.
    측정 결과는 ctx.metadata["duration_s"]에 저장된다.
    """
    if ctx.event in (
        HookEvent.PRE_TOOL_CALL,
        HookEvent.PRE_NODE,
        HookEvent.PRE_LLM_CALL,
    ):
        ctx.metadata["start_time"] = time.monotonic()
    elif ctx.event in (
        HookEvent.POST_TOOL_CALL,
        HookEvent.POST_NODE,
        HookEvent.POST_LLM_CALL,
    ):
        start = ctx.metadata.get("start_time")
        if start is not None:
            elapsed = time.monotonic() - start
            ctx.metadata["duration_s"] = elapsed
            logger.info(
                "[Hook:timing] %s tool=%s node=%s duration=%.3fs",
                ctx.event.value,
                ctx.tool_name,
                ctx.node_name,
                elapsed,
            )

    return ctx


async def audit_hook(ctx: HookContext) -> HookContext:
    """감사 훅: 민감 도구 사용을 기록한다.

    SENSITIVE_TOOLS에 포함된 도구의 호출과 결과를 WARNING 레벨로 로깅한다.
    감사 기록은 ctx.metadata["audit_logged"]로 표시된다.
    """
    if ctx.tool_name and ctx.tool_name in SENSITIVE_TOOLS:
        if ctx.event == HookEvent.PRE_TOOL_CALL:
            logger.warning(
                "[Hook:audit] 민감 도구 호출: tool=%s args=%s",
                ctx.tool_name,
                _truncate(str(ctx.tool_args), 300),
            )
            ctx.metadata["audit_logged"] = True
        elif ctx.event == HookEvent.POST_TOOL_CALL:
            logger.warning(
                "[Hook:audit] 민감 도구 완료: tool=%s result=%s",
                ctx.tool_name,
                _truncate(str(ctx.tool_result), 300),
            )
            ctx.metadata["audit_logged"] = True

    return ctx


def _truncate(text: str, max_len: int = 200) -> str:
    """문자열을 최대 길이로 잘라낸다."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
