"""LangGraph 도구 호출 객체에서 name/id/args를 안전하게 추출하는 유틸리티.

LangChain, OpenAI, dict 등 다양한 형태의 tool_call 객체를 처리한다.
"""

from typing import Any
import json


def tc_name(tool_call: Any) -> str | None:
    """도구 호출에서 이름을 추출한다."""
    if tool_call is None:
        return None
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        if name:
            return str(name)
        # OpenAI function calling 형태: {"function": {"name": ...}}
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            return fn.get("name")
        return tool_call.get("type")
    for attr in ("name", "tool_name"):
        val = getattr(tool_call, attr, None)
        if val:
            return str(val)
    fn = getattr(tool_call, "function", None)
    if fn:
        return getattr(fn, "name", None)
    return None


def tc_id(tool_call: Any) -> str | None:
    """도구 호출에서 ID를 추출한다."""
    if tool_call is None:
        return None
    if isinstance(tool_call, dict):
        return tool_call.get("id") or tool_call.get("tool_call_id")
    return getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", None)


def tc_args(tool_call: Any) -> dict[str, Any]:
    """도구 호출에서 인자를 추출한다. JSON 문자열도 파싱한다."""
    if tool_call is None:
        return {}
    raw: Any = None
    if isinstance(tool_call, dict):
        raw = tool_call.get("args") or tool_call.get("arguments")
        if raw is None:
            fn = tool_call.get("function")
            if isinstance(fn, dict):
                raw = fn.get("arguments")
    else:
        raw = getattr(tool_call, "args", None) or getattr(tool_call, "arguments", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}
