"""LangGraph 도구 호출 객체에서 name/id/args를 안전하게 추출하는 유틸리티.

LangChain, OpenAI, dict 등 다양한 형태의 tool_call 객체를 처리한다.
메시지 정합성 유틸리티도 포함 — 고아 tool_calls를 정리한다.
"""

from __future__ import annotations

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


def sanitize_messages_for_llm(
    messages: list[Any],
) -> list[Any]:
    """LLM에 전달할 메시지에서 고아 tool_calls를 정리한다.

    DashScope/OpenAI API는 tool_calls가 있는 AI 메시지 뒤에 반드시
    대응하는 ToolMessage가 있어야 한다. 이 함수는:
    1. tool_calls가 있는 AI 메시지에 대응하는 ToolMessage가 없으면
       해당 AI 메시지의 tool_calls를 제거한다.
    2. 메시지 순서와 내용은 최대한 보존한다.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # 1단계: 존재하는 ToolMessage의 tool_call_id를 수집
    answered_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                answered_ids.add(tcid)

    # 2단계: 고아 tool_calls가 있는 AI 메시지를 정리
    cleaned: list[Any] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # 모든 tool_call_id가 answered인지 확인
            orphaned = [
                tc for tc in msg.tool_calls
                if tc_id(tc) not in answered_ids
            ]
            if orphaned:
                # tool_calls를 제거한 깨끗한 메시지로 교체
                clean_msg = AIMessage(
                    content=msg.content or "[도구 호출 생략됨]",
                )
                cleaned.append(clean_msg)
                continue
        cleaned.append(msg)

    return cleaned
