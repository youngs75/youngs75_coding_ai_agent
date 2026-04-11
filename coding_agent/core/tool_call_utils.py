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


def _try_parse_json_args(raw: str) -> dict[str, Any]:
    """JSON 문자열을 파싱하되, 실패 시 복구를 시도한다.

    Qwen3 등 일부 모델이 이중 닫기 중괄호(}})나 불필요한 후행 문자를
    생성하는 문제를 방어한다.
    """
    # 1차: 원본 파싱
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass

    # 2차: 후행 중괄호/공백 제거 후 재시도
    stripped = raw.rstrip()
    while stripped.endswith("}}"):
        stripped = stripped[:-1]
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            continue

    # 3차: 첫 번째 '{' ~ 마지막 '}'만 추출하여 재시도
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


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
        return _try_parse_json_args(raw)
    return {}


def sanitize_messages_for_llm(
    messages: list[Any],
) -> list[Any]:
    """LLM에 전달할 메시지에서 양방향 고아 페어링을 정리한다.

    DashScope/OpenAI API는 AIMessage.tool_calls와 ToolMessage가 반드시
    1:1 대응해야 한다. 이 함수는 양방향으로 정리한다:

    1. 대응 ToolMessage가 없는 AIMessage의 tool_calls → 해당 call만 부분 제거
       (기존: 전체 tool_calls 제거 → 캐스케이딩 고아 발생)
    2. 대응 AIMessage가 없는 ToolMessage → 제거
    """
    from langchain_core.messages import AIMessage, ToolMessage

    # 1단계: 양방향 ID 수집
    ai_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                cid = tc_id(tc)
                if cid:
                    ai_call_ids.add(cid)
        elif isinstance(msg, ToolMessage):
            tcid = getattr(msg, "tool_call_id", None)
            if tcid:
                tool_result_ids.add(tcid)

    # 2단계: 정리
    cleaned: list[Any] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            # 대응 ToolMessage가 있는 call만 보존 (부분 제거)
            valid_calls = [
                tc for tc in msg.tool_calls
                if tc_id(tc) in tool_result_ids
            ]
            if len(valid_calls) == len(msg.tool_calls):
                # 모든 call이 유효 → 원본 유지
                cleaned.append(msg)
            elif valid_calls:
                # 일부만 유효 → 유효한 call만 포함하는 새 메시지
                # additional_kwargs.tool_calls도 동기화 (DashScope 호환)
                valid_ids = {tc_id(tc) for tc in valid_calls}
                ak = dict(getattr(msg, "additional_kwargs", {}) or {})
                ak_tcs = ak.get("tool_calls", [])
                if ak_tcs:
                    ak["tool_calls"] = [
                        tc for tc in ak_tcs
                        if tc.get("id", "") in valid_ids
                    ]
                cleaned.append(AIMessage(
                    content=msg.content or "",
                    tool_calls=valid_calls,
                    additional_kwargs=ak,
                    id=getattr(msg, "id", None),
                ))
            else:
                # 모든 call이 고아 → tool_calls 제거
                cleaned.append(AIMessage(
                    content=msg.content or "[도구 호출 생략됨]",
                ))
            continue

        if isinstance(msg, ToolMessage):
            # 대응 AIMessage가 없는 ToolMessage 제거
            tcid = getattr(msg, "tool_call_id", None)
            if tcid and tcid not in ai_call_ids:
                continue  # 고아 ToolMessage 제거

        cleaned.append(msg)

    return cleaned


def ensure_tool_calls_serializable(messages: list[Any]) -> list[Any]:
    """AIMessage의 tool_calls가 DashScope/OpenAI 호환 형식인지 보장한다.

    LangChain 내부 형식(tool_calls[].args: dict)은 DashScope가 직렬화하지 못한다.
    additional_kwargs.tool_calls가 없으면 생성하여 arguments를 JSON 문자열로 보장한다.

    "Can only get item pairs from a mapping" 에러 방지용.
    """
    from langchain_core.messages import AIMessage

    result: list[Any] = []
    for msg in messages:
        if not isinstance(msg, AIMessage) or not getattr(msg, "tool_calls", None):
            result.append(msg)
            continue

        ak = getattr(msg, "additional_kwargs", {}) or {}
        ak_tcs = ak.get("tool_calls", [])

        if ak_tcs:
            # additional_kwargs.tool_calls가 있으면 arguments를 JSON 문자열로 보장
            fixed = False
            for tc in ak_tcs:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args, ensure_ascii=False)
                    fixed = True
            if fixed:
                new_ak = dict(ak)
                new_ak["tool_calls"] = ak_tcs
                result.append(AIMessage(
                    content=msg.content or "",
                    tool_calls=msg.tool_calls,
                    additional_kwargs=new_ak,
                    id=getattr(msg, "id", None),
                ))
            else:
                result.append(msg)
        else:
            # additional_kwargs.tool_calls가 없으면 tool_calls에서 생성
            new_ak_tcs = []
            for tc in msg.tool_calls:
                tc_id_val = tc.get("id", "")
                tc_name_val = tc.get("name", "")
                tc_args = tc.get("args", {})
                args_str = json.dumps(tc_args, ensure_ascii=False) if isinstance(tc_args, dict) else str(tc_args)
                new_ak_tcs.append({
                    "id": tc_id_val,
                    "type": "function",
                    "function": {
                        "name": tc_name_val,
                        "arguments": args_str,
                    },
                })
            new_ak = dict(ak)
            new_ak["tool_calls"] = new_ak_tcs
            result.append(AIMessage(
                content=msg.content or "",
                tool_calls=msg.tool_calls,
                additional_kwargs=new_ak,
                id=getattr(msg, "id", None),
            ))

    return result
