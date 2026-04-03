"""커스텀 LangGraph 상태 리듀서."""

from typing import Any


def override_reducer(current_value: Any, new_value: Any) -> Any:
    """상태 값을 덮어쓰기 또는 누적하는 리듀서.

    new_value가 {"type": "override", "value": ...} 형태이면
    기존 값을 완전히 대체한다. 그 외에는 리스트 병합.

    사용 예:
        notes: Annotated[list[str], override_reducer]

        # 누적: return {"notes": ["새 노트"]}
        # 덮어쓰기: return {"notes": {"type": "override", "value": []}}
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value")
    if isinstance(current_value, list) and isinstance(new_value, list):
        return current_value + new_value
    return new_value
