"""LLM 응답에서 JSON 객체를 추출하는 공통 유틸리티.

LLM은 JSON을 반환할 때 종종 markdown code fence(```json ... ```)로
감싸거나, 앞뒤에 설명 텍스트를 추가합니다. 이 모듈은 다양한 형식의
LLM 응답에서 JSON 객체를 안전하게 추출합니다.

사용 예시:
    from coding_agent.eval_pipeline.llm.json_utils import extract_json_object

    raw = '```json\\n{"score": 0.85, "reason": "good"}\\n```'
    parsed = extract_json_object(raw)
    # → {"score": 0.85, "reason": "good"}
"""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(raw: str) -> dict[str, Any]:
    """LLM 응답 텍스트에서 JSON 객체(dict)를 추출합니다.

    추출 전략 (우선순위 순):
    1. markdown code fence (```json ... ```) 내부의 JSON 파싱
    2. 전체 텍스트를 JSON으로 직접 파싱
    3. 첫 번째 '{' ~ 마지막 '}' 범위를 추출하여 파싱

    Args:
        raw: LLM이 반환한 원문 텍스트

    Returns:
        파싱된 JSON 딕셔너리

    Raises:
        ValueError: JSON 객체를 찾을 수 없거나 파싱에 실패한 경우
    """
    text = raw.strip()
    if text.startswith("```"):
        blocks = text.split("```")
        if len(blocks) >= 3:
            candidate = blocks[1].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            text = candidate

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON object not found in model output")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Top-level JSON value must be an object")
    return parsed
