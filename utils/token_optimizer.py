"""프롬프트 토큰 최적화 모듈.

tiktoken 기반 토큰 수 측정, 프롬프트 압축, 토큰 예산 관리를 제공한다.
기존 에이전트 코드를 변경하지 않는 opt-in 유틸리티.

사용 예시:
    from youngs75_a2a.utils.token_optimizer import (
        count_tokens,
        compress_prompt,
        TokenBudget,
        report_prompt_tokens,
    )

    # 토큰 수 측정
    n = count_tokens("프롬프트 텍스트", model="deepseek/deepseek-v3.2")

    # 프롬프트 압축
    compressed = compress_prompt(long_prompt, max_tokens=2000)

    # 토큰 예산 관리
    budget = TokenBudget(parse=500, execute=4000, verify=1500)
    budget.check("parse", prompt_text)

    # 현재 프롬프트 토큰 리포트
    report = report_prompt_tokens()
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field

import tiktoken

logger = logging.getLogger(__name__)

# ── 모델-인코딩 매핑 캐시 ──

_encoding_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str = "deepseek/deepseek-v3.2") -> tiktoken.Encoding:
    """모델에 맞는 tiktoken 인코딩을 반환한다 (캐시 적용).

    알 수 없는 모델은 cl100k_base로 폴백한다.
    """
    if model in _encoding_cache:
        return _encoding_cache[model]

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # 알 수 없는 모델 → cl100k_base (GPT-4 계열 기본)
        enc = tiktoken.get_encoding("cl100k_base")

    _encoding_cache[model] = enc
    return enc


# ── 토큰 수 측정 ──


def count_tokens(text: str, *, model: str = "deepseek/deepseek-v3.2") -> int:
    """텍스트의 토큰 수를 반환한다.

    Args:
        text: 측정할 텍스트
        model: 토큰화에 사용할 모델 이름

    Returns:
        토큰 수
    """
    enc = _get_encoding(model)
    return len(enc.encode(text))


def count_messages_tokens(
    messages: list[dict[str, str]],
    *,
    model: str = "deepseek/deepseek-v3.2",
) -> int:
    """ChatML 메시지 리스트의 토큰 수를 추정한다.

    각 메시지는 {"role": ..., "content": ...} 형식.
    OpenAI의 메시지 오버헤드(role 토큰 등)를 포함하여 추정한다.

    Args:
        messages: ChatML 메시지 리스트
        model: 모델 이름

    Returns:
        추정 토큰 수
    """
    enc = _get_encoding(model)
    # 메시지당 오버헤드: <|im_start|>{role}\n ... <|im_end|>\n → 약 4토큰
    tokens_per_message = 4
    total = 0
    for msg in messages:
        total += tokens_per_message
        for value in msg.values():
            total += len(enc.encode(str(value)))
    # 전체 응답 프라이밍 토큰
    total += 2
    return total


# ── 프롬프트 압축 ──

# 연속 공백/빈 줄 패턴
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def compress_prompt(
    text: str,
    *,
    max_tokens: int | None = None,
    model: str = "deepseek/deepseek-v3.2",
    strip_comments: bool = False,
) -> str:
    """프롬프트를 압축한다.

    수행하는 최적화:
    1. 선행/후행 공백 제거 (dedent)
    2. 연속 빈 줄 → 단일 빈 줄
    3. 연속 스페이스/탭 → 단일 스페이스 (코드 블록 내부 제외)
    4. (선택) 주석 행 제거
    5. max_tokens 초과 시 뒤에서부터 잘라냄

    Args:
        text: 원본 프롬프트
        max_tokens: 최대 허용 토큰 수 (None이면 제한 없음)
        model: 토큰 카운팅에 사용할 모델
        strip_comments: True면 '#' 주석 행 제거

    Returns:
        압축된 프롬프트
    """
    # 1. dedent
    result = textwrap.dedent(text).strip()

    # 2. 연속 빈 줄 → 단일 빈 줄
    result = _MULTI_BLANK_RE.sub("\n\n", result)

    # 3. 코드 블록 밖의 연속 공백 축소
    lines = result.split("\n")
    in_code_block = False
    compressed_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            compressed_lines.append(line)
            continue

        if in_code_block:
            # 코드 블록 내부는 원본 유지
            compressed_lines.append(line)
        else:
            # 코드 블록 밖: 연속 공백 축소
            compressed = _MULTI_SPACE_RE.sub(" ", line).rstrip()
            if strip_comments and compressed.lstrip().startswith("#"):
                continue
            compressed_lines.append(compressed)

    result = "\n".join(compressed_lines)

    # 4. max_tokens 제한
    if max_tokens is not None:
        enc = _get_encoding(model)
        tokens = enc.encode(result)
        if len(tokens) > max_tokens:
            # 뒤에서부터 잘라내고 말줄임 표시 추가
            truncated_tokens = tokens[: max_tokens - 5]  # "...(truncated)" 여유
            result = enc.decode(truncated_tokens) + "\n...(truncated)"
            logger.warning(
                "프롬프트가 %d 토큰으로 잘렸습니다 (원본: %d 토큰)",
                max_tokens,
                len(tokens),
            )

    return result


# ── 토큰 예산 관리 ──


@dataclass
class TokenBudget:
    """노드별 토큰 예산을 관리한다.

    예산 초과 시 경고 로그를 남기고, 초과 여부를 반환한다.

    사용 예시:
        budget = TokenBudget(parse=500, execute=4000, verify=1500)
        is_over, info = budget.check("parse", prompt_text)
    """

    parse: int = 500
    execute: int = 4000
    verify: int = 1500
    default: int = 2000
    model: str = "deepseek/deepseek-v3.2"

    # 누적 사용량 추적
    _usage: dict[str, int] = field(default_factory=dict, repr=False)

    def get_budget(self, purpose: str) -> int:
        """목적별 토큰 예산을 반환한다."""
        return getattr(self, purpose, self.default)

    def check(self, purpose: str, text: str) -> tuple[bool, dict[str, int]]:
        """텍스트가 목적별 토큰 예산을 초과하는지 확인한다.

        Args:
            purpose: 노드 목적 (parse, execute, verify)
            text: 프롬프트 텍스트

        Returns:
            (초과 여부, {"tokens": 실제 토큰 수, "budget": 예산, "over_by": 초과량})
        """
        tokens = count_tokens(text, model=self.model)
        budget = self.get_budget(purpose)
        over_by = max(0, tokens - budget)
        is_over = over_by > 0

        # 누적 사용량 기록
        self._usage[purpose] = self._usage.get(purpose, 0) + tokens

        info = {"tokens": tokens, "budget": budget, "over_by": over_by}

        if is_over:
            logger.warning(
                "[TokenBudget] '%s' 예산 초과: %d/%d 토큰 (+%d)",
                purpose,
                tokens,
                budget,
                over_by,
            )

        return is_over, info

    @property
    def usage(self) -> dict[str, int]:
        """목적별 누적 토큰 사용량."""
        return dict(self._usage)

    def reset(self) -> None:
        """누적 사용량을 초기화한다."""
        self._usage.clear()


# ── 프롬프트 토큰 리포트 ──


def report_prompt_tokens(
    *,
    model: str = "deepseek/deepseek-v3.2",
) -> dict[str, dict[str, int | str]]:
    """현재 등록된 시스템 프롬프트들의 토큰 수를 측정하여 리포트한다.

    Returns:
        {"parse": {"tokens": N, "prompt_preview": "..."}, ...}
    """
    from youngs75_a2a.agents.coding_assistant.prompts import (
        EXECUTE_SYSTEM_PROMPT,
        PARSE_SYSTEM_PROMPT,
        VERIFY_SYSTEM_PROMPT,
    )

    prompts = {
        "parse": PARSE_SYSTEM_PROMPT,
        "execute": EXECUTE_SYSTEM_PROMPT,
        "verify": VERIFY_SYSTEM_PROMPT,
    }

    report: dict[str, dict[str, int | str]] = {}
    total = 0

    for name, text in prompts.items():
        tokens = count_tokens(text, model=model)
        total += tokens
        # 프롬프트 미리보기 (앞 80자)
        preview = text[:80].replace("\n", " ").strip()
        report[name] = {
            "tokens": tokens,
            "prompt_preview": preview + ("..." if len(text) > 80 else ""),
        }

    report["_total"] = {"tokens": total, "prompt_preview": "전체 시스템 프롬프트 합계"}

    return report


def report_prompt_tokens_text(*, model: str = "deepseek/deepseek-v3.2") -> str:
    """프롬프트 토큰 리포트를 사람이 읽기 쉬운 텍스트로 반환한다."""
    report = report_prompt_tokens(model=model)
    lines = ["=== 프롬프트 토큰 리포트 ===", f"모델: {model}", ""]

    for name, info in report.items():
        if name == "_total":
            continue
        lines.append(
            f"  {name:10s}: {info['tokens']:>6,} 토큰  | {info['prompt_preview']}"
        )

    total_info = report.get("_total", {})
    lines.append("")
    lines.append(f"  {'합계':10s}: {total_info.get('tokens', 0):>6,} 토큰")
    lines.append("=" * 40)

    return "\n".join(lines)
