"""컨텍스트 윈도우 관리 모듈.

Claude Code의 자동 컴팩션 패턴을 LangGraph에 맞게 구현한다.
토큰 사용량 추적, 임계치 초과 시 자동 컴팩션, 서브에이전트용 히스토리 필터링을 제공한다.

사용 예시:
    from youngs75_a2a.core.context_manager import ContextManager

    ctx = ContextManager(max_tokens=128000, compact_threshold=0.8)

    # 컴팩션 필요 여부 판단
    if ctx.should_compact(messages):
        messages = await ctx.compact(messages, llm)

    # 서브에이전트용 히스토리 필터링
    sub_messages = ctx.truncate_for_subagent(messages, last_n_turns=3)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from youngs75_a2a.utils.token_optimizer import count_messages_tokens

logger = logging.getLogger(__name__)

# 컴팩션 요약에 사용하는 프롬프트
_COMPACT_SUMMARY_PROMPT = """\
아래는 이전 대화 내용입니다. 핵심 정보를 간결하게 요약해 주세요.
유지해야 할 정보:
- 사용자의 원래 요청과 의도
- 중요한 결정 사항과 그 근거
- 생성/수정된 코드의 핵심 내용
- 현재 진행 상태

불필요한 정보는 제거하세요:
- 도구 호출의 세부 입출력
- 중간 시행착오 과정
- 반복되는 맥락 설명
"""

# max_tokens 복구에 사용하는 프롬프트
_MAX_TOKENS_RECOVERY_PROMPT = (
    "이전 응답이 토큰 제한으로 중단되었습니다. "
    "중단된 부분부터 이어서 완성해 주세요. 이미 생성된 내용을 반복하지 마세요."
)


class ContextManager:
    """컨텍스트 윈도우 관리자.

    토큰 사용량을 추적하고, 임계치 초과 시 자동 컴팩션을 수행한다.

    Args:
        max_tokens: 모델의 최대 컨텍스트 윈도우 크기
        compact_threshold: 컴팩션 트리거 비율 (0.0~1.0)
        keep_recent_turns: 컴팩션 시 유지할 최근 턴 수
        model: 토큰 카운팅에 사용할 모델 이름
    """

    def __init__(
        self,
        max_tokens: int = 128000,
        compact_threshold: float = 0.8,
        keep_recent_turns: int = 4,
        model: str = "deepseek/deepseek-v3.2",
    ) -> None:
        if not 0.0 < compact_threshold <= 1.0:
            raise ValueError(
                f"compact_threshold는 0.0~1.0 사이여야 합니다: {compact_threshold}"
            )
        if max_tokens <= 0:
            raise ValueError(f"max_tokens는 양수여야 합니다: {max_tokens}")
        if keep_recent_turns < 0:
            raise ValueError(
                f"keep_recent_turns는 0 이상이어야 합니다: {keep_recent_turns}"
            )

        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.keep_recent_turns = keep_recent_turns
        self.model = model

    # ── 토큰 카운팅 ──────────────────────────────────────────

    def count_messages_tokens(self, messages: list[BaseMessage]) -> int:
        """메시지 리스트의 총 토큰 수를 계산한다.

        langchain BaseMessage 리스트를 dict 형태로 변환한 뒤
        utils/token_optimizer.py의 count_messages_tokens를 활용한다.

        Args:
            messages: BaseMessage 리스트

        Returns:
            추정 토큰 수
        """
        msg_dicts = _messages_to_dicts(messages)
        return count_messages_tokens(msg_dicts, model=self.model)

    # ── 컴팩션 판단 ──────────────────────────────────────────

    def should_compact(self, messages: list[BaseMessage]) -> bool:
        """컴팩션이 필요한지 판단한다.

        토큰 사용량이 max_tokens * compact_threshold를 초과하면 True.

        Args:
            messages: 현재 메시지 리스트

        Returns:
            컴팩션 필요 여부
        """
        current_tokens = self.count_messages_tokens(messages)
        threshold_tokens = int(self.max_tokens * self.compact_threshold)
        should = current_tokens > threshold_tokens
        if should:
            logger.info(
                "[ContextManager] 컴팩션 필요: %d / %d 토큰 (임계치: %d)",
                current_tokens,
                self.max_tokens,
                threshold_tokens,
            )
        return should

    # ── 컴팩션 실행 ──────────────────────────────────────────

    async def compact(
        self,
        messages: list[BaseMessage],
        llm: Any,
    ) -> list[BaseMessage]:
        """메시지 히스토리를 압축한다.

        전략:
        1. 시스템 메시지는 항상 유지
        2. 최근 N턴(기본 4)은 유지
        3. 오래된 턴은 LLM으로 요약하여 단일 SystemMessage로 교체

        Args:
            messages: 현재 메시지 리스트
            llm: 요약에 사용할 LLM (BaseChatModel)

        Returns:
            압축된 메시지 리스트
        """
        # 시스템 메시지와 비-시스템 메시지 분리
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # 메시지가 충분히 적으면 컴팩션 불필요
        if len(non_system_msgs) <= self.keep_recent_turns:
            logger.debug("[ContextManager] 메시지가 충분히 적어 컴팩션 생략")
            return messages

        # 최근 N턴과 오래된 메시지 분리
        recent_msgs = non_system_msgs[-self.keep_recent_turns :]
        old_msgs = non_system_msgs[: -self.keep_recent_turns]

        # 오래된 메시지를 LLM으로 요약
        summary = await self._summarize_messages(old_msgs, llm)

        # 요약을 SystemMessage로 변환
        summary_msg = SystemMessage(content=f"[이전 대화 요약]\n{summary}")

        # 시스템 메시지 + 요약 + 최근 메시지로 재구성
        compacted = system_msgs + [summary_msg] + recent_msgs

        before_tokens = self.count_messages_tokens(messages)
        after_tokens = self.count_messages_tokens(compacted)
        logger.info(
            "[ContextManager] 컴팩션 완료: %d → %d 토큰 (%.1f%% 절감)",
            before_tokens,
            after_tokens,
            (1 - after_tokens / before_tokens) * 100 if before_tokens > 0 else 0,
        )

        return compacted

    async def _summarize_messages(
        self,
        messages: list[BaseMessage],
        llm: Any,
    ) -> str:
        """메시지 리스트를 LLM으로 요약한다."""
        # 요약할 내용을 텍스트로 변환
        conversation_text = "\n".join(
            f"[{_get_role(m)}]: {_get_content(m)}" for m in messages
        )

        summary_request = [
            SystemMessage(content=_COMPACT_SUMMARY_PROMPT),
            HumanMessage(content=conversation_text),
        ]

        response = await llm.ainvoke(summary_request)
        return _get_content(response)

    # ── 서브에이전트용 히스토리 필터링 ───────────────────────

    def truncate_for_subagent(
        self,
        messages: list[BaseMessage],
        last_n_turns: int = 3,
    ) -> list[BaseMessage]:
        """서브에이전트용으로 히스토리를 필터링한다.

        Codex의 LastNTurns 패턴:
        - 시스템/개발자 메시지: 항상 포함
        - 사용자(Human) 메시지: 항상 포함
        - assistant(AI) 메시지: tool_call이 없는 최종 응답만 포함
        - 최근 N턴만 유지

        Args:
            messages: 전체 메시지 리스트
            last_n_turns: 유지할 최근 턴 수

        Returns:
            필터링된 메시지 리스트
        """
        # 시스템 메시지는 항상 포함
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # tool_call이 있는 AI 메시지와 ToolMessage를 제외
        filtered: list[BaseMessage] = []
        for msg in non_system_msgs:
            if isinstance(msg, AIMessage):
                # tool_calls가 있는 메시지는 제외
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    continue
            # ToolMessage 제외 (role이 "tool"인 메시지)
            if _get_role(msg) == "tool":
                continue
            filtered.append(msg)

        # 턴 단위로 자르기: HumanMessage를 턴 시작으로 간주
        turns: list[list[BaseMessage]] = []
        current_turn: list[BaseMessage] = []
        for msg in filtered:
            if isinstance(msg, HumanMessage):
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)
        if current_turn:
            turns.append(current_turn)

        # 최근 N턴만 유지
        recent_turns = turns[-last_n_turns:] if last_n_turns > 0 else []

        # 시스템 메시지 + 최근 턴 결합
        result = system_msgs.copy()
        for turn in recent_turns:
            result.extend(turn)

        return result


# ── max_tokens 복구 유틸리티 ─────────────────────────────────


async def invoke_with_max_tokens_recovery(
    llm: Any,
    messages: list[BaseMessage],
    context_manager: ContextManager,
    *,
    max_retries: int = 3,
) -> BaseMessage:
    """LLM 호출 시 max_tokens 중단을 감지하고 자동 복구한다.

    stop_reason이 "max_tokens" (또는 "length")일 때:
    1. 컨텍스트 컴팩션 수행
    2. 이어서 생성하도록 재요청
    3. 최대 max_retries회 재시도

    Args:
        llm: BaseChatModel 인스턴스
        messages: 입력 메시지 리스트
        context_manager: ContextManager 인스턴스
        max_retries: 최대 재시도 횟수

    Returns:
        최종 AI 응답 메시지
    """
    current_messages = list(messages)

    for attempt in range(max_retries + 1):
        response = await llm.ainvoke(current_messages)

        # stop_reason 확인
        stop_reason = _extract_stop_reason(response)
        if stop_reason not in ("max_tokens", "length"):
            # 정상 완료
            return response

        logger.warning(
            "[max_tokens 복구] 토큰 제한 중단 감지 (시도 %d/%d)",
            attempt + 1,
            max_retries + 1,
        )

        if attempt >= max_retries:
            # 최대 재시도 도달 — 불완전하더라도 반환
            logger.error(
                "[max_tokens 복구] 최대 재시도 횟수(%d) 도달. 불완전한 응답 반환.",
                max_retries,
            )
            return response

        # 컴팩션 수행
        current_messages = await context_manager.compact(current_messages, llm)

        # 이전 불완전한 응답을 포함시키고 이어서 생성 요청
        partial_content = _get_content(response)
        current_messages.append(AIMessage(content=partial_content))
        current_messages.append(HumanMessage(content=_MAX_TOKENS_RECOVERY_PROMPT))

    # 루프가 끝난 경우 (이론상 도달하지 않음)
    return response  # type: ignore[possibly-undefined]


# ── 내부 헬퍼 함수 ───────────────────────────────────────────


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """BaseMessage 리스트를 dict 리스트로 변환한다."""
    return [{"role": _get_role(m), "content": _get_content(m)} for m in messages]


def _get_role(msg: BaseMessage) -> str:
    """메시지의 역할을 문자열로 반환한다."""
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    # ToolMessage 등
    return getattr(msg, "type", "unknown")


def _get_content(msg: BaseMessage) -> str:
    """메시지의 content를 문자열로 반환한다."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 멀티모달 content 처리
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return " ".join(parts)
    return str(content)


def _extract_stop_reason(response: BaseMessage) -> str | None:
    """AI 응답에서 stop_reason을 추출한다.

    LangChain의 AIMessage는 response_metadata에 finish_reason / stop_reason을 저장한다.
    """
    metadata = getattr(response, "response_metadata", {}) or {}

    # OpenAI 스타일: finish_reason
    finish_reason = metadata.get("finish_reason")
    if finish_reason:
        return finish_reason

    # Anthropic 스타일: stop_reason
    stop_reason = metadata.get("stop_reason")
    if stop_reason:
        return stop_reason

    return None
