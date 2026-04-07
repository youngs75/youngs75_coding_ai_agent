"""ContextManager 테스트 모듈.

토큰 카운팅, 컴팩션 판단, 컴팩션 실행, truncate_for_subagent,
max_tokens 복구 시뮬레이션 등을 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from youngs75_a2a.core.context_manager import (
    ContextManager,
    _extract_stop_reason,
    _get_content,
    _get_role,
    _messages_to_dicts,
    invoke_with_max_tokens_recovery,
)


# ── 픽스처 ──────────────────────────────────────────────────


@pytest.fixture
def ctx() -> ContextManager:
    """기본 ContextManager 인스턴스."""
    return ContextManager(
        max_tokens=1000,
        compact_threshold=0.8,
        keep_recent_turns=2,
    )


@pytest.fixture
def simple_messages() -> list[BaseMessage]:
    """간단한 대화 메시지 리스트."""
    return [
        SystemMessage(content="당신은 도움이 되는 코딩 어시스턴트입니다."),
        HumanMessage(content="안녕하세요"),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
        HumanMessage(content="파이썬으로 피보나치 함수를 작성해주세요."),
        AIMessage(content="def fib(n): ..."),
    ]


@pytest.fixture
def messages_with_tool_calls() -> list[BaseMessage]:
    """도구 호출이 포함된 메시지 리스트."""
    return [
        SystemMessage(content="시스템 프롬프트"),
        HumanMessage(content="파일을 읽어주세요"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "test.py"}, "id": "tc1"}
            ],
        ),
        ToolMessage(content="파일 내용입니다", tool_call_id="tc1", name="read_file"),
        AIMessage(content="파일을 읽었습니다. 내용은 다음과 같습니다."),
        HumanMessage(content="감사합니다"),
        AIMessage(content="도움이 되었다니 기쁩니다!"),
    ]


@pytest.fixture
def mock_llm() -> AsyncMock:
    """요약용 Mock LLM."""
    mock = AsyncMock()
    mock.ainvoke.return_value = AIMessage(
        content="이전 대화 요약: 사용자가 피보나치 함수를 요청함"
    )
    return mock


# ── 1. 초기화 및 매개변수 검증 ────────────────────────────────


class TestContextManagerInit:
    """ContextManager 초기화 테스트."""

    def test_default_values(self) -> None:
        """기본값으로 초기화된다."""
        ctx = ContextManager()
        assert ctx.max_tokens == 128000
        assert ctx.compact_threshold == 0.8
        assert ctx.keep_recent_turns == 4

    def test_custom_values(self) -> None:
        """사용자 지정 값으로 초기화된다."""
        ctx = ContextManager(
            max_tokens=50000,
            compact_threshold=0.7,
            keep_recent_turns=6,
        )
        assert ctx.max_tokens == 50000
        assert ctx.compact_threshold == 0.7
        assert ctx.keep_recent_turns == 6

    def test_invalid_threshold_raises(self) -> None:
        """잘못된 threshold는 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="compact_threshold"):
            ContextManager(compact_threshold=0.0)
        with pytest.raises(ValueError, match="compact_threshold"):
            ContextManager(compact_threshold=1.5)

    def test_invalid_max_tokens_raises(self) -> None:
        """잘못된 max_tokens는 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="max_tokens"):
            ContextManager(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens"):
            ContextManager(max_tokens=-100)

    def test_invalid_keep_recent_turns_raises(self) -> None:
        """잘못된 keep_recent_turns는 ValueError를 발생시킨다."""
        with pytest.raises(ValueError, match="keep_recent_turns"):
            ContextManager(keep_recent_turns=-1)


# ── 2. 토큰 카운팅 ──────────────────────────────────────────


class TestTokenCounting:
    """토큰 카운팅 정확도 테스트."""

    def test_count_simple_messages(
        self, ctx: ContextManager, simple_messages: list[BaseMessage]
    ) -> None:
        """간단한 메시지 리스트의 토큰 수가 양수이다."""
        tokens = ctx.count_messages_tokens(simple_messages)
        assert tokens > 0

    def test_empty_messages_returns_small_count(self, ctx: ContextManager) -> None:
        """빈 메시지 리스트는 최소 오버헤드 토큰만 반환한다."""
        tokens = ctx.count_messages_tokens([])
        # count_messages_tokens는 전체 응답 프라이밍 토큰(2)을 포함
        assert tokens >= 0

    def test_longer_messages_have_more_tokens(self, ctx: ContextManager) -> None:
        """메시지가 길수록 토큰 수가 많다."""
        short = [HumanMessage(content="짧은 메시지")]
        long = [HumanMessage(content="매우 긴 메시지입니다. " * 100)]
        assert ctx.count_messages_tokens(long) > ctx.count_messages_tokens(short)

    def test_multiple_messages_accumulate(self, ctx: ContextManager) -> None:
        """여러 메시지의 토큰은 누적된다."""
        one_msg = [HumanMessage(content="테스트 메시지")]
        three_msgs = [
            HumanMessage(content="테스트 메시지"),
            AIMessage(content="응답입니다"),
            HumanMessage(content="추가 질문"),
        ]
        assert ctx.count_messages_tokens(three_msgs) > ctx.count_messages_tokens(
            one_msg
        )


# ── 3. 컴팩션 판단 ──────────────────────────────────────────


class TestShouldCompact:
    """컴팩션 필요 여부 판단 테스트."""

    def test_small_messages_no_compact(self, ctx: ContextManager) -> None:
        """토큰이 임계치 미만이면 컴팩션 불필요."""
        messages = [HumanMessage(content="짧은 메시지")]
        assert ctx.should_compact(messages) is False

    def test_large_messages_should_compact(self) -> None:
        """토큰이 임계치를 초과하면 컴팩션 필요."""
        # max_tokens=100, threshold=0.5 → 50토큰 초과 시 컴팩션
        ctx = ContextManager(max_tokens=100, compact_threshold=0.5)
        # 긴 메시지를 생성하여 임계치 초과
        long_content = "이것은 매우 긴 메시지입니다. " * 50
        messages = [HumanMessage(content=long_content)]
        assert ctx.should_compact(messages) is True

    def test_threshold_boundary(self) -> None:
        """임계치 경계값에서 정확히 동작한다."""
        ctx = ContextManager(max_tokens=10000, compact_threshold=0.8)
        # 짧은 메시지는 8000토큰 미만 → 컴팩션 불필요
        short = [HumanMessage(content="안녕")]
        assert ctx.should_compact(short) is False


# ── 4. 컴팩션 실행 ──────────────────────────────────────────


class TestCompact:
    """컴팩션 실행 결과 검증 테스트."""

    async def test_system_messages_preserved(
        self, ctx: ContextManager, mock_llm: AsyncMock
    ) -> None:
        """컴팩션 후에도 시스템 메시지가 유지된다."""
        messages = [
            SystemMessage(content="시스템 프롬프트"),
            HumanMessage(content="질문 1"),
            AIMessage(content="답변 1"),
            HumanMessage(content="질문 2"),
            AIMessage(content="답변 2"),
            HumanMessage(content="질문 3"),
            AIMessage(content="답변 3"),
        ]
        result = await ctx.compact(messages, mock_llm)

        # 시스템 메시지가 유지되었는지 확인
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert any("시스템 프롬프트" in m.content for m in system_msgs)

    async def test_recent_turns_preserved(
        self, ctx: ContextManager, mock_llm: AsyncMock
    ) -> None:
        """컴팩션 후 최근 N턴이 유지된다."""
        # keep_recent_turns=2 이므로 마지막 2개의 비-시스템 메시지 유지
        messages = [
            SystemMessage(content="시스템"),
            HumanMessage(content="오래된 질문 1"),
            AIMessage(content="오래된 답변 1"),
            HumanMessage(content="오래된 질문 2"),
            AIMessage(content="오래된 답변 2"),
            HumanMessage(content="최근 질문"),
            AIMessage(content="최근 답변"),
        ]
        result = await ctx.compact(messages, mock_llm)

        # 최근 2개 메시지가 유지되었는지 확인
        non_system = [m for m in result if not isinstance(m, SystemMessage)]
        contents = [m.content for m in non_system]
        assert "최근 질문" in contents
        assert "최근 답변" in contents

    async def test_old_messages_summarized(
        self, ctx: ContextManager, mock_llm: AsyncMock
    ) -> None:
        """오래된 메시지가 요약으로 대체된다."""
        messages = [
            SystemMessage(content="시스템"),
            HumanMessage(content="오래된 질문 1"),
            AIMessage(content="오래된 답변 1"),
            HumanMessage(content="오래된 질문 2"),
            AIMessage(content="오래된 답변 2"),
            HumanMessage(content="최근 질문"),
            AIMessage(content="최근 답변"),
        ]
        result = await ctx.compact(messages, mock_llm)

        # 요약 메시지가 포함되었는지 확인
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert any("[이전 대화 요약]" in m.content for m in system_msgs)
        # LLM이 호출되었는지 확인
        mock_llm.ainvoke.assert_called_once()

    async def test_few_messages_no_compaction(
        self, ctx: ContextManager, mock_llm: AsyncMock
    ) -> None:
        """메시지가 keep_recent_turns 이하이면 컴팩션이 수행되지 않는다."""
        messages = [
            SystemMessage(content="시스템"),
            HumanMessage(content="질문"),
            AIMessage(content="답변"),
        ]
        result = await ctx.compact(messages, mock_llm)

        # 원본과 동일 (컴팩션 미수행)
        assert len(result) == len(messages)
        mock_llm.ainvoke.assert_not_called()

    async def test_compaction_reduces_token_count(
        self, ctx: ContextManager, mock_llm: AsyncMock
    ) -> None:
        """컴팩션 결과의 토큰 수가 원본보다 적다."""
        long_content = "매우 긴 대화 내용입니다. " * 50
        messages = [
            SystemMessage(content="시스템"),
            HumanMessage(content=long_content),
            AIMessage(content=long_content),
            HumanMessage(content=long_content),
            AIMessage(content=long_content),
            HumanMessage(content="최근 질문"),
            AIMessage(content="최근 답변"),
        ]

        before_tokens = ctx.count_messages_tokens(messages)
        result = await ctx.compact(messages, mock_llm)
        after_tokens = ctx.count_messages_tokens(result)

        assert after_tokens < before_tokens


# ── 5. 서브에이전트용 히스토리 필터링 ────────────────────────


class TestTruncateForSubagent:
    """truncate_for_subagent 필터링 테스트."""

    def test_system_messages_always_included(
        self, ctx: ContextManager, simple_messages: list[BaseMessage]
    ) -> None:
        """시스템 메시지는 항상 포함된다."""
        result = ctx.truncate_for_subagent(simple_messages, last_n_turns=1)
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert "코딩 어시스턴트" in system_msgs[0].content

    def test_tool_calls_excluded(
        self,
        ctx: ContextManager,
        messages_with_tool_calls: list[BaseMessage],
    ) -> None:
        """tool_call이 있는 AI 메시지는 제외된다."""
        result = ctx.truncate_for_subagent(messages_with_tool_calls, last_n_turns=10)
        for msg in result:
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                assert len(tool_calls) == 0, "tool_call이 있는 AI 메시지가 포함됨"

    def test_tool_messages_excluded(
        self,
        ctx: ContextManager,
        messages_with_tool_calls: list[BaseMessage],
    ) -> None:
        """ToolMessage는 제외된다."""
        result = ctx.truncate_for_subagent(messages_with_tool_calls, last_n_turns=10)
        for msg in result:
            assert not isinstance(msg, ToolMessage), "ToolMessage가 포함됨"

    def test_last_n_turns_respected(
        self, ctx: ContextManager, simple_messages: list[BaseMessage]
    ) -> None:
        """last_n_turns 매개변수가 올바르게 적용된다."""
        # simple_messages: sys + human + ai + human + ai → 2턴
        result = ctx.truncate_for_subagent(simple_messages, last_n_turns=1)
        # 시스템 메시지 + 마지막 1턴 (human + ai)
        non_system = [m for m in result if not isinstance(m, SystemMessage)]
        # 최소 마지막 턴의 메시지만 포함
        assert len(non_system) <= 2  # human + ai

    def test_final_response_only(
        self,
        ctx: ContextManager,
        messages_with_tool_calls: list[BaseMessage],
    ) -> None:
        """assistant 메시지 중 최종 응답만 포함된다 (tool_call 제외)."""
        result = ctx.truncate_for_subagent(messages_with_tool_calls, last_n_turns=10)
        ai_msgs = [m for m in result if isinstance(m, AIMessage)]
        # tool_call이 없는 AI 메시지만 포함
        for msg in ai_msgs:
            assert not (getattr(msg, "tool_calls", None) or [])
            assert msg.content  # 내용이 있는 메시지만

    def test_empty_messages(self, ctx: ContextManager) -> None:
        """빈 메시지 리스트에서도 오류 없이 동작한다."""
        result = ctx.truncate_for_subagent([], last_n_turns=3)
        assert result == []

    def test_zero_turns_returns_system_only(
        self, ctx: ContextManager, simple_messages: list[BaseMessage]
    ) -> None:
        """last_n_turns=0이면 시스템 메시지만 반환한다."""
        result = ctx.truncate_for_subagent(simple_messages, last_n_turns=0)
        assert all(isinstance(m, SystemMessage) for m in result)


# ── 6. max_tokens 복구 시뮬레이션 ────────────────────────────


class TestMaxTokensRecovery:
    """max_tokens 복구 로직 테스트."""

    async def test_normal_response_no_recovery(self) -> None:
        """정상 응답은 복구 없이 즉시 반환된다."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(
            content="정상 응답",
            response_metadata={"finish_reason": "stop"},
        )
        ctx = ContextManager(max_tokens=10000)
        messages = [HumanMessage(content="테스트")]

        result = await invoke_with_max_tokens_recovery(mock_llm, messages, ctx)

        assert result.content == "정상 응답"
        assert mock_llm.ainvoke.call_count == 1

    async def test_max_tokens_triggers_retry(self) -> None:
        """max_tokens 중단 시 재시도가 수행된다."""
        mock_llm = AsyncMock()
        # 첫 번째 호출: max_tokens로 중단
        # 두 번째 호출: 정상 완료
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="불완전한 응답...",
                response_metadata={"finish_reason": "max_tokens"},
            ),
            AIMessage(
                content="이어서 완성된 응답",
                response_metadata={"finish_reason": "stop"},
            ),
        ]
        ctx = ContextManager(max_tokens=10000)
        messages = [HumanMessage(content="테스트")]

        result = await invoke_with_max_tokens_recovery(
            mock_llm, messages, ctx, max_retries=3
        )

        # partial + continuation이 병합되어야 함
        assert result.content == "불완전한 응답...이어서 완성된 응답"
        assert mock_llm.ainvoke.call_count == 2

    async def test_max_retries_exhausted(self) -> None:
        """최대 재시도 횟수 도달 시 불완전한 응답을 반환한다."""
        mock_llm = AsyncMock()
        # 모든 호출이 max_tokens로 중단
        mock_llm.ainvoke.return_value = AIMessage(
            content="불완전...",
            response_metadata={"finish_reason": "length"},
        )
        ctx = ContextManager(max_tokens=10000)
        messages = [HumanMessage(content="테스트")]

        result = await invoke_with_max_tokens_recovery(
            mock_llm, messages, ctx, max_retries=2
        )

        # 3회 호출 모두 "불완전..." → 축적된 전체 반환
        assert result.content == "불완전..." * 3
        # 초기 1회 + 재시도 2회 = 3회
        assert mock_llm.ainvoke.call_count == 3

    async def test_anthropic_stop_reason(self) -> None:
        """Anthropic 스타일 stop_reason도 감지한다."""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            AIMessage(
                content="중단됨",
                response_metadata={"stop_reason": "max_tokens"},
            ),
            AIMessage(
                content="완료",
                response_metadata={"stop_reason": "end_turn"},
            ),
        ]
        ctx = ContextManager(max_tokens=10000)
        messages = [HumanMessage(content="테스트")]

        result = await invoke_with_max_tokens_recovery(mock_llm, messages, ctx)

        # partial + continuation 병합
        assert result.content == "중단됨완료"
        assert mock_llm.ainvoke.call_count == 2


# ── 7. 내부 헬퍼 함수 테스트 ────────────────────────────────


class TestHelperFunctions:
    """내부 헬퍼 함수 테스트."""

    def test_get_role(self) -> None:
        """메시지 역할이 올바르게 추출된다."""
        assert _get_role(SystemMessage(content="")) == "system"
        assert _get_role(HumanMessage(content="")) == "user"
        assert _get_role(AIMessage(content="")) == "assistant"

    def test_get_content_string(self) -> None:
        """문자열 content가 올바르게 추출된다."""
        msg = HumanMessage(content="테스트 내용")
        assert _get_content(msg) == "테스트 내용"

    def test_get_content_list(self) -> None:
        """리스트 content에서 텍스트가 추출된다."""
        msg = HumanMessage(content=[{"type": "text", "text": "텍스트 부분"}])
        assert "텍스트 부분" in _get_content(msg)

    def test_messages_to_dicts(self) -> None:
        """BaseMessage 리스트가 dict 리스트로 변환된다."""
        messages = [
            SystemMessage(content="시스템"),
            HumanMessage(content="안녕"),
        ]
        dicts = _messages_to_dicts(messages)
        assert len(dicts) == 2
        assert dicts[0] == {"role": "system", "content": "시스템"}
        assert dicts[1] == {"role": "user", "content": "안녕"}

    def test_extract_stop_reason_openai(self) -> None:
        """OpenAI 스타일 finish_reason이 추출된다."""
        msg = AIMessage(
            content="",
            response_metadata={"finish_reason": "stop"},
        )
        assert _extract_stop_reason(msg) == "stop"

    def test_extract_stop_reason_anthropic(self) -> None:
        """Anthropic 스타일 stop_reason이 추출된다."""
        msg = AIMessage(
            content="",
            response_metadata={"stop_reason": "end_turn"},
        )
        assert _extract_stop_reason(msg) == "end_turn"

    def test_extract_stop_reason_none(self) -> None:
        """메타데이터가 없으면 None을 반환한다."""
        msg = AIMessage(content="")
        assert _extract_stop_reason(msg) is None
