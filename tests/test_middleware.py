"""미들웨어 레이어 테스트.

검증 항목:
- MiddlewareChain 양파 패턴 실행
- MessageWindowMiddleware 슬라이딩 윈도우
- SummarizationMiddleware 토큰 기반 컴팩션
- SkillMiddleware 스킬 주입 + 중복 방지
- MemoryMiddleware 메모리 주입
- append_to_system_message 유틸리티
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from youngs75_a2a.core.middleware import (
    AgentMiddleware,
    MemoryMiddleware,
    MessageWindowMiddleware,
    MiddlewareChain,
    ModelRequest,
    ModelResponse,
    SkillMiddleware,
    SummarizationMiddleware,
    append_to_system_message,
)


# ── append_to_system_message ──


class TestAppendToSystemMessage:
    def test_append_to_existing(self):
        result = append_to_system_message("기존 프롬프트", "추가 내용")
        assert "기존 프롬프트\n\n추가 내용" == result

    def test_append_to_empty(self):
        result = append_to_system_message("", "추가 내용")
        assert result == "추가 내용"

    def test_append_to_none_like(self):
        result = append_to_system_message("", "first")
        assert result == "first"


# ── MiddlewareChain ──


class TestMiddlewareChain:
    @pytest.mark.asyncio
    async def test_empty_chain_calls_llm(self):
        chain = MiddlewareChain([])
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="hello")

        request = ModelRequest(
            system_message="system",
            messages=[HumanMessage(content="user")],
        )
        response = await chain.invoke(request, mock_llm)

        assert response.message.content == "hello"
        mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_middleware_modifies_request(self):
        """미들웨어가 system_message를 수정하는지 확인."""

        class PrefixMiddleware(AgentMiddleware):
            async def wrap_model_call(self, request, handler):
                modified = request.override(
                    system_message=request.system_message + " [modified]"
                )
                return await handler(modified)

        chain = MiddlewareChain([PrefixMiddleware()])
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="ok")

        request = ModelRequest(
            system_message="base",
            messages=[HumanMessage(content="test")],
        )
        await chain.invoke(request, mock_llm)

        # LLM에 전달된 첫 번째 메시지(SystemMessage)가 수정되었는지 확인
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_msg = call_args[0]
        assert "[modified]" in system_msg.content

    @pytest.mark.asyncio
    async def test_onion_order(self):
        """양파 패턴 — 첫 번째 미들웨어가 먼저 실행, 마지막에 반환."""
        order = []

        class OrderTracker(AgentMiddleware):
            def __init__(self, name):
                self._name = name

            async def wrap_model_call(self, request, handler):
                order.append(f"before:{self._name}")
                response = await handler(request)
                order.append(f"after:{self._name}")
                return response

        chain = MiddlewareChain([OrderTracker("A"), OrderTracker("B")])
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="ok")

        request = ModelRequest(system_message="", messages=[HumanMessage(content="test")])
        await chain.invoke(request, mock_llm)

        assert order == ["before:A", "before:B", "after:B", "after:A"]


# ── MessageWindowMiddleware ──


class TestMessageWindowMiddleware:
    @pytest.mark.asyncio
    async def test_small_messages_pass_through(self):
        """메시지가 적으면 그대로 통과."""
        mw = MessageWindowMiddleware(max_turns=4, max_messages=10)
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        request = ModelRequest(
            system_message="system",
            messages=[HumanMessage(content="hello"), AIMessage(content="hi")],
        )
        await mw.wrap_model_call(request, mock_handler)

        # handler에 전달된 messages가 원본과 동일해야 함
        passed_request = mock_handler.call_args[0][0]
        assert len(passed_request.messages) == 2

    @pytest.mark.asyncio
    async def test_large_messages_trimmed(self):
        """메시지가 많으면 윈도우로 잘림."""
        mw = MessageWindowMiddleware(max_turns=2, max_messages=4)
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        # 20개 메시지 (10턴)
        messages = []
        for i in range(10):
            messages.append(HumanMessage(content=f"user msg {i}"))
            messages.append(AIMessage(content=f"ai msg {i}"))

        request = ModelRequest(system_message="system", messages=messages)
        await mw.wrap_model_call(request, mock_handler)

        passed_request = mock_handler.call_args[0][0]
        # 원본 20개보다 적어야 함
        assert len(passed_request.messages) < 20
        # 첫 번째 메시지(원본 태스크)는 유지
        assert "user msg 0" in passed_request.messages[0].content
        # 마지막 메시지는 가장 최근 것
        assert "ai msg 9" in passed_request.messages[-1].content


# ── SummarizationMiddleware ──


class TestSummarizationMiddleware:
    @pytest.mark.asyncio
    async def test_small_context_no_compaction(self):
        """토큰이 임계치 이하면 컴팩션 없음."""
        mw = SummarizationMiddleware(token_threshold=100_000)
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        request = ModelRequest(
            system_message="short system",
            messages=[HumanMessage(content="short message")],
        )
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        assert len(passed.messages) == 1

    @pytest.mark.asyncio
    async def test_large_context_triggers_compaction(self):
        """토큰 초과 시 메시지가 줄어야 함."""
        mw = SummarizationMiddleware(
            token_threshold=100,  # 매우 낮은 임계치
            keep_recent_messages=2,
        )
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        # 큰 메시지 생성
        messages = [
            HumanMessage(content="x" * 500),
            AIMessage(content="y" * 500),
            HumanMessage(content="z" * 500),
            AIMessage(content="w" * 500),
        ]

        request = ModelRequest(system_message="system", messages=messages)
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        # 원본 4개와 같거나 적어야 함 (첫 메시지 + 요약 + 최근 2개 = 4)
        assert len(passed.messages) <= 4
        # 요약 메시지가 포함되어야 함
        summary_found = any("컨텍스트 축소" in getattr(m, "content", "") for m in passed.messages)
        assert summary_found, "요약 메시지가 없음"


# ── SkillMiddleware ──


class TestSkillMiddleware:
    @pytest.mark.asyncio
    async def test_no_registry_pass_through(self):
        mw = SkillMiddleware(skill_registry=None)
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        request = ModelRequest(system_message="base", messages=[])
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        assert passed.system_message == "base"

    @pytest.mark.asyncio
    async def test_skill_injected(self):
        """스킬이 시스템 프롬프트에 주입되는지 확인."""

        class MockRegistry:
            def get_active_skill_bodies(self):
                return ["## Flask 패턴\n- CORS 설정"]

        mw = SkillMiddleware(skill_registry=MockRegistry())
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        request = ModelRequest(system_message="base", messages=[])
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        assert "Flask 패턴" in passed.system_message

    @pytest.mark.asyncio
    async def test_duplicate_prevention(self):
        """이미 주입된 스킬은 중복 주입하지 않음."""

        class MockRegistry:
            def get_active_skill_bodies(self):
                return ["## Flask 패턴"]

        mw = SkillMiddleware(skill_registry=MockRegistry())
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        # 이미 스킬이 포함된 시스템 프롬프트
        request = ModelRequest(
            system_message="base\n\n## 활성 스킬\n- ## Flask 패턴",
            messages=[],
        )
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        # "Flask 패턴"이 한 번만 있어야 함
        assert passed.system_message.count("Flask 패턴") == 1


# ── MemoryMiddleware ──


class TestMemoryMiddleware:
    @pytest.mark.asyncio
    async def test_no_memory_pass_through(self):
        mw = MemoryMiddleware()
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        request = ModelRequest(system_message="base", messages=[], state={})
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        assert passed.system_message == "base"

    @pytest.mark.asyncio
    async def test_memory_injected(self):
        mw = MemoryMiddleware()
        mock_handler = AsyncMock(return_value=ModelResponse(message=AIMessage(content="ok")))

        request = ModelRequest(
            system_message="base",
            messages=[],
            state={
                "semantic_context": ["프로젝트는 Flask 기반입니다"],
                "episodic_log": ["이전에 CORS 설정 실패함"],
            },
        )
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        assert "Flask 기반" in passed.system_message
        assert "CORS 설정 실패" in passed.system_message


# ── 통합 테스트 ──


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_chain_with_all_middlewares(self):
        """모든 미들웨어를 체인으로 연결한 통합 테스트."""

        class MockRegistry:
            def get_active_skill_bodies(self):
                return ["## Test Skill"]

        chain = MiddlewareChain([
            MessageWindowMiddleware(max_turns=4),
            SummarizationMiddleware(token_threshold=100_000),
            SkillMiddleware(skill_registry=MockRegistry()),
            MemoryMiddleware(),
        ])

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="generated code")

        request = ModelRequest(
            system_message="base system",
            messages=[
                HumanMessage(content="칸반보드 만들어줘"),
                AIMessage(content="네, 만들겠습니다"),
            ],
            state={"semantic_context": ["Flask 프로젝트"]},
        )

        response = await chain.invoke(request, mock_llm)
        assert response.message.content == "generated code"

        # LLM에 전달된 시스템 프롬프트에 스킬 + 메모리가 주입되었는지 확인
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_content = call_args[0].content
        assert "Test Skill" in system_content
        assert "Flask 프로젝트" in system_content
