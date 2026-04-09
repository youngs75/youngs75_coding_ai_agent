"""미들웨어 레이어 테스트.

검증 항목:
- MiddlewareChain 양파 패턴 실행
- MessageWindowMiddleware 슬라이딩 윈도우
- SummarizationMiddleware 토큰 기반 컴팩션
- SkillMiddleware 스킬 주입 + 중복 방지
- MemoryMiddleware 메모리 주입 + 자동 축적
- ResilienceMiddleware 재시도 + fallback + abort 체크
- append_to_system_message 유틸리티
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from coding_agent.core.abort_controller import AbortController, AbortReason
from coding_agent.core.memory.schemas import MemoryType
from coding_agent.core.memory.store import MemoryStore
from coding_agent.core.middleware import (
    AgentMiddleware,
    MemoryMiddleware,
    MessageWindowMiddleware,
    MiddlewareChain,
    ModelRequest,
    ModelResponse,
    ResilienceMiddleware,
    SkillMiddleware,
    SummarizationMiddleware,
    append_to_system_message,
)
from coding_agent.core.resilience import FailureMatrix


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


# ── ResilienceMiddleware ──


class TestResilienceMiddleware:
    @pytest.fixture()
    def abort_controller(self) -> AbortController:
        return AbortController()

    @pytest.fixture()
    def resilience_mw(self, abort_controller: AbortController) -> ResilienceMiddleware:
        return ResilienceMiddleware(abort_controller=abort_controller)

    @pytest.mark.asyncio
    async def test_normal_flow(self, resilience_mw: ResilienceMiddleware):
        """정상 흐름 — abort 없이 handler 결과를 그대로 반환."""
        mock_handler = AsyncMock(
            return_value=ModelResponse(message=AIMessage(content="ok"))
        )
        request = ModelRequest(
            system_message="system",
            messages=[HumanMessage(content="test")],
        )
        response = await resilience_mw.wrap_model_call(request, mock_handler)

        assert response.message.content == "ok"
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_abort_before_llm(self, abort_controller: AbortController):
        """before 단계에서 abort 신호가 있으면 AbortError 발생."""
        from coding_agent.core.abort_controller import AbortError

        mw = ResilienceMiddleware(abort_controller=abort_controller)
        abort_controller.abort(AbortReason.USER_INTERRUPT)

        mock_handler = AsyncMock()
        request = ModelRequest(system_message="", messages=[])

        with pytest.raises(AbortError):
            await mw.wrap_model_call(request, mock_handler)

        # handler는 호출되지 않아야 함
        mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_abort_after_llm(self, abort_controller: AbortController):
        """after 단계에서 abort 신호가 있으면 AbortError 발생."""
        from coding_agent.core.abort_controller import AbortError

        call_count = 0

        async def handler_that_triggers_abort(req):
            nonlocal call_count
            call_count += 1
            # handler 실행 후 abort 신호 발생
            abort_controller.abort(AbortReason.BUDGET_EXCEEDED)
            return ModelResponse(message=AIMessage(content="done"))

        mw = ResilienceMiddleware(abort_controller=abort_controller)
        request = ModelRequest(system_message="", messages=[])

        with pytest.raises(AbortError):
            await mw.wrap_model_call(request, handler_that_triggers_abort)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, abort_controller: AbortController):
        """handler 실패 시 재시도가 수행되는지 확인."""
        matrix = FailureMatrix()
        mw = ResilienceMiddleware(
            abort_controller=abort_controller,
            failure_matrix=matrix,
        )

        call_count = 0

        async def flaky_handler(req):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient error")
            return ModelResponse(message=AIMessage(content="success"))

        request = ModelRequest(
            system_message="",
            messages=[],
            metadata={"purpose": "generation"},
        )
        response = await mw.wrap_model_call(request, flaky_handler)

        assert response.message.content == "success"
        assert call_count == 3  # 2번 실패 + 1번 성공

    @pytest.mark.asyncio
    async def test_fallback_on_exhausted_retries(
        self, abort_controller: AbortController
    ):
        """재시도 소진 후 fallback 체인이 호출되는지 확인."""
        mock_fallback_model = AsyncMock()
        mock_fallback_model.ainvoke.return_value = AIMessage(content="fallback result")

        from coding_agent.core.resilience import ModelFallbackChain

        def build_fallback():
            return ModelFallbackChain(
                models=[mock_fallback_model],
                model_names=["fallback_model"],
                timeout_per_model=10.0,
            )

        mw = ResilienceMiddleware(
            abort_controller=abort_controller,
            fallback_chain_builder=build_fallback,
        )

        async def always_fail(req):
            raise Exception("permanent failure")

        request = ModelRequest(
            system_message="",
            messages=[HumanMessage(content="test")],
            metadata={"purpose": "generation"},
        )
        response = await mw.wrap_model_call(request, always_fail)

        assert response.message.content == "fallback result"


# ── MemoryMiddleware (확장: store 검색 + 자동 축적) ──


class TestMemoryMiddlewareExtended:
    @pytest.fixture()
    def memory_store(self) -> MemoryStore:
        return MemoryStore()

    @pytest.mark.asyncio
    async def test_store_search_injection(self, memory_store: MemoryStore):
        """MemoryStore 검색 결과가 system prompt에 주입되는지 확인."""
        # 도메인 지식 추가
        memory_store.accumulate_domain_knowledge(
            content="Flask는 WSGI 기반 마이크로 프레임워크이다",
            tags=["flask", "python"],
        )

        mw = MemoryMiddleware(memory_store=memory_store)
        mock_handler = AsyncMock(
            return_value=ModelResponse(message=AIMessage(content="ok"))
        )

        request = ModelRequest(
            system_message="base",
            messages=[HumanMessage(content="Flask 프로젝트에 CORS를 설정하려면?")],
            metadata={"purpose": "generation"},
        )
        await mw.wrap_model_call(request, mock_handler)

        passed = mock_handler.call_args[0][0]
        assert "Flask" in passed.system_message
        assert "WSGI" in passed.system_message

    @pytest.mark.asyncio
    async def test_auto_accumulate_domain_knowledge(self, memory_store: MemoryStore):
        """after 단계에서 도메인 지식이 자동 축적되는지 확인."""

        async def mock_slm(messages):
            return '{"type": "domain_knowledge", "content": "React는 SPA 프레임워크이다", "tags": ["react"]}'

        mw = MemoryMiddleware(
            memory_store=memory_store,
            slm_invoker=mock_slm,
            auto_accumulate=True,
        )
        mock_handler = AsyncMock(
            return_value=ModelResponse(
                message=AIMessage(content="React는 SPA 프레임워크입니다.")
            )
        )

        request = ModelRequest(system_message="", messages=[], state={})
        await mw.wrap_model_call(request, mock_handler)

        items = memory_store.list_by_type(MemoryType.DOMAIN_KNOWLEDGE)
        assert len(items) == 1
        assert "React" in items[0].content

    @pytest.mark.asyncio
    async def test_auto_accumulate_user_profile(self, memory_store: MemoryStore):
        """after 단계에서 사용자 프로필이 자동 축적되는지 확인."""

        async def mock_slm(messages):
            return '{"type": "user_profile", "content": "사용자는 Python 전문가이다", "tags": ["skill"]}'

        mw = MemoryMiddleware(
            memory_store=memory_store,
            slm_invoker=mock_slm,
            auto_accumulate=True,
        )
        mock_handler = AsyncMock(
            return_value=ModelResponse(message=AIMessage(content="ok"))
        )

        request = ModelRequest(system_message="", messages=[], state={})
        await mw.wrap_model_call(request, mock_handler)

        items = memory_store.list_by_type(MemoryType.USER_PROFILE)
        assert len(items) == 1
        assert "Python" in items[0].content

    @pytest.mark.asyncio
    async def test_auto_accumulate_disabled(self, memory_store: MemoryStore):
        """auto_accumulate=False이면 축적하지 않음."""

        async def mock_slm(messages):
            return '{"type": "domain_knowledge", "content": "test", "tags": []}'

        mw = MemoryMiddleware(
            memory_store=memory_store,
            slm_invoker=mock_slm,
            auto_accumulate=False,
        )
        mock_handler = AsyncMock(
            return_value=ModelResponse(message=AIMessage(content="ok"))
        )

        request = ModelRequest(system_message="", messages=[], state={})
        await mw.wrap_model_call(request, mock_handler)

        assert memory_store.total_count == 0

    @pytest.mark.asyncio
    async def test_slm_failure_does_not_break_flow(self, memory_store: MemoryStore):
        """SLM 호출 실패 시 LLM 응답은 정상 반환."""

        async def failing_slm(messages):
            raise RuntimeError("SLM down")

        mw = MemoryMiddleware(
            memory_store=memory_store,
            slm_invoker=failing_slm,
            auto_accumulate=True,
        )
        mock_handler = AsyncMock(
            return_value=ModelResponse(message=AIMessage(content="important result"))
        )

        request = ModelRequest(system_message="", messages=[], state={})
        response = await mw.wrap_model_call(request, mock_handler)

        assert response.message.content == "important result"


# ── ResilienceMiddleware + MemoryMiddleware 체인 통합 ──


class TestResilienceMemoryIntegration:
    @pytest.mark.asyncio
    async def test_resilience_and_memory_in_chain(self):
        """ResilienceMiddleware + MemoryMiddleware를 체인으로 연결한 통합 테스트."""
        abort = AbortController()
        store = MemoryStore()
        store.accumulate_domain_knowledge(
            content="Python 비동기 패턴은 asyncio를 사용한다",
            tags=["python", "async"],
        )

        chain = MiddlewareChain([
            ResilienceMiddleware(abort_controller=abort),
            MemoryMiddleware(memory_store=store),
        ])

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="async 코드 생성 완료")

        request = ModelRequest(
            system_message="코드 생성 에이전트",
            messages=[HumanMessage(content="asyncio로 웹 크롤러를 만들어줘")],
            metadata={"purpose": "generation"},
        )
        response = await chain.invoke(request, mock_llm)

        assert response.message.content == "async 코드 생성 완료"

        # 메모리가 주입되었는지 확인
        call_args = mock_llm.ainvoke.call_args[0][0]
        system_content = call_args[0].content
        assert "asyncio" in system_content
