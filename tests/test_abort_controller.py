"""AbortController 단위 테스트."""

import pytest

from coding_agent.core.abort_controller import (
    AbortController,
    AbortError,
    AbortReason,
)


class TestAbortControllerBasic:
    """기본 동작 테스트."""

    def test_not_aborted_initially(self):
        """초기 상태에서는 중단되지 않음."""
        controller = AbortController()
        assert not controller.is_aborted
        assert controller.reason is None
        assert controller.message == ""

    def test_abort_sets_state(self):
        """abort() 호출 후 상태가 설정된다."""
        controller = AbortController()
        controller.abort(AbortReason.USER_INTERRUPT)
        assert controller.is_aborted
        assert controller.reason == AbortReason.USER_INTERRUPT
        assert "중단" in controller.message

    def test_abort_custom_message(self):
        """커스텀 메시지가 기본 메시지를 대체한다."""
        controller = AbortController()
        controller.abort(AbortReason.TIMEOUT, "커스텀 메시지")
        assert controller.message == "커스텀 메시지"

    def test_abort_default_messages(self):
        """각 AbortReason에 기본 메시지가 있다."""
        for reason in AbortReason:
            controller = AbortController()
            controller.abort(reason)
            assert controller.message  # 비어 있지 않음


class TestAbortControllerCheckOrRaise:
    """check_or_raise 테스트."""

    def test_no_raise_when_not_aborted(self):
        """중단되지 않았으면 예외 없음."""
        controller = AbortController()
        controller.check_or_raise()  # 예외 없이 통과

    def test_raises_abort_error(self):
        """중단 시 AbortError 발생."""
        controller = AbortController()
        controller.abort(AbortReason.STALL_DETECTED)
        with pytest.raises(AbortError) as exc_info:
            controller.check_or_raise()
        assert exc_info.value.reason == AbortReason.STALL_DETECTED


class TestAbortControllerReset:
    """reset 동작 테스트."""

    def test_reset_clears_state(self):
        """reset 후 초기 상태로 복원."""
        controller = AbortController()
        controller.abort(AbortReason.USER_INTERRUPT)
        assert controller.is_aborted

        controller.reset()

        assert not controller.is_aborted
        assert controller.reason is None
        assert controller.message == ""

    def test_check_or_raise_after_reset(self):
        """reset 후 check_or_raise가 통과."""
        controller = AbortController()
        controller.abort(AbortReason.BUDGET_EXCEEDED)
        controller.reset()
        controller.check_or_raise()  # 예외 없이 통과


class TestAbortReasonEnum:
    """AbortReason enum 테스트."""

    def test_all_reasons_have_values(self):
        """모든 reason이 문자열 값을 가진다."""
        for reason in AbortReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0

    def test_reason_string_conversion(self):
        """AbortReason은 문자열로 변환 가능."""
        assert str(AbortReason.USER_INTERRUPT) == "AbortReason.USER_INTERRUPT"
        assert AbortReason.USER_INTERRUPT.value == "user_interrupt"
