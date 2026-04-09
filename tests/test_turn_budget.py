"""TurnBudgetTracker 단위 테스트."""

from coding_agent.core.turn_budget import BudgetVerdict, TurnBudgetTracker


class TestTurnBudgetBasic:
    """기본 동작 테스트."""

    def test_ok_under_limit(self):
        """한도 내에서는 OK."""
        tracker = TurnBudgetTracker(max_llm_calls=15)
        assert tracker.record_llm_call(1000) == BudgetVerdict.OK
        assert tracker.llm_call_count == 1

    def test_stop_at_max_calls(self):
        """max_llm_calls 도달 시 STOP."""
        tracker = TurnBudgetTracker(max_llm_calls=3)
        tracker.record_llm_call(1000)
        tracker.record_llm_call(1000)
        assert tracker.record_llm_call(1000) == BudgetVerdict.STOP

    def test_total_output_tokens_tracked(self):
        """출력 토큰이 누적된다."""
        tracker = TurnBudgetTracker()
        tracker.record_llm_call(500)
        tracker.record_llm_call(300)
        assert tracker.total_output_tokens == 800


class TestDiminishingReturns:
    """감소수익 감지 테스트."""

    def test_diminishing_returns_detected(self):
        """연속 저효율 호출 시 STOP."""
        tracker = TurnBudgetTracker(
            max_llm_calls=20,
            diminishing_streak_limit=3,
            min_delta_tokens=500,
        )
        # 정상 호출
        assert tracker.record_llm_call(1000) == BudgetVerdict.OK

        # 저효율 연속 시작
        assert tracker.record_llm_call(100) == BudgetVerdict.OK  # streak=1
        assert (
            tracker.record_llm_call(100) == BudgetVerdict.WARN_DIMINISHING
        )  # streak=2
        assert tracker.record_llm_call(100) == BudgetVerdict.STOP  # streak=3

    def test_high_output_resets_streak(self):
        """높은 출력이 streak를 초기화한다."""
        tracker = TurnBudgetTracker(
            max_llm_calls=20,
            diminishing_streak_limit=3,
            min_delta_tokens=500,
        )
        tracker.record_llm_call(100)  # streak=1
        tracker.record_llm_call(100)  # streak=2
        tracker.record_llm_call(1000)  # streak=0 (리셋)
        assert tracker.record_llm_call(100) == BudgetVerdict.OK  # streak=1

    def test_warn_at_streak_2(self):
        """streak=2에서 WARN_DIMINISHING."""
        tracker = TurnBudgetTracker(
            diminishing_streak_limit=3,
            min_delta_tokens=500,
        )
        tracker.record_llm_call(100)  # streak=1
        assert tracker.record_llm_call(100) == BudgetVerdict.WARN_DIMINISHING


class TestTurnBudgetReset:
    """reset 동작 테스트."""

    def test_reset_clears_all(self):
        """reset 후 모든 상태가 초기화된다."""
        tracker = TurnBudgetTracker(max_llm_calls=3)
        tracker.record_llm_call(1000)
        tracker.record_llm_call(1000)
        assert tracker.llm_call_count == 2

        tracker.reset()

        assert tracker.llm_call_count == 0
        assert tracker.total_output_tokens == 0
        assert tracker.record_llm_call(1000) == BudgetVerdict.OK


class TestLenientBudget:
    """마지막 phase용 완화 설정 테스트."""

    def test_lenient_config_allows_more_low_output_calls(self):
        """streak_limit=5, min_delta=300일 때 저효율 5회까지 허용."""
        tracker = TurnBudgetTracker(
            max_llm_calls=20,
            diminishing_streak_limit=5,
            min_delta_tokens=300,
        )
        # 300 미만 호출 4회 — 아직 STOP 아님
        for _ in range(4):
            verdict = tracker.record_llm_call(200)
        assert verdict != BudgetVerdict.STOP

        # 5회째에 STOP
        assert tracker.record_llm_call(200) == BudgetVerdict.STOP

    def test_lenient_min_delta_accepts_medium_output(self):
        """min_delta_tokens=300이면 350토큰은 유의미한 진전으로 판정."""
        tracker = TurnBudgetTracker(
            max_llm_calls=20,
            diminishing_streak_limit=5,
            min_delta_tokens=300,
        )
        # 350토큰은 streak 리셋
        tracker.record_llm_call(100)  # streak=1
        tracker.record_llm_call(100)  # streak=2
        tracker.record_llm_call(350)  # streak=0 (>= 300)
        assert tracker.record_llm_call(100) == BudgetVerdict.OK  # streak=1


class TestTurnBudgetSummary:
    """get_summary 테스트."""

    def test_summary_format(self):
        """summary에 핵심 정보가 포함된다."""
        tracker = TurnBudgetTracker(max_llm_calls=15)
        tracker.record_llm_call(500)
        summary = tracker.get_summary()
        assert "1/15" in summary
        assert "500" in summary
