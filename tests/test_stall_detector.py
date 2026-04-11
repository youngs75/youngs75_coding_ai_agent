"""StallDetector 단위 테스트."""

from coding_agent.core.stall_detector import StallAction, StallDetector


class TestStallDetectorBasic:
    """기본 동작 테스트."""

    def test_no_stall_different_calls(self):
        """서로 다른 도구 호출은 stall이 아니다."""
        detector = StallDetector()
        assert (
            detector.record_and_check("read_file", {"path": "a.py"})
            == StallAction.CONTINUE
        )
        assert (
            detector.record_and_check("list_directory", {"path": "."})
            == StallAction.CONTINUE
        )
        assert (
            detector.record_and_check("search_code", {"query": "foo"})
            == StallAction.CONTINUE
        )

    def test_no_stall_same_tool_different_args(self):
        """같은 도구, 다른 인자는 stall이 아니다."""
        detector = StallDetector()
        assert (
            detector.record_and_check("read_file", {"path": "a.py"})
            == StallAction.CONTINUE
        )
        assert (
            detector.record_and_check("read_file", {"path": "b.py"})
            == StallAction.CONTINUE
        )
        assert (
            detector.record_and_check("read_file", {"path": "c.py"})
            == StallAction.CONTINUE
        )

    def test_warn_on_second_identical_call(self):
        """동일 도구+인자 2회 반복 시 WARN."""
        detector = StallDetector(warn_threshold=2, exit_threshold=3)
        assert (
            detector.record_and_check("list_directory", {"path": "."})
            == StallAction.CONTINUE
        )
        assert (
            detector.record_and_check("list_directory", {"path": "."})
            == StallAction.WARN
        )

    def test_force_exit_on_third_identical_call(self):
        """동일 도구+인자 3회 반복 시 FORCE_EXIT."""
        detector = StallDetector(warn_threshold=2, exit_threshold=3)
        detector.record_and_check("list_directory", {"path": "."})
        detector.record_and_check("list_directory", {"path": "."})
        assert (
            detector.record_and_check("list_directory", {"path": "."})
            == StallAction.FORCE_EXIT
        )

    def test_none_tool_name_continues(self):
        """tool_name이 None이면 CONTINUE."""
        detector = StallDetector()
        assert detector.record_and_check(None, {}) == StallAction.CONTINUE

    def test_none_tool_args_handled(self):
        """tool_args가 None이면 빈 dict로 처리."""
        detector = StallDetector()
        assert detector.record_and_check("test", None) == StallAction.CONTINUE
        assert detector.record_and_check("test", None) == StallAction.WARN


class TestStallDetectorReset:
    """reset 동작 테스트."""

    def test_reset_clears_state(self):
        """reset 후 카운트가 초기화된다."""
        detector = StallDetector(warn_threshold=2, exit_threshold=3)
        detector.record_and_check("list_directory", {"path": "."})
        detector.record_and_check("list_directory", {"path": "."})
        # 2회 반복 상태

        detector.reset()

        # reset 후 다시 시작
        assert (
            detector.record_and_check("list_directory", {"path": "."})
            == StallAction.CONTINUE
        )

    def test_reset_empty_summary(self):
        """reset 후 summary가 빈 문자열."""
        detector = StallDetector()
        detector.record_and_check("test", {"a": 1})
        detector.reset()
        assert detector.get_stall_summary() == ""


class TestStallDetectorSummary:
    """get_stall_summary 테스트."""

    def test_summary_after_stall(self):
        """stall 감지 후 적절한 요약 메시지를 반환한다."""
        detector = StallDetector(warn_threshold=2, exit_threshold=3)
        detector.record_and_check("list_directory", {"path": "."})
        detector.record_and_check("list_directory", {"path": "."})
        detector.record_and_check("list_directory", {"path": "."})

        summary = detector.get_stall_summary()
        assert "list_directory" in summary
        assert "3회" in summary


class TestStallDetectorCustomThresholds:
    """커스텀 임계값 테스트."""

    def test_high_thresholds(self):
        """높은 임계값에서는 더 많은 반복이 필요하다."""
        detector = StallDetector(warn_threshold=5, exit_threshold=10)
        for i in range(4):
            assert detector.record_and_check("tool", {"x": 1}) == StallAction.CONTINUE
        assert detector.record_and_check("tool", {"x": 1}) == StallAction.WARN

    def test_args_order_insensitive(self):
        """dict 키 순서가 달라도 동일한 해시."""
        detector = StallDetector(warn_threshold=2)
        detector.record_and_check("tool", {"a": 1, "b": 2})
        action = detector.record_and_check("tool", {"b": 2, "a": 1})
        assert action == StallAction.WARN


class TestStallDetectorToolThresholds:
    """도구별 임계값 오버라이드 테스트."""

    def test_read_file_higher_threshold(self):
        """read_file은 기본보다 높은 임계값을 가진다 (4회 WARN, 5회 EXIT)."""
        detector = StallDetector()
        args = {"path": "a.py"}
        # 1~3회: CONTINUE
        for _ in range(3):
            assert detector.record_and_check("read_file", args) == StallAction.CONTINUE
        # 4회: WARN
        assert detector.record_and_check("read_file", args) == StallAction.WARN
        # 5회: FORCE_EXIT
        assert detector.record_and_check("read_file", args) == StallAction.FORCE_EXIT

    def test_write_file_higher_threshold(self):
        """write_file은 기본보다 높은 임계값을 가진다 (4회 WARN, 6회 EXIT)."""
        detector = StallDetector()
        args = {"path": "a.py", "content": "x"}
        # 1~3회: CONTINUE (write→read→rewrite 패턴 허용)
        for _ in range(3):
            assert detector.record_and_check("write_file", args) == StallAction.CONTINUE
        # 4회: WARN
        assert detector.record_and_check("write_file", args) == StallAction.WARN
        # 5회: WARN
        assert detector.record_and_check("write_file", args) == StallAction.WARN
        # 6회: FORCE_EXIT
        assert detector.record_and_check("write_file", args) == StallAction.FORCE_EXIT

    def test_run_shell_higher_threshold(self):
        """run_shell은 기본보다 높은 임계값을 가진다 (4회 WARN, 6회 EXIT)."""
        detector = StallDetector()
        args = {"command": "pytest tests/ -q"}
        # 1~3회: CONTINUE
        for _ in range(3):
            assert detector.record_and_check("run_shell", args) == StallAction.CONTINUE
        # 4회: WARN
        assert detector.record_and_check("run_shell", args) == StallAction.WARN
        # 5회: WARN (아직 EXIT 아님)
        assert detector.record_and_check("run_shell", args) == StallAction.WARN
        # 6회: FORCE_EXIT
        assert detector.record_and_check("run_shell", args) == StallAction.FORCE_EXIT

    def test_custom_tool_threshold_override(self):
        """커스텀 도구 임계값을 설정할 수 있다."""
        from coding_agent.core.stall_detector import _ToolThreshold

        detector = StallDetector(
            tool_thresholds={"search_code": _ToolThreshold(warn=3, exit=6)}
        )
        args = {"query": "foo"}
        for _ in range(2):
            assert detector.record_and_check("search_code", args) == StallAction.CONTINUE
        assert detector.record_and_check("search_code", args) == StallAction.WARN
