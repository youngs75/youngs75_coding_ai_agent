"""pytest를 사용한 피보나치 함수 단위 테스트."""

from __future__ import annotations

import pytest


@pytest.fixture
def fibonacci():
    """테스트용 피보나치 함수."""

    def _fibonacci(n: int) -> int:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n < 2:
            return n

        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    return _fibonacci


def test_fibonacci_base_cases(fibonacci):
    """기본 동작: 0과 1은 그대로 반환한다."""
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2, 1),
        (3, 2),
        (5, 5),
        (7, 13),
        (10, 55),
    ],
)
def test_fibonacci_representative_values(fibonacci, n, expected):
    """대표적인 입력값에 대해 올바른 피보나치 수를 반환한다."""
    assert fibonacci(n) == expected


def test_fibonacci_raises_for_negative_input(fibonacci):
    """음수 입력은 예외를 발생시킨다."""
    with pytest.raises(ValueError):
        fibonacci(-1)
