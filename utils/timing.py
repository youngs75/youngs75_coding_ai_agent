"""함수 실행 시간을 측정하는 데코레이터 유틸리티."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def measure_execution_time(
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    message_template: str = "Function {name} executed in {elapsed:.6f}s",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """함수 실행 시간을 측정하고 로그를 남기는 데코레이터를 반환한다.

    Args:
        logger: 실행 시간을 기록할 로거. 지정하지 않으면 모듈 로거를 사용한다.
        level: 로그 레벨.
        message_template: 로그 메시지 템플릿.
            사용 가능한 키: name, elapsed.

    Returns:
        데코레이터 함수.
    """

    active_logger = logger or logging.getLogger(__name__)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                active_logger.log(
                    level,
                    message_template.format(name=func.__name__, elapsed=elapsed),
                )

        return wrapper

    return decorator
