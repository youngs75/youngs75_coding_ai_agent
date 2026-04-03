"""로깅 설정."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """프로젝트 전체 로깅을 설정한다."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
