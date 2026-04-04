"""LLM 응답 캐싱 모듈.

동일한 입력에 대한 LLM 응답을 인메모리 LRU 캐시에 저장한다.
parse 노드처럼 결정적(deterministic) 결과가 예상되는 곳에 적용 가능.

캐시 키: (model_name, prompt_hash, temperature)
TTL 기반 자동 만료, 캐시 히트율 메트릭 제공.

사용 예시:
    from youngs75_a2a.utils.llm_cache import LLMCache

    cache = LLMCache(max_size=256, ttl_seconds=600)

    # 캐시 확인
    hit = cache.get("gpt-5.4", prompt_text, temperature=0.0)
    if hit is not None:
        return hit

    # LLM 호출 후 캐시 저장
    response = await llm.ainvoke(messages)
    cache.put("gpt-5.4", prompt_text, temperature=0.0, response=response_text)

    # 메트릭 확인
    print(cache.metrics)
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """캐시 히트/미스 메트릭."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """캐시 히트율 (0.0 ~ 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> dict[str, int | float]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
        }

    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


@dataclass
class _CacheEntry:
    """캐시 항목 (값 + 만료 시각)."""

    value: str
    expires_at: float  # time.monotonic() 기준


class LLMCache:
    """인메모리 LRU 캐시 (TTL 기반 만료 지원).

    스레드 안전: 내부 Lock으로 동시 접근을 보호한다.

    Args:
        max_size: 최대 캐시 항목 수 (LRU 정책으로 오래된 항목 제거)
        ttl_seconds: 캐시 항목 유효 기간 (초). 0이면 만료 없음
    """

    def __init__(self, *, max_size: int = 256, ttl_seconds: float = 600.0) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._metrics = CacheMetrics()

    @property
    def metrics(self) -> CacheMetrics:
        """캐시 메트릭 (읽기 전용 참조)."""
        return self._metrics

    @property
    def size(self) -> int:
        """현재 캐시 항목 수."""
        with self._lock:
            return len(self._store)

    @staticmethod
    def _make_key(model: str, prompt: str, temperature: float) -> str:
        """캐시 키를 생성한다.

        prompt의 SHA-256 해시를 사용하여 메모리를 절약한다.
        """
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"{model}:{prompt_hash}:{temperature}"

    def get(self, model: str, prompt: str, *, temperature: float = 0.0) -> str | None:
        """캐시에서 응답을 조회한다.

        Args:
            model: 모델 이름
            prompt: 프롬프트 텍스트
            temperature: 모델 temperature

        Returns:
            캐시된 응답 문자열 또는 None (미스/만료)
        """
        key = self._make_key(model, prompt, temperature)

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._metrics.misses += 1
                return None

            # TTL 만료 확인
            if self._ttl_seconds > 0 and time.monotonic() > entry.expires_at:
                del self._store[key]
                self._metrics.expirations += 1
                self._metrics.misses += 1
                logger.debug("캐시 항목 만료: %s", key[:40])
                return None

            # LRU: 최근 접근 항목을 끝으로 이동
            self._store.move_to_end(key)
            self._metrics.hits += 1
            return entry.value

    def put(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        response: str,
    ) -> None:
        """응답을 캐시에 저장한다.

        Args:
            model: 모델 이름
            prompt: 프롬프트 텍스트
            temperature: 모델 temperature
            response: 캐시할 응답 문자열
        """
        key = self._make_key(model, prompt, temperature)
        now = time.monotonic()
        expires_at = now + self._ttl_seconds if self._ttl_seconds > 0 else float("inf")

        with self._lock:
            # 이미 존재하면 업데이트 + LRU 끝으로 이동
            if key in self._store:
                self._store[key] = _CacheEntry(value=response, expires_at=expires_at)
                self._store.move_to_end(key)
                return

            # 용량 초과 시 가장 오래된 항목 제거
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                self._metrics.evictions += 1
                logger.debug("캐시 LRU 제거: %s", evicted_key[:40])

            self._store[key] = _CacheEntry(value=response, expires_at=expires_at)

    def invalidate(self, model: str, prompt: str, *, temperature: float = 0.0) -> bool:
        """특정 캐시 항목을 제거한다.

        Returns:
            제거 성공 여부
        """
        key = self._make_key(model, prompt, temperature)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """캐시를 완전히 비운다."""
        with self._lock:
            self._store.clear()

    def cleanup_expired(self) -> int:
        """만료된 항목들을 일괄 제거한다.

        Returns:
            제거된 항목 수
        """
        if self._ttl_seconds <= 0:
            return 0

        now = time.monotonic()
        removed = 0

        with self._lock:
            expired_keys = [k for k, v in self._store.items() if now > v.expires_at]
            for k in expired_keys:
                del self._store[k]
                self._metrics.expirations += 1
                removed += 1

        if removed:
            logger.debug("만료된 캐시 항목 %d개 제거", removed)
        return removed


# ── 전역 캐시 인스턴스 ──

_global_cache: LLMCache | None = None


def get_llm_cache(
    *,
    max_size: int = 256,
    ttl_seconds: float = 600.0,
) -> LLMCache:
    """전역 LLM 캐시 인스턴스를 반환한다 (싱글턴).

    첫 호출 시 생성되며, 이후에는 동일한 인스턴스를 반환한다.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = LLMCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _global_cache


def reset_llm_cache() -> None:
    """전역 LLM 캐시를 초기화한다 (테스트용)."""
    global _global_cache
    _global_cache = None
