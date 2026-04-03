"""GAM 이중 구조 메모리 검색.

2단계 검색:
  1단계 — 태그 기반 경량 필터링 (후보 축소)
  2단계 — BM25 컨텐츠 기반 심화 순위 (정밀 랭킹)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter

from youngs75_a2a.core.memory.schemas import MemoryItem


def _tokenize(text: str) -> list[str]:
    """간단한 토큰화: 소문자 변환 + 비알파벳 분리."""
    return re.findall(r"[a-z0-9가-힣]+", text.lower())


class MemorySearchStrategy(ABC):
    """메모리 검색 전략 인터페이스."""

    @abstractmethod
    def search(
        self, query: str, candidates: list[MemoryItem], limit: int
    ) -> list[MemoryItem]:
        ...


class TagBasedSearch(MemorySearchStrategy):
    """1단계: 태그 기반 경량 검색.

    쿼리를 토큰화하여 각 메모리 항목의 태그와 매칭하고,
    매칭된 태그 수 기준으로 순위를 매긴다.
    """

    def search(
        self, query: str, candidates: list[MemoryItem], limit: int
    ) -> list[MemoryItem]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return candidates[:limit]

        scored: list[tuple[float, MemoryItem]] = []
        for item in candidates:
            tag_tokens = set()
            for tag in item.tags:
                tag_tokens.update(_tokenize(tag))
            overlap = len(query_tokens & tag_tokens)
            if overlap > 0:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 태그 매칭이 없는 항목도 뒤에 붙여 최소 결과 보장
        matched_ids = {item.id for _, item in scored}
        unmatched = [item for item in candidates if item.id not in matched_ids]

        results = [item for _, item in scored] + unmatched
        return results[:limit]


class ContentBasedSearch(MemorySearchStrategy):
    """2단계: BM25 컨텐츠 기반 심화 검색.

    간소화된 BM25 스코어링으로 content 필드를 검색한다.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b

    def search(
        self, query: str, candidates: list[MemoryItem], limit: int
    ) -> list[MemoryItem]:
        query_tokens = _tokenize(query)
        if not query_tokens or not candidates:
            return candidates[:limit]

        # 문서별 토큰 및 평균 길이 계산
        doc_tokens_list = [_tokenize(item.content) for item in candidates]
        avg_dl = sum(len(dt) for dt in doc_tokens_list) / max(len(candidates), 1)

        scored: list[tuple[float, MemoryItem]] = []
        for item, doc_tokens in zip(candidates, doc_tokens_list):
            score = self._bm25_score(query_tokens, doc_tokens, avg_dl)
            item_copy = item.model_copy(update={"score": score})
            scored.append((score, item_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def _bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        avg_dl: float,
    ) -> float:
        dl = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            tf = tf_map.get(term, 0)
            if tf > 0:
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * dl / max(avg_dl, 1)
                )
                score += numerator / denominator
        return score


class TwoStageSearch:
    """GAM 이중 구조: 태그 기반 필터링 → BM25 컨텐츠 순위.

    Args:
        tag_limit: 1단계에서 통과시킬 최대 후보 수
        final_limit: 최종 반환 결과 수
    """

    def __init__(self, tag_limit: int = 50, final_limit: int = 10):
        self.tag_search = TagBasedSearch()
        self.content_search = ContentBasedSearch()
        self.tag_limit = tag_limit
        self.final_limit = final_limit

    def search(self, query: str, candidates: list[MemoryItem]) -> list[MemoryItem]:
        """2단계 검색 실행."""
        # 1단계: 태그 기반 후보 축소
        tag_results = self.tag_search.search(query, candidates, self.tag_limit)
        # 2단계: BM25 컨텐츠 순위
        return self.content_search.search(query, tag_results, self.final_limit)
