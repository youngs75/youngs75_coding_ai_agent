"""메모리 저장소.

LangGraph InMemoryStore를 래핑하여 MemoryItem CRUD와
GAM 2단계 검색을 제공한다.

네임스페이스 구조:
  ("memory", <memory_type>)           — 타입별 기본 네임스페이스
  ("memory", <memory_type>, <session>) — 세션 스코프 (episodic 등)
"""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore

from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.memory.search import TwoStageSearch


def _namespace(memory_type: MemoryType, session_id: str | None = None) -> tuple[str, ...]:
    base = ("memory", memory_type.value)
    if session_id:
        return (*base, session_id)
    return base


class MemoryStore:
    """메모리 저장소 — InMemoryStore 래핑 + 2단계 검색."""

    def __init__(
        self,
        store: InMemoryStore | None = None,
        search: TwoStageSearch | None = None,
    ):
        self._store = store or InMemoryStore()
        self._search = search or TwoStageSearch()
        # 타입별 메모리 인덱스 (빠른 조회용)
        self._index: dict[tuple[str, ...], dict[str, MemoryItem]] = {}

    def put(self, item: MemoryItem) -> None:
        """메모리 항목 저장."""
        ns = _namespace(item.type, item.session_id)
        if ns not in self._index:
            self._index[ns] = {}
        self._index[ns][item.id] = item

    def get(self, item_id: str, memory_type: MemoryType, session_id: str | None = None) -> MemoryItem | None:
        """ID로 메모리 항목 조회."""
        ns = _namespace(memory_type, session_id)
        bucket = self._index.get(ns, {})
        return bucket.get(item_id)

    def search(
        self,
        query: str,
        *,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """2단계 검색: 태그 기반 필터링 → BM25 컨텐츠 순위.

        Args:
            query: 검색 쿼리
            memory_type: 특정 타입만 검색 (None이면 전체)
            tags: 태그 사전 필터링
            session_id: 세션 스코프 필터
            limit: 최대 반환 수
        """
        candidates = self._collect_candidates(memory_type, session_id)

        # 태그 사전 필터링
        if tags:
            candidates = [c for c in candidates if c.matches_tags(tags)]

        searcher = TwoStageSearch(
            tag_limit=min(50, len(candidates)),
            final_limit=limit,
        )
        return searcher.search(query, candidates)

    def list_by_type(
        self,
        memory_type: MemoryType,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        """타입별 메모리 항목 목록."""
        ns = _namespace(memory_type, session_id)
        bucket = self._index.get(ns, {})
        return sorted(bucket.values(), key=lambda x: x.created_at, reverse=True)

    def delete(self, item_id: str, memory_type: MemoryType, session_id: str | None = None) -> bool:
        """메모리 항목 삭제. 성공 시 True."""
        ns = _namespace(memory_type, session_id)
        bucket = self._index.get(ns, {})
        if item_id in bucket:
            del bucket[item_id]
            return True
        return False

    def clear(self, memory_type: MemoryType | None = None) -> int:
        """메모리 초기화. 삭제된 항목 수 반환."""
        count = 0
        if memory_type is None:
            for bucket in self._index.values():
                count += len(bucket)
            self._index.clear()
        else:
            to_remove = [ns for ns in self._index if ns[1] == memory_type.value]
            for ns in to_remove:
                count += len(self._index[ns])
                del self._index[ns]
        return count

    @property
    def total_count(self) -> int:
        """저장된 전체 메모리 항목 수."""
        return sum(len(bucket) for bucket in self._index.values())

    def _collect_candidates(
        self,
        memory_type: MemoryType | None,
        session_id: str | None,
    ) -> list[MemoryItem]:
        """검색 후보 수집."""
        candidates: list[MemoryItem] = []
        for ns, bucket in self._index.items():
            if memory_type and ns[1] != memory_type.value:
                continue
            if session_id and len(ns) > 2 and ns[2] != session_id:
                continue
            candidates.extend(bucket.values())
        return candidates
