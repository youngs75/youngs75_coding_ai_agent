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
from youngs75_a2a.core.memory.search import TwoStageSearch, _tokenize


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

    # ── Procedural Memory (Voyager 패턴) ──────────────────────

    def accumulate_skill(
        self,
        code: str,
        description: str,
        tags: list[str] | None = None,
        *,
        novelty_threshold: float = 0.7,
    ) -> MemoryItem | None:
        """성공적인 코드 패턴을 Procedural Memory로 누적 저장한다.

        Voyager 패턴: 실행 성공 시 코드 패턴을 추출하여 저장하되,
        기존 스킬과 유사도가 높으면 중복으로 판단하여 저장하지 않는다.

        Args:
            code: 저장할 코드 패턴
            description: 스킬 설명
            tags: 스킬 분류 태그
            novelty_threshold: 중복 판단 임계값 (0~1, 높을수록 엄격)

        Returns:
            저장된 MemoryItem 또는 중복으로 판단 시 None
        """
        if not code or not code.strip():
            return None

        content = f"{description}\n\n```\n{code}\n```"

        # Novelty 필터링: 기존 procedural 메모리와 유사도 검사
        if not self._is_novel(content, novelty_threshold):
            return None

        item = MemoryItem(
            type=MemoryType.PROCEDURAL,
            content=content,
            tags=tags or [],
            metadata={"code": code, "description": description},
        )
        self.put(item)
        return item

    def retrieve_skills(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """Procedural Memory에서 관련 스킬을 검색한다.

        Args:
            query: 검색 쿼리 (작업 설명 등)
            tags: 태그 필터
            limit: 최대 반환 수
        """
        return self.search(
            query,
            memory_type=MemoryType.PROCEDURAL,
            tags=tags,
            limit=limit,
        )

    def _is_novel(self, content: str, threshold: float) -> bool:
        """기존 procedural 메모리 대비 새로운 패턴인지 판단한다.

        토큰 기반 Jaccard 유사도를 사용하여 중복을 판단한다.
        """
        existing = self.list_by_type(MemoryType.PROCEDURAL)
        if not existing:
            return True

        new_tokens = set(_tokenize(content))
        if not new_tokens:
            return False

        for item in existing:
            existing_tokens = set(_tokenize(item.content))
            if not existing_tokens:
                continue
            # Jaccard 유사도
            intersection = len(new_tokens & existing_tokens)
            union = len(new_tokens | existing_tokens)
            similarity = intersection / union if union > 0 else 0.0
            if similarity >= threshold:
                return False
        return True

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
