"""메모리 시스템 유닛 테스트.

LLM 호출 없이 스키마, 검색, 저장소, 상태를 검증한다.
"""

from __future__ import annotations

import pytest

from youngs75_a2a.core.memory.schemas import MemoryItem, MemoryType
from youngs75_a2a.core.memory.search import (
    ContentBasedSearch,
    TagBasedSearch,
    TwoStageSearch,
    _tokenize,
)
from youngs75_a2a.core.memory.state import MemoryAwareState
from youngs75_a2a.core.memory.store import MemoryStore


# ── 헬퍼 ──


def _make_item(
    content: str,
    tags: list[str] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    session_id: str | None = None,
) -> MemoryItem:
    return MemoryItem(
        type=memory_type,
        content=content,
        tags=tags or [],
        session_id=session_id,
    )


# ── MemoryItem ──


class TestMemoryItem:
    def test_auto_id(self):
        a = _make_item("a")
        b = _make_item("b")
        assert a.id != b.id

    def test_matches_tags_overlap(self):
        item = _make_item("x", tags=["python", "testing"])
        assert item.matches_tags(["python"])
        assert item.matches_tags(["testing", "java"])
        assert not item.matches_tags(["java"])

    def test_matches_tags_empty_query(self):
        item = _make_item("x", tags=["python"])
        assert item.matches_tags([])  # 빈 쿼리 → 모두 매칭


# ── 토큰화 ──


class TestTokenize:
    def test_english(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_korean(self):
        tokens = _tokenize("프로젝트 규칙")
        assert "프로젝트" in tokens
        assert "규칙" in tokens

    def test_mixed(self):
        tokens = _tokenize("snake_case 변수명")
        assert "snake" in tokens
        assert "case" in tokens
        assert "변수명" in tokens


# ── TagBasedSearch ──


class TestTagBasedSearch:
    def test_matching_items_ranked_first(self):
        items = [
            _make_item("no match", tags=["java"]),
            _make_item("match one", tags=["python"]),
            _make_item("match two", tags=["python", "testing"]),
        ]
        results = TagBasedSearch().search("python testing", items, limit=3)
        # "python testing" 쿼리 → tags에 python+testing 둘 다 있는 항목이 1위
        assert results[0].content == "match two"
        assert results[1].content == "match one"

    def test_returns_unmatched_as_fallback(self):
        items = [_make_item("only item", tags=["java"])]
        results = TagBasedSearch().search("python", items, limit=5)
        assert len(results) == 1  # 매칭 없어도 반환

    def test_empty_query(self):
        items = [_make_item("a"), _make_item("b")]
        results = TagBasedSearch().search("", items, limit=5)
        assert len(results) == 2


# ── ContentBasedSearch ──


class TestContentBasedSearch:
    def test_relevant_content_ranked_higher(self):
        items = [
            _make_item("이 프로젝트는 Java로 작성됨"),
            _make_item("이 프로젝트는 Python snake_case 컨벤션 사용"),
            _make_item("데이터베이스 접속 설정"),
        ]
        results = ContentBasedSearch().search("python 컨벤션", items, limit=3)
        assert results[0].content.startswith("이 프로젝트는 Python")

    def test_score_assigned(self):
        items = [_make_item("python coding agent")]
        results = ContentBasedSearch().search("python", items, limit=1)
        assert results[0].score > 0

    def test_empty_candidates(self):
        assert ContentBasedSearch().search("query", [], limit=5) == []


# ── TwoStageSearch ──


class TestTwoStageSearch:
    def test_combined_search(self):
        items = [
            _make_item("Java 스타일 가이드", tags=["java", "style"]),
            _make_item("Python snake_case 컨벤션", tags=["python", "convention"]),
            _make_item("Python 타입 힌트 가이드", tags=["python", "typing"]),
            _make_item("DB 커넥션 풀 설정", tags=["database"]),
        ]
        searcher = TwoStageSearch(tag_limit=10, final_limit=2)
        results = searcher.search("python convention", items)
        assert len(results) <= 2
        # Python 관련 항목이 상위
        assert all("Python" in r.content or "python" in str(r.tags) for r in results)

    def test_respects_final_limit(self):
        items = [_make_item(f"item {i}", tags=["common"]) for i in range(20)]
        results = TwoStageSearch(tag_limit=50, final_limit=5).search("common", items)
        assert len(results) <= 5


# ── MemoryStore ──


class TestMemoryStore:
    @pytest.fixture()
    def store(self):
        return MemoryStore()

    def test_put_and_get(self, store):
        item = _make_item("test content")
        store.put(item)
        retrieved = store.get(item.id, MemoryType.SEMANTIC)
        assert retrieved is not None
        assert retrieved.content == "test content"

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent", MemoryType.SEMANTIC) is None

    def test_list_by_type(self, store):
        store.put(_make_item("semantic 1"))
        store.put(_make_item("semantic 2"))
        store.put(_make_item("episodic 1", memory_type=MemoryType.EPISODIC))
        assert len(store.list_by_type(MemoryType.SEMANTIC)) == 2
        assert len(store.list_by_type(MemoryType.EPISODIC)) == 1

    def test_search(self, store):
        store.put(_make_item("Python 코딩 컨벤션", tags=["python"]))
        store.put(_make_item("Java 스타일 가이드", tags=["java"]))
        store.put(_make_item("DB 설정 방법", tags=["database"]))
        results = store.search("python convention", limit=2)
        assert len(results) <= 2
        assert any("Python" in r.content for r in results)

    def test_delete(self, store):
        item = _make_item("to delete")
        store.put(item)
        assert store.delete(item.id, MemoryType.SEMANTIC)
        assert store.get(item.id, MemoryType.SEMANTIC) is None

    def test_delete_nonexistent(self, store):
        assert not store.delete("nonexistent", MemoryType.SEMANTIC)

    def test_clear_all(self, store):
        store.put(_make_item("a"))
        store.put(_make_item("b", memory_type=MemoryType.EPISODIC))
        count = store.clear()
        assert count == 2
        assert store.total_count == 0

    def test_clear_by_type(self, store):
        store.put(_make_item("semantic"))
        store.put(_make_item("episodic", memory_type=MemoryType.EPISODIC))
        count = store.clear(MemoryType.SEMANTIC)
        assert count == 1
        assert store.total_count == 1

    def test_session_scoped(self, store):
        store.put(_make_item("session A", memory_type=MemoryType.EPISODIC, session_id="s1"))
        store.put(_make_item("session B", memory_type=MemoryType.EPISODIC, session_id="s2"))
        items = store.list_by_type(MemoryType.EPISODIC, session_id="s1")
        assert len(items) == 1
        assert items[0].content == "session A"

    def test_total_count(self, store):
        assert store.total_count == 0
        store.put(_make_item("a"))
        store.put(_make_item("b"))
        assert store.total_count == 2


# ── MemoryAwareState ──


class TestMemoryAwareState:
    def test_state_type_annotations(self):
        """MemoryAwareState 필드가 올바르게 정의되어 있는지 확인."""
        annotations = MemoryAwareState.__annotations__
        assert "messages" in annotations
        assert "semantic_context" in annotations
        assert "episodic_log" in annotations


# ── core import 통합 ──


class TestCoreImport:
    def test_memory_exports_from_core(self):
        from youngs75_a2a.core import (
            MemoryAwareState,
            MemoryItem,
            MemoryStore,
            MemoryType,
            TwoStageSearch,
        )

        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryStore is not None
