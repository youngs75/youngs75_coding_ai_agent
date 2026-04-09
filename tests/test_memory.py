"""메모리 시스템 유닛 테스트.

LLM 호출 없이 스키마, 검색, 저장소, 상태를 검증한다.
"""

from __future__ import annotations

import pytest

from coding_agent.core.memory.schemas import MemoryItem, MemoryType
from coding_agent.core.memory.search import (
    ContentBasedSearch,
    TagBasedSearch,
    TwoStageSearch,
    _tokenize,
)
from coding_agent.core.memory.state import MemoryAwareState
from coding_agent.core.memory.store import MemoryStore


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
        store.put(
            _make_item("session A", memory_type=MemoryType.EPISODIC, session_id="s1")
        )
        store.put(
            _make_item("session B", memory_type=MemoryType.EPISODIC, session_id="s2")
        )
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


# ── Procedural Memory (Voyager 패턴) ──


class TestAccumulateSkill:
    """accumulate_skill 메서드 — Voyager 패턴 스킬 누적."""

    @pytest.fixture()
    def store(self):
        return MemoryStore()

    def test_accumulate_new_skill(self, store):
        """새로운 스킬이 정상 저장되는지 확인."""
        item = store.accumulate_skill(
            code="def fibonacci(n): ...",
            description="피보나치 함수 생성",
            tags=["generate", "python"],
        )
        assert item is not None
        assert item.type == MemoryType.PROCEDURAL
        assert "fibonacci" in item.content
        assert "피보나치" in item.content
        assert item.metadata["code"] == "def fibonacci(n): ..."
        assert item.metadata["description"] == "피보나치 함수 생성"

    def test_accumulate_stores_in_procedural_namespace(self, store):
        """저장된 스킬이 PROCEDURAL 타입으로 조회 가능한지 확인."""
        store.accumulate_skill(code="x = 1", description="간단한 할당")
        items = store.list_by_type(MemoryType.PROCEDURAL)
        assert len(items) == 1

    def test_novelty_filter_blocks_duplicate(self, store):
        """동일한 코드 패턴은 novelty 필터에 의해 차단."""
        store.accumulate_skill(code="def hello(): print('hi')", description="인사 함수")
        duplicate = store.accumulate_skill(
            code="def hello(): print('hi')", description="인사 함수"
        )
        assert duplicate is None
        assert len(store.list_by_type(MemoryType.PROCEDURAL)) == 1

    def test_novelty_filter_allows_different_skill(self, store):
        """충분히 다른 코드 패턴은 통과."""
        store.accumulate_skill(
            code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            description="피보나치 재귀",
        )
        item = store.accumulate_skill(
            code="class DatabaseConnection:\n    def connect(self): ...\n    def query(self, sql): ...",
            description="DB 연결 클래스",
        )
        assert item is not None
        assert len(store.list_by_type(MemoryType.PROCEDURAL)) == 2

    def test_novelty_threshold_adjustable(self, store):
        """novelty_threshold 조정으로 필터 강도 변경."""
        store.accumulate_skill(code="print('hello world')", description="출력 1")
        # 낮은 임계값으로 유사한 패턴도 통과
        item = store.accumulate_skill(
            code="print('hello everyone')",
            description="출력 2",
            novelty_threshold=0.95,
        )
        assert item is not None

    def test_retrieve_skills(self, store):
        """retrieve_skills로 관련 스킬 검색."""
        store.accumulate_skill(
            code="def sort_list(lst): return sorted(lst)",
            description="리스트 정렬 함수",
            tags=["generate", "python"],
        )
        store.accumulate_skill(
            code="class APIClient:\n    def get(self, url): ...",
            description="API 클라이언트 클래스",
            tags=["generate", "python"],
        )
        results = store.retrieve_skills("정렬 sort", limit=1)
        assert len(results) == 1
        assert "sort" in results[0].content

    def test_retrieve_skills_with_tags(self, store):
        """태그 기반 스킬 검색 필터링."""
        store.accumulate_skill(
            code="console.log('hi')",
            description="JS 로깅",
            tags=["generate", "javascript"],
        )
        store.accumulate_skill(
            code="print('hi')",
            description="Python 출력",
            tags=["generate", "python"],
        )
        results = store.retrieve_skills("출력", tags=["python"])
        assert all("python" in r.tags for r in results)

    def test_empty_code_not_accumulated(self, store):
        """빈 코드는 저장하지 않음 (novelty 필터에서 차단)."""
        item = store.accumulate_skill(code="", description="빈 코드")
        assert item is None


class TestIsNovel:
    """_is_novel 내부 메서드 — Jaccard 유사도 기반 중복 판단."""

    def test_first_item_always_novel(self):
        store = MemoryStore()
        assert store._is_novel("완전히 새로운 콘텐츠", threshold=0.7) is True

    def test_identical_content_not_novel(self):
        store = MemoryStore()
        store.put(
            _make_item(
                "def foo(): pass",
                memory_type=MemoryType.PROCEDURAL,
            )
        )
        assert store._is_novel("def foo(): pass", threshold=0.7) is False

    def test_different_content_is_novel(self):
        store = MemoryStore()
        store.put(
            _make_item(
                "def fibonacci(n): return n",
                memory_type=MemoryType.PROCEDURAL,
            )
        )
        assert (
            store._is_novel(
                "class DatabaseConnection:\n    def connect(self): ...",
                threshold=0.7,
            )
            is True
        )


# ── MemoryAwareState procedural ──


class TestMemoryAwareStateProcedural:
    def test_procedural_context_field_exists(self):
        """MemoryAwareState에 procedural_context 필드가 정의되어 있는지 확인."""
        annotations = MemoryAwareState.__annotations__
        assert "procedural_context" in annotations


# ── CodingState procedural_skills ──


class TestCodingStateProcedural:
    def test_procedural_skills_field_exists(self):
        """CodingState에 procedural_skills 필드가 정의되어 있는지 확인."""
        from coding_agent.agents.coding_assistant.schemas import CodingState

        annotations = CodingState.__annotations__
        assert "procedural_skills" in annotations


# ── core import 통합 ──


class TestCoreImport:
    def test_memory_exports_from_core(self):
        from coding_agent.core import (
            MemoryStore,
            MemoryType,
        )

        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryStore is not None


# ── SemanticMemoryLoader ──


class TestSemanticMemoryLoader:
    """SemanticMemoryLoader — AGENTS.md/pyproject.toml에서 자동 로딩."""

    @pytest.fixture()
    def store(self):
        return MemoryStore()

    @pytest.fixture()
    def workspace(self, tmp_path):
        return tmp_path

    def test_load_from_agents_md(self, store, workspace):
        """AGENTS.md에서 핵심 섹션을 추출하여 Semantic Memory에 저장."""
        agents_md = workspace / "AGENTS.md"
        agents_md.write_text(
            "# Repo\n\n"
            "## 커뮤니케이션 규칙\n한국어로 소통합니다.\n\n"
            "## 주요 기술 스택\n- LangGraph\n- MCP\n\n"
            "## 프로젝트 구조\n```\nsrc/\n```\n",
            encoding="utf-8",
        )
        from coding_agent.core.memory.semantic_loader import SemanticMemoryLoader

        loader = SemanticMemoryLoader(workspace=workspace, store=store)
        count = loader._load_from_agents_md()
        assert count == 3
        items = store.list_by_type(MemoryType.SEMANTIC)
        assert len(items) == 3
        contents = [i.content for i in items]
        assert any("커뮤니케이션" in c for c in contents)
        assert any("기술 스택" in c for c in contents)
        assert any("프로젝트 구조" in c for c in contents)

    def test_load_from_pyproject(self, store, workspace):
        """pyproject.toml에서 메타데이터를 추출."""
        pyproject = workspace / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "test-project"\nversion = "1.0.0"\n'
            'requires-python = ">=3.11"\n'
            'dependencies = [\n  "langchain>=1.2",\n  "pydantic>=2.0",\n]\n',
            encoding="utf-8",
        )
        from coding_agent.core.memory.semantic_loader import SemanticMemoryLoader

        loader = SemanticMemoryLoader(workspace=workspace, store=store)
        count = loader._load_from_pyproject()
        assert count == 2  # meta + deps
        items = store.list_by_type(MemoryType.SEMANTIC)
        contents = [i.content for i in items]
        assert any("test-project" in c for c in contents)
        assert any("langchain" in c for c in contents)

    def test_load_all_clears_existing(self, store, workspace):
        """load_all은 기존 semantic 메모리를 초기화하고 재로딩."""
        store.put(_make_item("기존 메모리"))
        assert store.total_count == 1

        (workspace / "AGENTS.md").write_text(
            "# Repo\n\n## 커밋 및 PR 규칙\nConventional Commits\n",
            encoding="utf-8",
        )
        from coding_agent.core.memory.semantic_loader import SemanticMemoryLoader

        loader = SemanticMemoryLoader(workspace=workspace, store=store)
        count = loader.load_all()
        assert count == 1
        assert store.total_count == 1

    def test_load_all_no_files(self, store, workspace):
        """AGENTS.md도 pyproject.toml도 없을 때 0건."""
        from coding_agent.core.memory.semantic_loader import SemanticMemoryLoader

        loader = SemanticMemoryLoader(workspace=workspace, store=store)
        count = loader.load_all()
        assert count == 0

    def test_real_workspace(self, store):
        """실제 프로젝트 루트에서 로딩 (통합 테스트)."""
        import pathlib

        project_root = pathlib.Path(__file__).resolve().parent.parent
        if not (project_root / "AGENTS.md").exists():
            pytest.skip("AGENTS.md not found")

        from coding_agent.core.memory.semantic_loader import SemanticMemoryLoader

        loader = SemanticMemoryLoader(workspace=project_root, store=store)
        count = loader.load_all()
        assert count >= 3  # AGENTS.md에서 최소 3개 섹션 + pyproject 메타


# ── DeepResearch 메모리 상태 필드 ──


class TestDeepResearchMemoryState:
    def test_agent_state_has_memory_fields(self):
        """AgentState에 semantic_context, episodic_log 필드 존재 확인."""
        from coding_agent.agents.deep_research.schemas import AgentState

        annotations = AgentState.__annotations__
        assert "semantic_context" in annotations
        assert "episodic_log" in annotations

    def test_hitl_state_inherits_memory_fields(self):
        """HITLAgentState가 AgentState의 메모리 필드를 상속."""
        from coding_agent.agents.deep_research.schemas import HITLAgentState

        # 직접 또는 상속으로 보유
        all_fields = {}
        for cls in HITLAgentState.__mro__:
            if hasattr(cls, "__annotations__"):
                all_fields.update(cls.__annotations__)
        assert "semantic_context" in all_fields
        assert "episodic_log" in all_fields
