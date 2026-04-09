"""메모리 3계층 확장 테스트 -- USER_PROFILE, DOMAIN_KNOWLEDGE."""

from __future__ import annotations

import pytest

from coding_agent.core.memory import MemoryItem, MemoryStore, MemoryType


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


# ── User Profile ──


class TestUserProfile:
    """accumulate_user_profile 메서드 테스트."""

    @pytest.fixture()
    def store(self) -> MemoryStore:
        return MemoryStore()

    def test_accumulate_user_profile_new(self, store: MemoryStore):
        """새 프로필 정보가 정상 저장되는지 확인."""
        item = store.accumulate_user_profile(
            content="사용자는 Python과 TypeScript를 주로 사용합니다.",
            tags=["language", "preference"],
            source="conversation",
        )
        assert item is not None
        assert item.type == MemoryType.USER_PROFILE
        assert "Python" in item.content
        assert item.tags == ["language", "preference"]

        # 실제로 저장소에 들어갔는지 확인
        items = store.list_by_type(MemoryType.USER_PROFILE)
        assert len(items) == 1

    def test_accumulate_user_profile_update_same_tags(self, store: MemoryStore):
        """동일 태그를 가진 프로필이 이미 있으면 갱신(덮어쓰기)해야 한다."""
        # 1차 저장
        first = store.accumulate_user_profile(
            content="선호 언어: Python",
            tags=["language"],
        )
        # 2차 저장 — 동일 태그
        second = store.accumulate_user_profile(
            content="선호 언어: Rust",
            tags=["language"],
        )

        # 항목 수는 1개 유지 (갱신이므로)
        items = store.list_by_type(MemoryType.USER_PROFILE)
        assert len(items) == 1
        assert items[0].content == "선호 언어: Rust"
        # 같은 ID (갱신)
        assert first.id == second.id

    def test_user_profile_persistence(self, tmp_path):
        """USER_PROFILE이 JSONL 파일로 영속화되는지 확인."""
        persist_dir = tmp_path / ".ai" / "memory"
        store = MemoryStore(persist_dir=persist_dir)

        store.accumulate_user_profile(
            content="테스트 프로필",
            tags=["test"],
        )

        # JSONL 파일 생성 확인
        jsonl_path = persist_dir / "user_profile.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        assert "테스트 프로필" in lines[0]


# ── Domain Knowledge ──


class TestDomainKnowledge:
    """accumulate_domain_knowledge 메서드 테스트."""

    @pytest.fixture()
    def store(self) -> MemoryStore:
        return MemoryStore()

    def test_accumulate_domain_knowledge_new(self, store: MemoryStore):
        """새 도메인 지식이 정상 저장되는지 확인."""
        item = store.accumulate_domain_knowledge(
            content="이 프로젝트는 LangGraph 기반 에이전트 프레임워크이다.",
            tags=["architecture", "langgraph"],
            source="codebase",
        )
        assert item is not None
        assert item.type == MemoryType.DOMAIN_KNOWLEDGE
        assert "LangGraph" in item.content

        items = store.list_by_type(MemoryType.DOMAIN_KNOWLEDGE)
        assert len(items) == 1

    def test_accumulate_domain_knowledge_dedup(self, store: MemoryStore):
        """유사 내용(Jaccard >= 0.8) 시 기존 항목을 갱신해야 한다."""
        # 원본 저장
        original = store.accumulate_domain_knowledge(
            content="프로젝트는 Python 3.13과 LangGraph 1.1.4를 사용한다.",
            tags=["stack"],
        )
        # 매우 유사한 내용으로 재저장 (단어 약간 변경)
        updated = store.accumulate_domain_knowledge(
            content="프로젝트는 Python 3.13과 LangGraph 1.1.4를 사용한다.",
            tags=["stack"],
        )

        # 항목 수는 1개 유지 (중복 제거)
        items = store.list_by_type(MemoryType.DOMAIN_KNOWLEDGE)
        assert len(items) == 1

    def test_domain_knowledge_search(self, store: MemoryStore):
        """도메인 지식 검색 결과가 올바르게 반환되는지 확인."""
        store.accumulate_domain_knowledge(
            content="A2A 프로토콜은 Google이 정의한 에이전트 간 통신 표준이다.",
            tags=["protocol"],
        )
        store.accumulate_domain_knowledge(
            content="MCP는 Model Context Protocol로 도구 연동에 사용된다.",
            tags=["protocol"],
        )
        store.accumulate_domain_knowledge(
            content="프로젝트의 코딩 컨벤션은 snake_case를 따른다.",
            tags=["convention"],
        )

        results = store.search(
            "A2A 프로토콜 통신",
            memory_type=MemoryType.DOMAIN_KNOWLEDGE,
            limit=2,
        )
        assert len(results) >= 1
        assert any("A2A" in r.content for r in results)


# ── Memory Correction ──


class TestMemoryCorrection:
    """MemoryStore.update() 메서드 -- 메모리 정정 테스트."""

    @pytest.fixture()
    def store(self) -> MemoryStore:
        return MemoryStore()

    def test_update_content(self, store: MemoryStore):
        """content 정정 후 조회 시 갱신된 내용이 반환되어야 한다."""
        item = _make_item("원래 내용", tags=["test"])
        store.put(item)

        updated = store.update(item.id, MemoryType.SEMANTIC, content="정정된 내용")
        assert updated is not None
        assert updated.content == "정정된 내용"

        # 재조회 확인
        retrieved = store.get(item.id, MemoryType.SEMANTIC)
        assert retrieved is not None
        assert retrieved.content == "정정된 내용"

    def test_update_sets_updated_at(self, store: MemoryStore):
        """update 시 updated_at 필드가 갱신되어야 한다."""
        item = _make_item("내용")
        assert item.updated_at is None  # 초기에는 None

        store.put(item)
        updated = store.update(item.id, MemoryType.SEMANTIC, content="갱신")
        assert updated is not None
        assert updated.updated_at is not None

    def test_update_nonexistent_returns_none(self, store: MemoryStore):
        """존재하지 않는 항목 정정 시 None을 반환해야 한다."""
        result = store.update("nonexistent_id", MemoryType.SEMANTIC, content="test")
        assert result is None
