"""메모리 저장소.

LangGraph InMemoryStore를 래핑하여 MemoryItem CRUD와
GAM 2단계 검색을 제공한다.

네임스페이스 구조:
  ("memory", <memory_type>)           — 타입별 기본 네임스페이스
  ("memory", <memory_type>, <session>) — 세션 스코프 (episodic 등)

영속화:
  Procedural, USER_PROFILE, DOMAIN_KNOWLEDGE, SEMANTIC Memory는
  JSONL 파일로 영속화되어 세션 간 학습이 유지된다.
  저장 경로: {workspace}/.ai/memory/{type}.jsonl
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.store.memory import InMemoryStore

from coding_agent.core.memory.schemas import MemoryItem, MemoryType
from coding_agent.core.memory.search import TwoStageSearch, _tokenize

logger = logging.getLogger(__name__)


def _namespace(
    memory_type: MemoryType, session_id: str | None = None
) -> tuple[str, ...]:
    base = ("memory", memory_type.value)
    if session_id:
        return (*base, session_id)
    return base


_PERSISTENT_TYPES = {
    MemoryType.PROCEDURAL,
    MemoryType.USER_PROFILE,
    MemoryType.DOMAIN_KNOWLEDGE,
    MemoryType.SEMANTIC,
}


class MemoryStore:
    """메모리 저장소 — InMemoryStore 래핑 + 2단계 검색 + 파일 영속화.

    LangGraph InMemoryStore를 래핑하여 MemoryItem CRUD,
    GAM 2단계 검색(태그 필터 → BM25 랭킹), Voyager 패턴 스킬 축적,
    도메인 지식/사용자 프로필 자동 중복 제거를 제공한다.

    영속화 대상 타입(PROCEDURAL, USER_PROFILE, DOMAIN_KNOWLEDGE, SEMANTIC)은
    JSONL 파일로 자동 저장되어 세션 간 학습이 유지된다.

    Args:
        store: 내부 InMemoryStore 인스턴스. None이면 새로 생성.
        search: 2단계 검색 인스턴스. None이면 기본 설정으로 생성.
        persist_dir: 영속화 디렉토리 경로. None이면 영속화 비활성화.
    """

    def __init__(
        self,
        store: InMemoryStore | None = None,
        search: TwoStageSearch | None = None,
        persist_dir: str | Path | None = None,
    ):
        self._store = store or InMemoryStore()
        self._search = search or TwoStageSearch()
        # 타입별 메모리 인덱스 (빠른 조회용)
        self._index: dict[tuple[str, ...], dict[str, MemoryItem]] = {}

        # 영속화 경로 설정
        self._persist_dir: Path | None = None
        if persist_dir:
            self._persist_dir = Path(persist_dir)
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted()

    def put(self, item: MemoryItem) -> None:
        """메모리 항목을 저장한다.

        영속화 대상 타입은 JSONL 파일에도 append한다.

        Args:
            item: 저장할 MemoryItem.
        """
        ns = _namespace(item.type, item.session_id)
        if ns not in self._index:
            self._index[ns] = {}
        self._index[ns][item.id] = item

        # 영속화 대상 타입은 파일에 추가
        if item.type in _PERSISTENT_TYPES and self._persist_dir:
            self._append_to_file(item)

    def get(
        self, item_id: str, memory_type: MemoryType, session_id: str | None = None
    ) -> MemoryItem | None:
        """ID로 메모리 항목을 조회한다.

        Args:
            item_id: 조회할 항목 ID.
            memory_type: 메모리 타입.
            session_id: 세션 스코프 ID. None이면 전역 네임스페이스.

        Returns:
            MemoryItem 또는 None (미존재 시).
        """
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
        """타입별 메모리 항목 목록을 반환한다.

        Args:
            memory_type: 조회할 메모리 타입.
            session_id: 세션 스코프 ID. None이면 전역 네임스페이스.

        Returns:
            created_at 역순으로 정렬된 MemoryItem 리스트.
        """
        ns = _namespace(memory_type, session_id)
        bucket = self._index.get(ns, {})
        return sorted(bucket.values(), key=lambda x: x.created_at, reverse=True)

    def delete(
        self, item_id: str, memory_type: MemoryType, session_id: str | None = None
    ) -> bool:
        """메모리 항목을 삭제한다.

        Args:
            item_id: 삭제할 항목 ID.
            memory_type: 메모리 타입.
            session_id: 세션 스코프 ID.

        Returns:
            삭제 성공 시 True, 항목 미존재 시 False.
        """
        ns = _namespace(memory_type, session_id)
        bucket = self._index.get(ns, {})
        if item_id in bucket:
            del bucket[item_id]
            return True
        return False

    def update(
        self,
        item_id: str,
        memory_type: MemoryType,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> MemoryItem | None:
        """메모리 항목을 정정(correction)한다.

        Args:
            item_id: 갱신할 항목 ID.
            memory_type: 메모리 타입.
            content: 새 콘텐츠. None이면 기존 값 유지.
            tags: 새 태그 리스트. None이면 기존 값 유지.
            metadata: 새 메타데이터. None이면 기존 값 유지.
            session_id: 세션 스코프 ID.

        Returns:
            갱신된 MemoryItem 또는 None (항목 미존재 시).
        """
        existing = self.get(item_id, memory_type, session_id)
        if existing is None:
            return None

        updates: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if content is not None:
            updates["content"] = content
        if tags is not None:
            updates["tags"] = tags
        if metadata is not None:
            updates["metadata"] = metadata

        updated_item = existing.model_copy(update=updates)
        ns = _namespace(memory_type, session_id)
        self._index[ns][item_id] = updated_item

        # 영속화 대상이면 파일 전체 재작성
        if memory_type in _PERSISTENT_TYPES and self._persist_dir:
            self._rewrite_file(memory_type)

        logger.debug("메모리 갱신: %s (type=%s)", item_id[:8], memory_type.value)
        return updated_item

    def clear(self, memory_type: MemoryType | None = None) -> int:
        """메모리를 초기화한다.

        Args:
            memory_type: 초기화할 타입. None이면 전체 초기화.

        Returns:
            삭제된 항목 수.
        """
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

    # ── Domain Knowledge ──────────────────────────────────────

    def accumulate_domain_knowledge(
        self,
        content: str,
        tags: list[str] | None = None,
        source: str = "user",
    ) -> MemoryItem:
        """도메인 지식을 저장한다. 중복이면 기존 항목을 갱신."""
        existing = self.list_by_type(MemoryType.DOMAIN_KNOWLEDGE)
        new_tokens = set(_tokenize(content))

        # 기존 도메인 지식과 Jaccard 유사도 0.8 이상이면 갱신
        if new_tokens:
            for item in existing:
                existing_tokens = set(_tokenize(item.content))
                if not existing_tokens:
                    continue
                intersection = len(new_tokens & existing_tokens)
                union = len(new_tokens | existing_tokens)
                similarity = intersection / union if union > 0 else 0.0
                if similarity >= 0.8:
                    updated = self.update(
                        item.id,
                        MemoryType.DOMAIN_KNOWLEDGE,
                        content=content,
                        tags=tags if tags is not None else None,
                        metadata={"source": source},
                    )
                    if updated is not None:
                        return updated

        # 유사 항목 없으면 새로 저장
        item = MemoryItem(
            type=MemoryType.DOMAIN_KNOWLEDGE,
            content=content,
            tags=tags or [],
            source=source,
            metadata={"source": source},
        )
        self.put(item)
        return item

    # ── User Profile ─────────────────────────────────────────

    def accumulate_user_profile(
        self,
        content: str,
        tags: list[str] | None = None,
        source: str = "user",
    ) -> MemoryItem:
        """사용자 프로필 정보를 저장한다. 동일 태그의 기존 항목은 갱신."""
        if tags:
            existing = self.list_by_type(MemoryType.USER_PROFILE)
            tag_set = set(tags)
            for item in existing:
                if set(item.tags) == tag_set:
                    updated = self.update(
                        item.id,
                        MemoryType.USER_PROFILE,
                        content=content,
                        tags=tags,
                        metadata={"source": source},
                    )
                    if updated is not None:
                        return updated

        # 동일 태그 항목 없으면 새로 저장
        item = MemoryItem(
            type=MemoryType.USER_PROFILE,
            content=content,
            tags=tags or [],
            source=source,
            metadata={"source": source},
        )
        self.put(item)
        return item

    # ── Novelty 판단 ─────────────────────────────────────────

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

    # ── 파일 영속화 ────────────────────────────────────────────

    def _persist_path(self, memory_type: MemoryType) -> Path | None:
        """영속화 파일 경로."""
        if not self._persist_dir:
            return None
        return self._persist_dir / f"{memory_type.value}.jsonl"

    def _append_to_file(self, item: MemoryItem) -> None:
        """메모리 항목을 JSONL 파일에 추가."""
        path = self._persist_path(item.type)
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(item.model_dump_json() + "\n")
            logger.debug("메모리 영속화: %s → %s", item.id[:8], path.name)
        except OSError as e:
            logger.warning("메모리 영속화 실패: %s", e)

    def _rewrite_file(self, memory_type: MemoryType) -> None:
        """특정 memory_type의 JSONL 전체를 현재 인덱스 기반으로 재작성."""
        path = self._persist_path(memory_type)
        if not path:
            return
        try:
            items = self.list_by_type(memory_type)
            with open(path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(item.model_dump_json() + "\n")
            logger.debug(
                "메모리 파일 재작성: %s (%d개)", path.name, len(items)
            )
        except OSError as e:
            logger.warning("메모리 파일 재작성 실패: %s", e)

    def _load_persisted(self) -> None:
        """영속화 대상 타입의 메모리를 파일에서 로드."""
        total_loaded = 0
        for mem_type in _PERSISTENT_TYPES:
            path = self._persist_path(mem_type)
            if not path or not path.exists():
                continue

            loaded = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = MemoryItem.model_validate_json(line)
                    ns = _namespace(item.type, item.session_id)
                    if ns not in self._index:
                        self._index[ns] = {}
                    # 중복 방지 (같은 ID가 이미 있으면 스킵)
                    if item.id not in self._index[ns]:
                        self._index[ns][item.id] = item
                        loaded += 1
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning("영속 메모리 파싱 실패 (무시): %s", e)

            if loaded:
                logger.info(
                    "%s Memory %d개 로드됨 (%s)", mem_type.value, loaded, path
                )
            total_loaded += loaded

        if total_loaded:
            logger.info("영속 메모리 총 %d개 로드 완료", total_loaded)

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
