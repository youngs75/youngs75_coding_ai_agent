"""A2A 에이전트 디스커버리 모듈.

AgentCard 기반 에이전트 등록/검색/능력 매칭을 담당한다.

주요 컴포넌트:
  - AgentCardRegistry: 에이전트 카드 등록/검색 레지스트리
  - fetch_agent_card: 원격 에이전트에서 AgentCard를 가져오는 유틸리티
  - match_skills: 요청에 맞는 에이전트를 스킬/태그 기반으로 매칭

사용 예:
    registry = AgentCardRegistry()
    await registry.discover("http://localhost:8080")
    matches = registry.find_by_skill("code_generation")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from a2a.client import A2ACardResolver
from a2a.types import AgentCard, AgentSkill

logger = logging.getLogger(__name__)


@dataclass
class AgentCardEntry:
    """레지스트리에 등록된 에이전트 카드 항목."""

    card: AgentCard
    url: str
    # 마지막 헬스체크 성공 시각 (epoch)
    last_healthy: float = 0.0
    # 연속 실패 횟수
    consecutive_failures: int = 0
    # 등록 시각
    registered_at: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        """최근 60초 내 헬스체크 성공 여부."""
        return (time.time() - self.last_healthy) < 60.0

    def mark_healthy(self) -> None:
        """헬스체크 성공 기록."""
        self.last_healthy = time.time()
        self.consecutive_failures = 0

    def mark_failed(self) -> None:
        """헬스체크 실패 기록."""
        self.consecutive_failures += 1


@dataclass
class DiscoveryResult:
    """디스커버리 결과."""

    entry: AgentCardEntry
    # 매칭 점수 (0.0 ~ 1.0)
    match_score: float
    # 매칭 근거
    matched_skills: list[str] = field(default_factory=list)
    matched_tags: list[str] = field(default_factory=list)


class AgentCardRegistry:
    """에이전트 카드 기반 디스커버리 레지스트리.

    - 수동 등록 및 원격 디스커버리 지원
    - 스킬/태그 기반 에이전트 검색
    - 주기적 헬스체크로 가용성 관리
    """

    def __init__(
        self,
        health_check_interval: float = 30.0,
        health_check_timeout: float = 5.0,
    ) -> None:
        # url -> AgentCardEntry 매핑
        self._entries: dict[str, AgentCardEntry] = {}
        self._health_check_interval = health_check_interval
        self._health_check_timeout = health_check_timeout
        self._health_task: asyncio.Task | None = None

    # ── 등록/해제 ──────────────────────────────────────────────

    def register(self, card: AgentCard, url: str | None = None) -> AgentCardEntry:
        """에이전트 카드를 수동으로 등록한다.

        Args:
            card: A2A AgentCard
            url: 에이전트 URL (card.url과 다를 경우 오버라이드)

        Returns:
            등록된 AgentCardEntry
        """
        agent_url = url or card.url
        entry = AgentCardEntry(card=card, url=agent_url)
        entry.mark_healthy()  # 수동 등록은 건강 상태로 시작
        self._entries[agent_url] = entry
        logger.info(f"에이전트 등록: {card.name} ({agent_url})")
        return entry

    def unregister(self, url: str) -> bool:
        """에이전트 등록 해제."""
        removed = self._entries.pop(url, None)
        if removed:
            logger.info(f"에이전트 해제: {removed.card.name} ({url})")
            return True
        return False

    async def discover(
        self,
        url: str,
        *,
        timeout: float = 10.0,
    ) -> AgentCardEntry | None:
        """원격 에이전트에서 AgentCard를 가져와 등록한다.

        A2A 프로토콜의 /.well-known/agent-card.json 엔드포인트를 사용.

        Args:
            url: 에이전트 서버 기본 URL
            timeout: 요청 타임아웃 (초)

        Returns:
            등록된 AgentCardEntry 또는 실패 시 None
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=url)
                card = await resolver.get_agent_card()
                entry = self.register(card, url=url)
                logger.info(f"에이전트 디스커버리 성공: {card.name} ({url})")
                return entry

        except Exception as e:
            logger.warning(f"에이전트 디스커버리 실패 ({url}): {e}")
            return None

    async def discover_many(
        self,
        urls: list[str],
        *,
        timeout: float = 10.0,
    ) -> list[AgentCardEntry]:
        """여러 에이전트를 병렬로 디스커버리한다.

        Args:
            urls: 에이전트 서버 URL 목록
            timeout: 개별 요청 타임아웃

        Returns:
            성공적으로 등록된 AgentCardEntry 목록
        """
        tasks = [self.discover(url, timeout=timeout) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        entries = []
        for result in results:
            if isinstance(result, AgentCardEntry):
                entries.append(result)
        return entries

    # ── 검색 ──────────────────────────────────────────────────

    def get(self, url: str) -> AgentCardEntry | None:
        """URL로 에이전트 조회."""
        return self._entries.get(url)

    def get_by_name(self, name: str) -> AgentCardEntry | None:
        """이름으로 에이전트 조회 (부분 매칭 지원)."""
        name_lower = name.lower()
        for entry in self._entries.values():
            if entry.card.name.lower() == name_lower:
                return entry
        # 부분 매칭
        for entry in self._entries.values():
            if name_lower in entry.card.name.lower():
                return entry
        return None

    def list_all(self) -> list[AgentCardEntry]:
        """모든 등록된 에이전트 목록."""
        return list(self._entries.values())

    def list_healthy(self) -> list[AgentCardEntry]:
        """건강한 에이전트만 반환."""
        return [e for e in self._entries.values() if e.is_healthy]

    def find_by_skill(
        self,
        skill_query: str,
        *,
        only_healthy: bool = True,
    ) -> list[DiscoveryResult]:
        """스킬 이름/설명으로 에이전트를 검색한다.

        Args:
            skill_query: 검색할 스킬 키워드
            only_healthy: True면 건강한 에이전트만 검색

        Returns:
            매칭 점수가 높은 순으로 정렬된 DiscoveryResult 목록
        """
        candidates = self.list_healthy() if only_healthy else self.list_all()
        query_lower = skill_query.lower()
        results: list[DiscoveryResult] = []

        for entry in candidates:
            matched_skills: list[str] = []
            max_score = 0.0

            for skill in entry.card.skills:
                score = self._compute_skill_match(skill, query_lower)
                if score > 0:
                    matched_skills.append(skill.name)
                    max_score = max(max_score, score)

            if max_score > 0:
                results.append(DiscoveryResult(
                    entry=entry,
                    match_score=max_score,
                    matched_skills=matched_skills,
                ))

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results

    def find_by_tags(
        self,
        tags: list[str],
        *,
        only_healthy: bool = True,
    ) -> list[DiscoveryResult]:
        """태그로 에이전트를 검색한다.

        Args:
            tags: 검색할 태그 목록
            only_healthy: True면 건강한 에이전트만 검색

        Returns:
            매칭된 태그 수가 많은 순으로 정렬된 DiscoveryResult 목록
        """
        candidates = self.list_healthy() if only_healthy else self.list_all()
        tag_set = {t.lower() for t in tags}
        results: list[DiscoveryResult] = []

        for entry in candidates:
            matched_tags: list[str] = []
            for skill in entry.card.skills:
                for tag in skill.tags:
                    if tag.lower() in tag_set:
                        matched_tags.append(tag)

            if matched_tags:
                score = len(set(matched_tags)) / len(tag_set) if tag_set else 0.0
                results.append(DiscoveryResult(
                    entry=entry,
                    match_score=score,
                    matched_tags=list(set(matched_tags)),
                ))

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results

    # ── 헬스체크 ──────────────────────────────────────────────

    async def health_check(self, url: str) -> bool:
        """단일 에이전트 헬스체크.

        /health 또는 /.well-known/agent-card.json 으로 가용성 확인.

        Returns:
            True면 건강, False면 불건강
        """
        entry = self._entries.get(url)
        if not entry:
            return False

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._health_check_timeout)
            ) as client:
                # /health 엔드포인트 우선 시도
                try:
                    resp = await client.get(f"{url}/health")
                    if resp.status_code == 200:
                        entry.mark_healthy()
                        return True
                except httpx.HTTPError:
                    pass

                # /.well-known/agent-card.json 폴백
                resp = await client.get(f"{url}/.well-known/agent-card.json")
                if resp.status_code == 200:
                    entry.mark_healthy()
                    return True

        except Exception as e:
            logger.debug(f"헬스체크 실패 ({url}): {e}")

        entry.mark_failed()
        return False

    async def health_check_all(self) -> dict[str, bool]:
        """모든 등록된 에이전트에 대해 헬스체크 실행.

        Returns:
            url -> 건강 여부 매핑
        """
        tasks = {url: self.health_check(url) for url in self._entries}
        results: dict[str, bool] = {}
        for url, task_coro in tasks.items():
            results[url] = await task_coro
        return results

    def start_periodic_health_check(self) -> None:
        """주기적 헬스체크 백그라운드 태스크를 시작한다."""
        if self._health_task and not self._health_task.done():
            return  # 이미 실행 중

        self._health_task = asyncio.create_task(self._periodic_health_loop())

    def stop_periodic_health_check(self) -> None:
        """주기적 헬스체크를 중지한다."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            self._health_task = None

    async def _periodic_health_loop(self) -> None:
        """헬스체크 루프 (백그라운드)."""
        while True:
            try:
                await self.health_check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"주기적 헬스체크 오류: {e}")
            await asyncio.sleep(self._health_check_interval)

    # ── 내부 유틸리티 ─────────────────────────────────────────

    @staticmethod
    def _compute_skill_match(skill: AgentSkill, query_lower: str) -> float:
        """스킬과 쿼리 간의 매칭 점수를 계산한다.

        Returns:
            0.0 ~ 1.0 사이의 매칭 점수
        """
        score = 0.0

        # 스킬 이름 정확 매칭 (가장 높은 점수)
        if skill.name.lower() == query_lower:
            score = max(score, 1.0)
        # 스킬 이름 부분 매칭
        elif query_lower in skill.name.lower():
            score = max(score, 0.8)
        # 스킬 ID 매칭
        elif query_lower in skill.id.lower():
            score = max(score, 0.7)

        # 스킬 설명 매칭
        if query_lower in skill.description.lower():
            score = max(score, 0.5)

        # 태그 매칭
        for tag in skill.tags:
            if query_lower == tag.lower():
                score = max(score, 0.6)
            elif query_lower in tag.lower():
                score = max(score, 0.4)

        return score
