"""A2A 메시지 라우팅 고도화 모듈.

에이전트 디스커버리와 능력 기반 라우팅을 결합하여
최적의 에이전트에게 태스크를 위임한다.

주요 컴포넌트:
  - RoutingStrategy: 라우팅 전략 (능력 기반, 라운드로빈, 가중치 기반)
  - TaskDelegator: 태스크 위임/결과 수집 플로우
  - AgentRouter: 에이전트 디스커버리 + 라우팅 + 복원력 통합

사용 예:
    router = AgentRouter()
    router.register_agent(card, url="http://localhost:8080")
    result = await router.route_and_delegate("코드를 생성해주세요")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from a2a.types import AgentCard, SendMessageResponse

from .discovery import AgentCardEntry, AgentCardRegistry, DiscoveryResult
from .resilience import (
    AgentMonitor,
    AsyncStreamingResponse,
    CircuitBreaker,
    CircuitOpenError,
    ResilientA2AClient,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


# ── 라우팅 전략 ──────────────────────────────────────────────


class RoutingMode(str, Enum):
    """라우팅 모드."""

    SKILL_BASED = "skill_based"  # 스킬 매칭 기반
    ROUND_ROBIN = "round_robin"  # 라운드 로빈
    LEAST_LOADED = "least_loaded"  # 최소 부하
    WEIGHTED = "weighted"  # 가중치 기반 (성능 + 비용)


@dataclass
class RoutingDecision:
    """라우팅 결정 결과."""

    target_url: str
    agent_name: str
    # 선택 근거
    reason: str
    # 매칭 점수 (0.0 ~ 1.0)
    confidence: float = 0.0
    # 대안 에이전트 목록
    alternatives: list[str] = field(default_factory=list)


@dataclass
class DelegationResult:
    """태스크 위임 결과."""

    success: bool
    agent_name: str
    agent_url: str
    response: SendMessageResponse | None = None
    streaming_response: AsyncStreamingResponse | None = None
    error: str | None = None
    latency_ms: float = 0.0
    # 폴백 사용 여부
    used_fallback: bool = False
    fallback_agent: str | None = None


class TaskDelegator:
    """태스크 위임 및 결과 수집.

    단일 에이전트 위임뿐 아니라, 다중 에이전트 병렬 위임도 지원한다.
    """

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._retry_policy = retry_policy or RetryPolicy()
        self._timeout = timeout
        self._monitor = AgentMonitor()
        # url -> CircuitBreaker 매핑
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """URL별 서킷 브레이커 조회/생성."""
        if url not in self._circuit_breakers:
            self._circuit_breakers[url] = CircuitBreaker()
        return self._circuit_breakers[url]

    @property
    def monitor(self) -> AgentMonitor:
        """에이전트 모니터 접근."""
        return self._monitor

    async def delegate(
        self,
        url: str,
        content: str,
        *,
        agent_name: str = "unknown",
        fallback_urls: list[str] | None = None,
    ) -> DelegationResult:
        """단일 에이전트에 태스크를 위임한다.

        Args:
            url: 대상 에이전트 URL
            content: 전송할 메시지
            agent_name: 에이전트 이름 (로깅용)
            fallback_urls: 폴백 에이전트 URL 목록

        Returns:
            DelegationResult
        """
        start_time = time.time()
        client = ResilientA2AClient(
            url=url,
            retry_policy=self._retry_policy,
            circuit_breaker=self._get_circuit_breaker(url),
            monitor=self._monitor,
            fallback_urls=fallback_urls or [],
            timeout=self._timeout,
        )

        try:
            response = await client.send_message(content)
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"태스크 위임 성공: {agent_name} ({url}) [{latency_ms:.0f}ms]")
            return DelegationResult(
                success=True,
                agent_name=agent_name,
                agent_url=url,
                response=response,
                latency_ms=latency_ms,
            )

        except CircuitOpenError:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"서킷 열림 - 에이전트 차단: {agent_name}"
            logger.warning(error_msg)
            return DelegationResult(
                success=False,
                agent_name=agent_name,
                agent_url=url,
                error=error_msg,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"태스크 위임 실패: {agent_name} - {e}"
            logger.error(error_msg)
            return DelegationResult(
                success=False,
                agent_name=agent_name,
                agent_url=url,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def delegate_streaming(
        self,
        url: str,
        content: str,
        *,
        agent_name: str = "unknown",
    ) -> DelegationResult:
        """스트리밍 방식으로 태스크를 위임한다.

        Args:
            url: 대상 에이전트 URL
            content: 전송할 메시지
            agent_name: 에이전트 이름 (로깅용)

        Returns:
            DelegationResult (streaming_response 필드에 스트리밍 응답)
        """
        start_time = time.time()
        client = ResilientA2AClient(
            url=url,
            retry_policy=self._retry_policy,
            circuit_breaker=self._get_circuit_breaker(url),
            monitor=self._monitor,
            timeout=self._timeout,
        )

        try:
            stream_resp = await client.send_message_streaming(content)
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"스트리밍 위임 시작: {agent_name} ({url}) "
                f"[초기 응답 {latency_ms:.0f}ms]"
            )
            return DelegationResult(
                success=True,
                agent_name=agent_name,
                agent_url=url,
                streaming_response=stream_resp,
                latency_ms=latency_ms,
            )

        except CircuitOpenError:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"서킷 열림 - 스트리밍 위임 차단: {agent_name}"
            logger.warning(error_msg)
            return DelegationResult(
                success=False,
                agent_name=agent_name,
                agent_url=url,
                error=error_msg,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = f"스트리밍 위임 실패: {agent_name} - {e}"
            logger.error(error_msg)
            return DelegationResult(
                success=False,
                agent_name=agent_name,
                agent_url=url,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def delegate_parallel(
        self,
        targets: list[dict[str, str]],
        content: str,
    ) -> list[DelegationResult]:
        """여러 에이전트에 병렬로 태스크를 위임한다.

        Args:
            targets: [{"url": "...", "name": "..."}] 형태의 대상 목록
            content: 전송할 메시지

        Returns:
            각 에이전트별 DelegationResult 목록
        """
        tasks = [
            self.delegate(
                url=t["url"],
                content=content,
                agent_name=t.get("name", "unknown"),
            )
            for t in targets
        ]
        return list(await asyncio.gather(*tasks))

    async def delegate_with_consensus(
        self,
        targets: list[dict[str, str]],
        content: str,
        *,
        min_success: int = 1,
    ) -> list[DelegationResult]:
        """여러 에이전트에 위임하고 최소 성공 수를 보장한다.

        Args:
            targets: 대상 에이전트 목록
            content: 전송할 메시지
            min_success: 최소 성공 응답 수

        Returns:
            DelegationResult 목록

        Raises:
            RuntimeError: 최소 성공 수를 충족하지 못할 때
        """
        results = await self.delegate_parallel(targets, content)
        success_count = sum(1 for r in results if r.success)

        if success_count < min_success:
            raise RuntimeError(
                f"합의 실패: {success_count}/{len(targets)} 성공 "
                f"(최소 {min_success} 필요)"
            )

        return results


# ── 통합 라우터 ──────────────────────────────────────────────


class AgentRouter:
    """에이전트 디스커버리 + 능력 기반 라우팅 + 복원력 통합 라우터.

    사용 흐름:
      1. register_agent()로 에이전트 등록 (또는 discover()로 자동 등록)
      2. route()로 최적 에이전트 결정
      3. delegate()로 태스크 위임
      4. route_and_delegate()로 1~3 통합 실행

    라우팅 전략:
      - SKILL_BASED: 스킬 매칭 점수 + 성공률 + 응답 시간 종합 평가
      - ROUND_ROBIN: 순차적 분배
      - LEAST_LOADED: 활성 요청이 가장 적은 에이전트 선택
      - WEIGHTED: 성공률/응답시간 기반 가중치 계산
    """

    def __init__(
        self,
        *,
        routing_mode: RoutingMode = RoutingMode.SKILL_BASED,
        retry_policy: RetryPolicy | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.registry = AgentCardRegistry()
        self.delegator = TaskDelegator(
            retry_policy=retry_policy,
            timeout=timeout,
        )
        self._routing_mode = routing_mode
        self._round_robin_index = 0

    # ── 등록/디스커버리 ──────────────────────────────────

    def register_agent(
        self,
        card: AgentCard,
        url: str | None = None,
    ) -> AgentCardEntry:
        """에이전트를 수동으로 등록한다."""
        return self.registry.register(card, url=url)

    async def discover(self, url: str) -> AgentCardEntry | None:
        """원격 에이전트를 디스커버리하여 등록한다."""
        return await self.registry.discover(url)

    async def discover_many(self, urls: list[str]) -> list[AgentCardEntry]:
        """여러 에이전트를 병렬 디스커버리한다."""
        return await self.registry.discover_many(urls)

    # ── 라우팅 ──────────────────────────────────────────

    def route(
        self,
        query: str,
        *,
        required_tags: list[str] | None = None,
    ) -> RoutingDecision | None:
        """쿼리에 최적인 에이전트를 결정한다.

        Args:
            query: 사용자 요청 텍스트
            required_tags: 필수 태그 목록

        Returns:
            RoutingDecision 또는 매칭 에이전트 없으면 None
        """
        if self._routing_mode == RoutingMode.ROUND_ROBIN:
            return self._route_round_robin()
        elif self._routing_mode == RoutingMode.WEIGHTED:
            return self._route_weighted()
        else:
            return self._route_skill_based(query, required_tags)

    def _route_skill_based(
        self,
        query: str,
        required_tags: list[str] | None = None,
    ) -> RoutingDecision | None:
        """스킬 매칭 기반 라우팅."""
        # 스킬 검색
        skill_results = self.registry.find_by_skill(query, only_healthy=True)

        # 태그 검색 (태그가 지정된 경우)
        if required_tags:
            tag_results = self.registry.find_by_tags(required_tags, only_healthy=True)
            # 스킬 + 태그 결과 통합
            combined = self._merge_results(skill_results, tag_results)
        else:
            combined = skill_results

        if not combined:
            # 매칭 결과가 없으면 건강한 에이전트 중 첫 번째 선택 (폴백)
            healthy = self.registry.list_healthy()
            if not healthy:
                return None
            first = healthy[0]
            return RoutingDecision(
                target_url=first.url,
                agent_name=first.card.name,
                reason="매칭되는 스킬 없음 — 기본 에이전트 선택",
                confidence=0.1,
            )

        # 최고 점수 에이전트 선택
        best = combined[0]
        alternatives = [r.entry.card.name for r in combined[1:3]]

        # 성공률 보정 (모니터링 데이터가 있으면)
        stats = self.delegator.monitor.get_stats(best.entry.url)
        adjusted_score = best.match_score * stats.success_rate

        return RoutingDecision(
            target_url=best.entry.url,
            agent_name=best.entry.card.name,
            reason=(
                f"스킬 매칭: {best.matched_skills}, "
                f"점수: {best.match_score:.2f}, "
                f"성공률 보정: {adjusted_score:.2f}"
            ),
            confidence=adjusted_score,
            alternatives=alternatives,
        )

    def _route_round_robin(self) -> RoutingDecision | None:
        """라운드 로빈 라우팅."""
        healthy = self.registry.list_healthy()
        if not healthy:
            return None

        entry = healthy[self._round_robin_index % len(healthy)]
        self._round_robin_index += 1

        return RoutingDecision(
            target_url=entry.url,
            agent_name=entry.card.name,
            reason=f"라운드 로빈: 인덱스 {self._round_robin_index - 1}",
            confidence=0.5,
        )

    def _route_weighted(self) -> RoutingDecision | None:
        """가중치 기반 라우팅 (성공률 + 응답 시간)."""
        healthy = self.registry.list_healthy()
        if not healthy:
            return None

        best_entry: AgentCardEntry | None = None
        best_score = -1.0

        for entry in healthy:
            stats = self.delegator.monitor.get_stats(entry.url)
            # 성공률 (70%) + 응답 시간 역수 (30%)
            latency_score = 1.0 / (1.0 + stats.avg_latency_ms / 1000.0)
            score = 0.7 * stats.success_rate + 0.3 * latency_score

            if score > best_score:
                best_score = score
                best_entry = entry

        if not best_entry:
            return None

        return RoutingDecision(
            target_url=best_entry.url,
            agent_name=best_entry.card.name,
            reason=f"가중치 기반: 점수={best_score:.2f}",
            confidence=best_score,
        )

    # ── 위임 ────────────────────────────────────────────

    async def delegate(
        self,
        decision: RoutingDecision,
        content: str,
    ) -> DelegationResult:
        """라우팅 결정에 따라 태스크를 위임한다.

        Args:
            decision: 라우팅 결정
            content: 전송할 메시지

        Returns:
            DelegationResult
        """
        # 대안 에이전트 URL을 폴백으로 설정
        fallback_urls = []
        for alt_name in decision.alternatives:
            alt_entry = self.registry.get_by_name(alt_name)
            if alt_entry:
                fallback_urls.append(alt_entry.url)

        return await self.delegator.delegate(
            url=decision.target_url,
            content=content,
            agent_name=decision.agent_name,
            fallback_urls=fallback_urls,
        )

    async def route_and_delegate(
        self,
        query: str,
        *,
        required_tags: list[str] | None = None,
    ) -> DelegationResult:
        """라우팅 + 위임을 한 번에 수행한다.

        Args:
            query: 사용자 요청 텍스트
            required_tags: 필수 태그 목록

        Returns:
            DelegationResult

        Raises:
            RuntimeError: 라우팅 대상 에이전트를 찾지 못했을 때
        """
        decision = self.route(query, required_tags=required_tags)
        if not decision:
            raise RuntimeError("라우팅 대상 에이전트를 찾지 못했습니다.")

        logger.info(
            f"라우팅 결정: {decision.agent_name} ({decision.target_url}) "
            f"[{decision.reason}]"
        )

        return await self.delegate(decision, query)

    async def route_and_delegate_streaming(
        self,
        query: str,
        *,
        required_tags: list[str] | None = None,
    ) -> DelegationResult:
        """스트리밍 방식으로 라우팅 + 위임을 수행한다.

        Args:
            query: 사용자 요청 텍스트
            required_tags: 필수 태그 목록

        Returns:
            DelegationResult (streaming_response 필드 포함)

        Raises:
            RuntimeError: 라우팅 대상 에이전트를 찾지 못했을 때
        """
        decision = self.route(query, required_tags=required_tags)
        if not decision:
            raise RuntimeError("라우팅 대상 에이전트를 찾지 못했습니다.")

        return await self.delegator.delegate_streaming(
            url=decision.target_url,
            content=query,
            agent_name=decision.agent_name,
        )

    async def broadcast(
        self,
        query: str,
        *,
        required_tags: list[str] | None = None,
        max_agents: int = 0,
    ) -> list[DelegationResult]:
        """여러 에이전트에 병렬로 태스크를 위임한다.

        스킬/태그 매칭된 모든 에이전트에 동시 요청을 보낸다.

        Args:
            query: 사용자 요청 텍스트
            required_tags: 태그 목록
            max_agents: 최대 에이전트 수 (0이면 전체)

        Returns:
            DelegationResult 목록
        """
        results = self.registry.find_by_skill(query, only_healthy=True)
        if required_tags:
            tag_results = self.registry.find_by_tags(required_tags, only_healthy=True)
            results = self._merge_results(results, tag_results)

        if not results:
            return []

        targets = [{"url": r.entry.url, "name": r.entry.card.name} for r in results]
        if max_agents > 0:
            targets = targets[:max_agents]

        return await self.delegator.delegate_parallel(targets, query)

    # ── 내부 유틸리티 ──────────────────────────────────

    @staticmethod
    def _merge_results(
        skill_results: list[DiscoveryResult],
        tag_results: list[DiscoveryResult],
    ) -> list[DiscoveryResult]:
        """스킬 검색 결과와 태그 검색 결과를 병합한다.

        같은 에이전트가 양쪽에 모두 있으면 점수를 합산한다.
        """
        merged: dict[str, DiscoveryResult] = {}

        for r in skill_results:
            key = r.entry.url
            if key in merged:
                existing = merged[key]
                existing.match_score = max(existing.match_score, r.match_score)
                existing.matched_skills.extend(r.matched_skills)
            else:
                merged[key] = DiscoveryResult(
                    entry=r.entry,
                    match_score=r.match_score,
                    matched_skills=list(r.matched_skills),
                    matched_tags=[],
                )

        for r in tag_results:
            key = r.entry.url
            if key in merged:
                existing = merged[key]
                # 태그 매칭은 보너스 점수 (+0.2)
                existing.match_score += r.match_score * 0.2
                existing.matched_tags.extend(r.matched_tags)
            else:
                merged[key] = DiscoveryResult(
                    entry=r.entry,
                    match_score=r.match_score * 0.5,  # 태그만 매칭은 절반 점수
                    matched_skills=[],
                    matched_tags=list(r.matched_tags),
                )

        results = list(merged.values())
        results.sort(key=lambda x: x.match_score, reverse=True)
        return results
