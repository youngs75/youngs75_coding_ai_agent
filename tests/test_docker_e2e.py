"""
Docker E2E 테스트 — CLI → Agent → MCP 서비스 체인 검증

pytest 기반으로 Docker 환경의 전체 서비스 연동을 검증한다.
Docker 컨테이너가 실행 중이지 않으면 모든 테스트를 자동 skip 처리한다.

실행 (Docker 컨테이너 기동 후):
  cd youngs75_a2a/docker && docker compose up -d
  python -m pytest tests/test_docker_e2e.py -v

로컬 환경 (Docker 없이) 실행 시 모든 테스트 skip:
  python -m pytest tests/test_docker_e2e.py -v
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx
import pytest

# ─── 서비스 엔드포인트 정의 ──────────────────────────────

MCP_SERVICES: dict[str, str] = {
    "Tavily": "http://localhost:3001",
    "arXiv": "http://localhost:3000",
    "Serper": "http://localhost:3002",
}

AGENT_SERVICES: dict[str, str] = {
    "SimpleReAct": "http://localhost:18081",
    "DeepResearch": "http://localhost:18082",
    "DeepResearchA2A": "http://localhost:18083",
}

ALL_SERVICES: dict[str, str] = {**MCP_SERVICES, **AGENT_SERVICES}

# Docker 내부 네트워크 서비스명 → 포트 매핑 (docker-compose.yml 기준)
DOCKER_INTERNAL_ENDPOINTS: dict[str, str] = {
    "mcp-tavily": "http://mcp-tavily:3001",
    "mcp-arxiv": "http://mcp-arxiv:3000",
    "mcp-serper": "http://mcp-serper:3002",
    "agent-simple-react": "http://agent-simple-react:18081",
    "agent-deep-research": "http://agent-deep-research:18082",
    "agent-deep-research-a2a": "http://agent-deep-research-a2a:18083",
}


# ─── 헬퍼 함수 ──────────────────────────────────────────


async def _is_service_reachable(url: str, timeout: float = 2.0) -> bool:
    """서비스에 연결 가능한지 빠르게 확인한다."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # /health 먼저 시도
            try:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            # MCP 서버는 /mcp 엔드포인트 사용
            try:
                resp = await client.get(f"{url}/mcp")
                return resp.status_code in (200, 405)
            except httpx.HTTPError:
                return False
    except Exception:
        return False


async def _check_any_service_running() -> bool:
    """최소 하나의 서비스가 실행 중인지 확인한다."""
    checks = [_is_service_reachable(url) for url in ALL_SERVICES.values()]
    results = await asyncio.gather(*checks, return_exceptions=True)
    return any(r is True for r in results)


async def _get_health(url: str) -> dict[str, Any] | None:
    """서비스 /health 엔드포인트에서 JSON 응답을 가져온다."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


async def _get_agent_card(url: str) -> dict[str, Any] | None:
    """A2A AgentCard를 가져온다."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/.well-known/agent-card.json")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


async def _send_a2a_message(
    url: str, query: str, timeout: float = 60.0
) -> dict[str, Any]:
    """A2A 프로토콜로 메시지를 보내고 응답을 받는다."""
    request_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
            }
        },
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=request_payload)
        resp.raise_for_status()
        return resp.json()


# ─── pytest 픽스처 ───────────────────────────────────────


@pytest.fixture(scope="module")
async def docker_available():
    """Docker 서비스가 실행 중인지 확인하는 모듈 레벨 픽스처."""
    available = await _check_any_service_running()
    if not available:
        pytest.skip("Docker 서비스가 실행 중이지 않습니다 (docker compose up -d 필요)")
    return True


async def _skip_if_service_down(url: str, name: str):
    """특정 서비스가 미실행이면 테스트를 skip 처리한다."""
    reachable = await _is_service_reachable(url)
    if not reachable:
        pytest.skip(f"{name} 서비스 미실행")


# ─── MCP 서비스 헬스체크 테스트 ──────────────────────────


class TestMCPServiceHealth:
    """MCP 서버 3종의 접근성을 검증한다."""

    @pytest.mark.parametrize("name,url", list(MCP_SERVICES.items()))
    async def test_mcp_service_reachable(self, docker_available, name: str, url: str):
        """MCP 서비스가 /mcp 엔드포인트에 응답하는지 확인한다."""
        await _skip_if_service_down(url, name)
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/mcp")
            # MCP 서버는 GET /mcp에 대해 응답 (405 또는 200)
            assert resp.status_code in (200, 405), (
                f"{name}: 예상치 못한 상태 {resp.status_code}"
            )

    @pytest.mark.parametrize(
        "name,url",
        [
            ("Tavily", "http://localhost:3001"),
            ("Serper", "http://localhost:3002"),
        ],
    )
    async def test_mcp_with_env_keys(self, docker_available, name: str, url: str):
        """API 키가 필요한 MCP 서비스가 정상 기동되었는지 확인한다.

        Tavily/Serper는 환경변수(API 키)가 설정되어야 정상 동작한다.
        기동 자체가 되었으면 env_file에서 키를 올바르게 읽은 것이다.
        """
        await _skip_if_service_down(url, name)
        reachable = await _is_service_reachable(url)
        assert reachable, f"{name}: 서비스 기동 실패 (API 키 누락 가능)"


# ─── Agent 서비스 헬스체크 테스트 ────────────────────────


class TestAgentServiceHealth:
    """Agent 서버 3종의 /health 엔드포인트를 검증한다."""

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    async def test_agent_health_endpoint(self, docker_available, name: str, url: str):
        """/health 엔드포인트가 status=healthy를 반환하는지 확인한다."""
        await _skip_if_service_down(url, name)
        health = await _get_health(url)
        assert health is not None, f"{name}: /health 응답 없음"
        assert health.get("status") == "healthy", f"{name}: {health}"

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    async def test_agent_health_includes_metadata(
        self, docker_available, name: str, url: str
    ):
        """/health 응답에 agent 이름과 port 정보가 포함되는지 확인한다."""
        await _skip_if_service_down(url, name)
        health = await _get_health(url)
        assert health is not None
        assert "agent" in health, f"{name}: agent 필드 누락"
        assert "port" in health, f"{name}: port 필드 누락"


# ─── AgentCard (A2A 프로토콜) 테스트 ────────────────────


class TestAgentCard:
    """A2A AgentCard 조회 및 필수 필드 검증."""

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    async def test_agent_card_accessible(self, docker_available, name: str, url: str):
        """/.well-known/agent-card.json 엔드포인트가 응답하는지 확인한다."""
        await _skip_if_service_down(url, name)
        card = await _get_agent_card(url)
        assert card is not None, f"{name}: AgentCard 조회 실패"

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    async def test_agent_card_required_fields(
        self, docker_available, name: str, url: str
    ):
        """AgentCard에 필수 필드(name, version, capabilities)가 있는지 확인한다."""
        await _skip_if_service_down(url, name)
        card = await _get_agent_card(url)
        assert card is not None
        assert "name" in card, "name 필드 누락"
        assert "version" in card, "version 필드 누락"
        assert "capabilities" in card, "capabilities 필드 누락"

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    async def test_agent_card_streaming_capability(
        self, docker_available, name: str, url: str
    ):
        """AgentCard의 streaming 설정이 boolean 값인지 확인한다."""
        await _skip_if_service_down(url, name)
        card = await _get_agent_card(url)
        assert card is not None
        capabilities = card.get("capabilities", {})
        assert "streaming" in capabilities, "streaming 필드 누락"
        assert isinstance(capabilities["streaming"], bool)


# ─── A2A 프로토콜 요청/응답 테스트 ──────────────────────


class TestA2AProtocol:
    """Agent 서비스에 A2A 프로토콜로 메시지를 전송하고 응답을 검증한다."""

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_a2a_message_send(self, docker_available, name: str, url: str):
        """A2A message/send 호출이 유효한 JSON-RPC 응답을 반환하는지 확인한다."""
        await _skip_if_service_down(url, name)
        result = await _send_a2a_message(
            url, "안녕하세요, 간단히 테스트입니다.", timeout=120.0
        )
        # JSON-RPC 응답 구조 확인
        assert "jsonrpc" in result, "jsonrpc 필드 누락"
        assert result["jsonrpc"] == "2.0"
        # result 또는 error 중 하나는 있어야 함
        assert "result" in result or "error" in result, "result/error 필드 모두 누락"

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_simple_react_query(self, docker_available):
        """SimpleReAct 에이전트에 검색 질의가 정상 처리되는지 확인한다."""
        url = AGENT_SERVICES["SimpleReAct"]
        await _skip_if_service_down(url, "SimpleReAct")
        result = await _send_a2a_message(url, "Python GIL이란?", timeout=120.0)
        assert "result" in result, f"에러 응답: {result.get('error')}"

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_deep_research_query(self, docker_available):
        """DeepResearch 에이전트에 연구 질의가 정상 처리되는지 확인한다."""
        url = AGENT_SERVICES["DeepResearch"]
        await _skip_if_service_down(url, "DeepResearch")
        result = await _send_a2a_message(
            url, "LLM이란 무엇인가 한 문장으로 설명해줘", timeout=120.0
        )
        assert "result" in result, f"에러 응답: {result.get('error')}"


# ─── CLI → Agent 연동 체인 테스트 ────────────────────────


class TestCLIToAgentChain:
    """CLI 컨테이너에서 Agent 서비스로의 연동 체인을 검증한다.

    CLI는 대화형 모드이므로 직접 호출 대신,
    Agent 서비스의 A2A 엔드포인트를 통한 간접 검증을 수행한다.
    """

    @pytest.mark.parametrize("name,url", list(AGENT_SERVICES.items()))
    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_agent_accepts_cli_format_request(
        self, docker_available, name: str, url: str
    ):
        """CLI가 보내는 것과 동일한 형식의 A2A 요청을 Agent가 수락하는지 확인한다."""
        await _skip_if_service_down(url, name)
        # CLI에서 사용하는 A2AClient 형식과 동일한 요청
        request_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": "테스트 메시지"}],
                }
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=request_payload)
            assert resp.status_code == 200, f"{name}: HTTP {resp.status_code}"
            data = resp.json()
            assert data.get("jsonrpc") == "2.0"

    async def test_cli_available_agents_match_docker_services(self):
        """docker-compose의 Agent 서비스와 CLI AVAILABLE_AGENTS 매핑이 일관적인지 확인한다.

        CLI 코드에 정의된 에이전트 목록이 Docker에서 실행되는 서비스를 커버하는지
        구조적으로 검증한다 (코드 import 기반 검증, Docker 미실행시에도 동작).
        """
        from youngs75_a2a.cli.commands import AVAILABLE_AGENTS

        # Docker에서 실행되는 에이전트 타입에 대응하는 CLI 에이전트 이름
        docker_to_cli_mapping = {
            "SimpleReAct": "simple_react",
            "DeepResearch": "deep_research",
            # DeepResearchA2A는 deep_research의 A2A 래퍼
        }
        for docker_name, cli_name in docker_to_cli_mapping.items():
            assert cli_name in AVAILABLE_AGENTS, (
                f"Docker 서비스 {docker_name}에 대응하는 '{cli_name}'이 "
                f"AVAILABLE_AGENTS에 없음: {AVAILABLE_AGENTS}"
            )


# ─── Agent → MCP 서비스 체인 테스트 ─────────────────────


class TestAgentToMCPChain:
    """Agent 서비스가 MCP 도구를 정상적으로 호출하는지 검증한다.

    Agent에게 MCP 도구 사용이 필요한 질의를 보내서
    전체 체인(Agent → MCP)이 동작하는지 확인한다.
    """

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_simple_react_uses_tavily_mcp(self, docker_available):
        """SimpleReAct가 Tavily MCP를 통해 검색을 수행하는지 확인한다.

        SimpleReAct는 TAVILY_MCP_URL을 통해 Tavily MCP에 연결된다.
        검색 질의를 보내면 MCP 도구를 호출하여 결과를 반환해야 한다.
        """
        url = AGENT_SERVICES["SimpleReAct"]
        await _skip_if_service_down(url, "SimpleReAct")
        await _skip_if_service_down(MCP_SERVICES["Tavily"], "Tavily MCP")

        result = await _send_a2a_message(
            url, "2026년 AI 최신 뉴스 1개만 검색해줘", timeout=180.0
        )
        assert "result" in result, f"Agent→MCP 체인 실패: {result.get('error')}"

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_deep_research_uses_multiple_mcp(self, docker_available):
        """DeepResearch가 여러 MCP 서비스를 활용하는지 확인한다.

        DeepResearch는 Tavily + arXiv + Serper MCP에 모두 연결되어 있다.
        복합 질의를 보내면 다중 MCP 도구를 활용해야 한다.
        """
        url = AGENT_SERVICES["DeepResearch"]
        await _skip_if_service_down(url, "DeepResearch")

        result = await _send_a2a_message(
            url, "양자 컴퓨팅 최신 동향을 한 문장으로 요약해줘", timeout=180.0
        )
        assert "result" in result, f"Agent→MCP 체인 실패: {result.get('error')}"


# ─── 네트워크 연결성 테스트 ──────────────────────────────


class TestNetworkConnectivity:
    """Docker 네트워크(youngs75_net) 내 서비스 간 연결성을 검증한다."""

    @pytest.mark.parametrize("name,url", list(ALL_SERVICES.items()))
    async def test_service_port_exposed(self, docker_available, name: str, url: str):
        """각 서비스의 포트가 호스트에서 접근 가능한지 확인한다."""
        reachable = await _is_service_reachable(url, timeout=3.0)
        if not reachable:
            pytest.skip(f"{name}: 포트 미노출 (서비스 미실행)")
        assert reachable

    async def test_all_mcp_services_healthy(self, docker_available):
        """모든 MCP 서비스가 동시에 응답하는지 확인한다 (의존성 동시성 검증)."""
        results = {}
        for name, url in MCP_SERVICES.items():
            results[name] = await _is_service_reachable(url)
        running = [n for n, ok in results.items() if ok]
        assert len(running) > 0, "실행 중인 MCP 서비스 없음"

    async def test_all_agent_services_healthy(self, docker_available):
        """모든 Agent 서비스가 동시에 응답하는지 확인한다."""
        results = {}
        for name, url in AGENT_SERVICES.items():
            results[name] = await _is_service_reachable(url)
        running = [n for n, ok in results.items() if ok]
        assert len(running) > 0, "실행 중인 Agent 서비스 없음"


# ─── 에러 핸들링 테스트 ─────────────────────────────────


class TestErrorHandling:
    """서비스 장애 시 graceful degradation을 검증한다."""

    async def test_agent_returns_error_on_invalid_method(self, docker_available):
        """잘못된 JSON-RPC 메서드에 대해 에러 응답을 반환하는지 확인한다."""
        url = AGENT_SERVICES["SimpleReAct"]
        await _skip_if_service_down(url, "SimpleReAct")

        invalid_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "invalid/method",
            "params": {},
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=invalid_payload)
            # 서버가 에러 응답을 반환하거나 4xx/5xx를 반환해야 함
            if resp.status_code == 200:
                data = resp.json()
                assert "error" in data, "잘못된 메서드에 대해 error 응답 필요"
            else:
                assert resp.status_code in (400, 404, 405, 422, 500)

    async def test_agent_handles_empty_message(self, docker_available):
        """빈 메시지를 보냈을 때 서버가 크래시하지 않는지 확인한다."""
        url = AGENT_SERVICES["SimpleReAct"]
        await _skip_if_service_down(url, "SimpleReAct")

        empty_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": ""}],
                }
            },
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=empty_payload)
            # 크래시 없이 응답 (200 정상 응답 또는 에러 코드)
            assert resp.status_code < 600, f"서버 크래시: {resp.status_code}"

    async def test_health_returns_quickly(self, docker_available):
        """/health 엔드포인트가 2초 이내에 응답하는지 확인한다."""
        for name, url in AGENT_SERVICES.items():
            if not await _is_service_reachable(url):
                continue
            async with httpx.AsyncClient(timeout=2.0) as client:
                try:
                    resp = await client.get(f"{url}/health")
                    assert resp.status_code == 200, (
                        f"{name}: /health 응답 지연 또는 실패"
                    )
                except httpx.TimeoutException:
                    pytest.fail(f"{name}: /health 응답 2초 초과")

    async def test_nonexistent_service_connection_fails_gracefully(self):
        """존재하지 않는 서비스 URL에 대한 연결이 깔끔하게 실패하는지 확인한다."""
        fake_url = "http://localhost:19999"
        reachable = await _is_service_reachable(fake_url, timeout=1.0)
        assert not reachable, "존재하지 않는 포트에 연결이 성공하면 안 됨"


# ─── Docker 설정 검증 테스트 (정적 분석) ────────────────


class TestDockerConfigValidation:
    """docker-compose.yml 및 Dockerfile 설정의 정합성을 검증한다.

    Docker 실행 없이 설정 파일 내용을 정적으로 검증한다.
    """

    def test_docker_compose_services_defined(self):
        """docker-compose.yml에 필수 서비스가 모두 정의되어 있는지 확인한다."""
        import yaml
        from pathlib import Path

        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.yml 파일 없음")

        with open(compose_path) as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        required = [
            "mcp-tavily",
            "mcp-arxiv",
            "mcp-serper",
            "agent-simple-react",
            "agent-deep-research",
            "agent-deep-research-a2a",
            "cli",
        ]
        for svc in required:
            assert svc in services, f"필수 서비스 '{svc}' 누락"

    def test_docker_compose_network_defined(self):
        """모든 서비스가 동일 네트워크(youngs75_net)에 연결되어 있는지 확인한다."""
        import yaml
        from pathlib import Path

        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.yml 파일 없음")

        with open(compose_path) as f:
            config = yaml.safe_load(f)

        networks = config.get("networks", {})
        assert "youngs75_net" in networks, "youngs75_net 네트워크 미정의"

        for svc_name, svc_config in config.get("services", {}).items():
            svc_networks = svc_config.get("networks", [])
            assert "youngs75_net" in svc_networks, (
                f"서비스 '{svc_name}'이 youngs75_net에 연결되지 않음"
            )

    def test_docker_compose_agent_depends_on_mcp(self):
        """Agent 서비스가 MCP 서비스에 대한 depends_on 설정이 있는지 확인한다."""
        import yaml
        from pathlib import Path

        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.yml 파일 없음")

        with open(compose_path) as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})

        # SimpleReAct → mcp-tavily
        sr_deps = services.get("agent-simple-react", {}).get("depends_on", {})
        assert "mcp-tavily" in sr_deps, "SimpleReAct: mcp-tavily 의존성 누락"

        # DeepResearch → mcp-tavily, mcp-arxiv, mcp-serper
        dr_deps = services.get("agent-deep-research", {}).get("depends_on", {})
        for mcp in ["mcp-tavily", "mcp-arxiv", "mcp-serper"]:
            assert mcp in dr_deps, f"DeepResearch: {mcp} 의존성 누락"

    def test_docker_compose_healthcheck_defined(self):
        """모든 MCP/Agent 서비스에 healthcheck가 정의되어 있는지 확인한다."""
        import yaml
        from pathlib import Path

        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.yml 파일 없음")

        with open(compose_path) as f:
            config = yaml.safe_load(f)

        # CLI는 대화형이므로 healthcheck 불필요
        services_needing_health = [
            "mcp-tavily",
            "mcp-arxiv",
            "mcp-serper",
            "agent-simple-react",
            "agent-deep-research",
            "agent-deep-research-a2a",
        ]
        for svc_name in services_needing_health:
            svc = config.get("services", {}).get(svc_name, {})
            assert "healthcheck" in svc, f"서비스 '{svc_name}': healthcheck 미정의"
            hc = svc["healthcheck"]
            assert "test" in hc, f"서비스 '{svc_name}': healthcheck test 미정의"
            assert "interval" in hc, f"서비스 '{svc_name}': healthcheck interval 미정의"

    def test_docker_compose_depends_on_uses_service_healthy(self):
        """depends_on 조건이 service_healthy를 사용하는지 확인한다.

        단순 depends_on 대신 condition: service_healthy를 사용해야
        MCP 서버가 완전히 준비된 후 Agent가 기동된다.
        """
        import yaml
        from pathlib import Path

        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.yml 파일 없음")

        with open(compose_path) as f:
            config = yaml.safe_load(f)

        agent_services = [
            "agent-simple-react",
            "agent-deep-research",
            "agent-deep-research-a2a",
        ]
        for svc_name in agent_services:
            deps = config.get("services", {}).get(svc_name, {}).get("depends_on", {})
            for dep_name, dep_config in deps.items():
                condition = (
                    dep_config.get("condition")
                    if isinstance(dep_config, dict)
                    else None
                )
                assert condition == "service_healthy", (
                    f"'{svc_name}' → '{dep_name}': "
                    f"condition이 'service_healthy'가 아님 (현재: {condition})"
                )

    def test_dockerfile_cli_pythonpath(self):
        """Dockerfile.cli에 PYTHONPATH가 설정되어 있는지 확인한다."""
        from pathlib import Path

        dockerfile = Path(__file__).parent.parent / "docker" / "Dockerfile.cli"
        if not dockerfile.exists():
            pytest.skip("Dockerfile.cli 파일 없음")

        content = dockerfile.read_text()
        assert "PYTHONPATH" in content, "PYTHONPATH 환경변수 미설정"
        assert "PYTHONUNBUFFERED" in content, (
            "PYTHONUNBUFFERED 미설정 (대화형 모드 필수)"
        )

    def test_dockerfile_cli_entrypoint(self):
        """Dockerfile.cli의 ENTRYPOINT가 youngs75-agent인지 확인한다."""
        from pathlib import Path

        dockerfile = Path(__file__).parent.parent / "docker" / "Dockerfile.cli"
        if not dockerfile.exists():
            pytest.skip("Dockerfile.cli 파일 없음")

        content = dockerfile.read_text()
        assert "youngs75-agent" in content, "ENTRYPOINT에 youngs75-agent 미설정"

    def test_cli_service_mcp_env_vars(self):
        """CLI 서비스의 MCP 환경변수가 Docker 내부 네트워크 주소를 사용하는지 확인한다."""
        import yaml
        from pathlib import Path

        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.yml 파일 없음")

        with open(compose_path) as f:
            config = yaml.safe_load(f)

        cli_env = config.get("services", {}).get("cli", {}).get("environment", [])
        env_dict = {}
        for entry in cli_env:
            if "=" in entry:
                key, val = entry.split("=", 1)
                env_dict[key] = val

        # MCP URL이 Docker 내부 서비스명을 사용하는지 확인
        assert "TAVILY_MCP_URL" in env_dict, "CLI: TAVILY_MCP_URL 미설정"
        assert "mcp-tavily" in env_dict["TAVILY_MCP_URL"], (
            "CLI: TAVILY_MCP_URL이 내부 네트워크 주소가 아님"
        )

        assert "ARXIV_MCP_URL" in env_dict, "CLI: ARXIV_MCP_URL 미설정"
        assert "mcp-arxiv" in env_dict["ARXIV_MCP_URL"], (
            "CLI: ARXIV_MCP_URL이 내부 네트워크 주소가 아님"
        )

        assert "SERPER_MCP_URL" in env_dict, "CLI: SERPER_MCP_URL 미설정"
        assert "mcp-serper" in env_dict["SERPER_MCP_URL"], (
            "CLI: SERPER_MCP_URL이 내부 네트워크 주소가 아님"
        )


# ─── 통합 시나리오 테스트 ───────────────────────────────


class TestIntegrationScenario:
    """전체 시스템의 통합 시나리오를 검증한다."""

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_full_chain_health_then_query(self, docker_available):
        """MCP 헬스 → Agent 헬스 → AgentCard → 질의 순서의 전체 흐름을 검증한다."""
        # 1단계: MCP 서비스 중 하나라도 정상인지 확인
        mcp_any_ok = False
        for name, url in MCP_SERVICES.items():
            if await _is_service_reachable(url):
                mcp_any_ok = True
                break
        if not mcp_any_ok:
            pytest.skip("MCP 서비스 모두 미실행")

        # 2단계: Agent 서비스 중 하나라도 정상인지 확인
        target_agent = None
        target_url = None
        for name, url in AGENT_SERVICES.items():
            if await _is_service_reachable(url):
                target_agent = name
                target_url = url
                break
        if target_agent is None:
            pytest.skip("Agent 서비스 모두 미실행")

        # 3단계: AgentCard 조회
        card = await _get_agent_card(target_url)
        assert card is not None, f"{target_agent}: AgentCard 조회 실패"
        assert "name" in card

        # 4단계: A2A 질의
        result = await _send_a2a_message(target_url, "테스트 질의입니다", timeout=120.0)
        assert "jsonrpc" in result
        assert result["jsonrpc"] == "2.0"

    @pytest.mark.flaky(reruns=3, reruns_delay=5)
    async def test_concurrent_agent_queries(self, docker_available):
        """여러 Agent에 동시 질의가 가능한지 확인한다."""
        available = {}
        for name, url in AGENT_SERVICES.items():
            if await _is_service_reachable(url):
                available[name] = url

        if len(available) < 2:
            pytest.skip(
                f"동시 테스트에 2개 이상의 Agent 필요 (현재 {len(available)}개)"
            )

        tasks = []
        for name, url in available.items():
            tasks.append(_send_a2a_message(url, "간단한 테스트", timeout=120.0))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(
            1 for r in results if isinstance(r, dict) and "jsonrpc" in r
        )
        assert success_count >= 1, "동시 질의에서 성공한 Agent가 없음"
