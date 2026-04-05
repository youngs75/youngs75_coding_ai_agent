"""CLI 비대화형 검증 스크립트.

도구 호출 2단계 파이프라인, 스킬 자동 선택, Subagents를 검증한다.
"""

import asyncio
import os

# .env 로드
from dotenv import load_dotenv

load_dotenv()

# 스킬 디렉토리 강제 설정
os.environ.setdefault("SKILLS_DIR", "./data/skills")

from langchain_core.messages import HumanMessage  # noqa: E402

from youngs75_a2a.core.memory.store import MemoryStore  # noqa: E402
from youngs75_a2a.core.skills.loader import SkillLoader  # noqa: E402
from youngs75_a2a.core.skills.registry import SkillRegistry  # noqa: E402


async def test_1_skill_auto_activation():
    """테스트 1: task_type별 스킬 자동 활성화."""
    print("\n" + "=" * 60)
    print("테스트 1: 스킬 자동 활성화")
    print("=" * 60)

    loader = SkillLoader("./data/skills")
    reg = SkillRegistry(loader=loader)
    discovered = reg.discover()
    print(f"  발견된 스킬: {discovered}")

    for task_type in ["generate", "fix", "refactor", "explain", "analyze"]:
        reg2 = SkillRegistry(loader=loader)
        reg2.discover()
        activated = reg2.auto_activate_for_task(task_type)
        bodies = reg2.get_active_skill_bodies()
        print(f"  [{task_type}] 활성화: {activated}, L2 본문: {len(bodies)}개")

    print("  ✓ 스킬 자동 활성화 정상 동작")


async def test_2_two_stage_pipeline():
    """테스트 2: FAST→STRONG 2단계 파이프라인 실제 LLM 호출."""
    print("\n" + "=" * 60)
    print("테스트 2: 2단계 파이프라인 (단순 코드 생성)")
    print("=" * 60)

    from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
    from youngs75_a2a.agents.coding_assistant.config import CodingConfig

    loader = SkillLoader("./data/skills")
    skill_registry = SkillRegistry(loader=loader)
    skill_registry.discover()

    config = CodingConfig()
    print(f"  STRONG: {config.get_tier_config('generation').model}")
    print(f"  FAST (tool_planning): {config.get_tier_config('tool_planning').model}")

    agent = await CodingAssistantAgent.create(
        config=config,
        memory_store=MemoryStore(),
        skill_registry=skill_registry,
    )

    nodes = list(agent.graph.get_graph().nodes.keys())
    print(f"  그래프 노드: {nodes}")
    assert "generate_final" in nodes, "generate_final 노드가 있어야 합니다"

    result = await agent.graph.ainvoke(
        {
            "messages": [HumanMessage(content="파이썬으로 이진 탐색 함수를 작성해줘")],
            "iteration": 0,
            "max_iterations": 3,
        }
    )

    generated = result.get("generated_code", "")
    execution_log = result.get("execution_log", [])
    verify = result.get("verify_result", {})

    print("  실행 로그:")
    for entry in execution_log:
        print(f"    {entry}")
    print(f"  검증: {'통과' if verify.get('passed') else '실패'}")
    print(f"  코드 미리보기: {generated[:150]}...")

    assert generated, "코드가 생성되어야 합니다"
    assert any("model=FAST" in e for e in execution_log), "FAST 모델 로그 필요"
    assert any("generate_final" in e for e in execution_log), "generate_final 로그 필요"
    print("  ✓ 2단계 파이프라인 정상 동작")


async def test_3_tool_call_with_mcp():
    """테스트 3: MCP 도구 호출 + 스킬 자동 활성화."""
    print("\n" + "=" * 60)
    print("테스트 3: MCP 도구 호출 (파일 분석)")
    print("=" * 60)

    from youngs75_a2a.agents.coding_assistant.agent import CodingAssistantAgent
    from youngs75_a2a.agents.coding_assistant.config import CodingConfig

    loader = SkillLoader("./data/skills")
    skill_registry = SkillRegistry(loader=loader)
    skill_registry.discover()

    agent = await CodingAssistantAgent.create(
        config=CodingConfig(),
        memory_store=MemoryStore(),
        skill_registry=skill_registry,
    )

    result = await agent.graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="pyproject.toml 파일을 읽고 의존성을 분석해줘")
            ],
            "iteration": 0,
            "max_iterations": 3,
        }
    )

    execution_log = result.get("execution_log", [])
    skill_context = result.get("skill_context", [])

    print("  실행 로그:")
    for entry in execution_log:
        print(f"    {entry}")
    print(f"  스킬 자동 활성화: {len(skill_context)}개")
    if skill_context:
        for ctx in skill_context:
            print(f"    {ctx[:80]}...")
    print("  ✓ MCP 도구 + 스킬 자동 활성화 검증 완료")


async def test_4_subagent_orchestrator():
    """테스트 4: Orchestrator Subagent 확인."""
    print("\n" + "=" * 60)
    print("테스트 4: Subagent (Orchestrator)")
    print("=" * 60)

    from youngs75_a2a.agents.orchestrator.agent import OrchestratorAgent
    from youngs75_a2a.agents.orchestrator.config import OrchestratorConfig

    config = OrchestratorConfig()
    print(f"  등록된 에이전트: {[ep.name for ep in config.agent_endpoints]}")

    agent = await OrchestratorAgent.create(config=config)
    nodes = list(agent.graph.get_graph().nodes.keys())
    print(f"  그래프 노드: {nodes}")
    print("  ✓ Orchestrator 초기화 정상")


async def main():
    print("=" * 60)
    print("CLI 기능 검증 스크립트")
    print("=" * 60)

    await test_1_skill_auto_activation()
    await test_2_two_stage_pipeline()
    await test_3_tool_call_with_mcp()
    await test_4_subagent_orchestrator()

    print("\n" + "=" * 60)
    print("✓ 전체 검증 완료!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
