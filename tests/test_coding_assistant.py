"""Coding Assistant 에이전트 테스트.

Step 1: 프레임워크 자체 검증 (LLM 호출 없이 그래프 구조 확인)
Step 2: LLM 연동 테스트 (API 키 필요)
"""

import asyncio
import json
import sys

from langchain_core.messages import HumanMessage


def test_step1_graph_structure():
    """그래프 노드/엣지 구조가 올바른지 확인한다."""
    from youngs75_a2a.agents.coding_assistant import CodingAssistantAgent, CodingConfig

    config = CodingConfig()
    agent = CodingAssistantAgent(config=config)
    # CodingAssistantAgent는 MCP 비동기 로딩을 위해 auto_build=False.
    # 그래프 구조 검증만 하므로 수동으로 빌드한다.
    agent.build_graph()

    assert agent.graph is not None, "그래프가 빌드되지 않음"

    # 노드 확인
    node_names = set(agent.graph.get_graph().nodes.keys())
    expected = {
        "parse_request",
        "execute_code",
        "verify_result",
        "__start__",
        "__end__",
    }
    assert expected.issubset(node_names), f"누락된 노드: {expected - node_names}"

    print("[PASS] 그래프 구조 검증 통과")
    print(f"  노드: {node_names}")


def test_step1_safety_envelope():
    """ActionValidator가 위험 코드를 차단하는지 확인한다."""
    from youngs75_a2a.core.action_validator import ActionValidator

    validator = ActionValidator()

    # 안전한 코드
    safe_code = 'def hello():\n    return "hello world"'
    report = validator.validate(safe_code)
    assert report.is_safe, f"안전한 코드가 차단됨: {report.summary()}"

    # 시크릿 노출
    secret_code = 'API_KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"'
    report = validator.validate(secret_code)
    assert not report.is_safe, "시크릿 코드가 통과됨"
    assert "secret_exposure" in report.blocked_rules

    # 위험 명령
    danger_code = 'os.system("rm -rf /")'
    report = validator.validate(danger_code)
    assert not report.is_safe, "위험 명령이 통과됨"
    assert "dangerous_command" in report.blocked_rules

    # 파일 확장자 검증
    report = validator.validate(safe_code, target_files=["main.py"])
    assert report.is_safe

    report = validator.validate(safe_code, target_files=["virus.exe"])
    assert not report.is_safe
    assert "file_extension" in report.blocked_rules

    print("[PASS] Safety Envelope 검증 통과")


async def test_step2_llm_integration():
    """LLM 연동 전체 파이프라인 테스트."""
    from youngs75_a2a.agents.coding_assistant import CodingAssistantAgent, CodingConfig

    config = CodingConfig()
    agent = await CodingAssistantAgent.create(config=config)

    result = await agent.graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="파이썬으로 피보나치 수열의 n번째 값을 반환하는 함수를 작성해줘"
                )
            ],
            "iteration": 0,
            "max_iterations": 2,
        }
    )

    print("\n[결과]")
    print(
        f"  parse_result: {json.dumps(result.get('parse_result', {}), ensure_ascii=False, indent=2)}"
    )
    print(
        f"  verify_result: {json.dumps(result.get('verify_result', {}), ensure_ascii=False, indent=2)}"
    )
    print(f"  iteration: {result.get('iteration')}")
    print(f"  execution_log: {result.get('execution_log')}")
    print(f"\n  generated_code (앞 500자):\n{result.get('generated_code', '')[:500]}")
    print("\n[PASS] LLM 연동 테스트 통과")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "1"

    if step == "1":
        print("=== Step 1: 프레임워크 검증 (LLM 불필요) ===\n")
        test_step1_graph_structure()
        test_step1_safety_envelope()
    elif step == "2":
        print("=== Step 2: LLM 연동 테스트 ===\n")
        asyncio.run(test_step2_llm_integration())
    else:
        print(f"알 수 없는 step: {step}")
