"""CodingAssistant 전체 파이프라인 E2E 디버그.

Docker MCP 서버 + LiteLLM Proxy를 실제로 사용하여 완전한 E2E를 수행한다.
"""
import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


async def main():
    from coding_agent.agents.coding_assistant.agent import CodingAssistantAgent
    from coding_agent.agents.coding_assistant.config import CodingConfig

    workspace = tempfile.mkdtemp(prefix="debug_kanban_")
    os.environ["CODE_TOOLS_WORKSPACE"] = workspace
    os.environ["LITELLM_PROXY_URL"] = "http://localhost:4000/v1"
    os.environ["CODE_TOOLS_MCP_URL"] = "http://localhost:3003/mcp/"

    print(f"Workspace: {workspace}")

    config = CodingConfig()
    # async_init → MCP 도구 로딩 포함
    agent = await CodingAssistantAgent.create(config=config)

    print(f"MCP 도구: {len(agent._tools)}개 — {[t.name for t in agent._tools[:5]]}...")
    print(f"Model: {getattr(agent._get_gen_model(), 'model_name', '?')}")

    state = {
        "messages": [
            HumanMessage(content=(
                "## Phase 1/3: 백엔드 API 서버 구현\n\n"
                "### 사용자 요청\n"
                "칸반 보드 앱의 백엔드를 Flask + SQLAlchemy로 구현해줘.\n\n"
                "### 현재 페이즈 지시사항\n"
                "Flask 앱을 생성하세요. Board, Column, Card 모델을 정의합니다.\n"
                "routes.py에서 CRUD API를 구현하세요.\n"
                "tests/test_api.py에서 테스트를 작성하세요.\n\n"
                "### 생성 필수 파일 체크리스트\n"
                "- [ ] backend/__init__.py\n"
                "- [ ] backend/models.py\n"
                "- [ ] backend/routes.py\n"
                "- [ ] tests/test_api.py\n"
                "- [ ] requirements.txt\n"
            ))
        ],
        "parse_result": {"task_type": "generate", "language": "python", "description": "칸반 보드 백엔드"},
        "planned_files": ["backend/__init__.py", "backend/models.py", "backend/routes.py", "tests/test_api.py", "requirements.txt"],
        "execution_log": [],
        "iteration": 0,
        "max_iterations": 3,
        "tool_call_count": 0,
        "written_files": [],
        "generated_code": "",
        "skill_context": [],
        "project_context": [],
    }

    print("\n=== 그래프 실행 (RETRIEVE_MEMORY → GENERATE → RUN_TESTS → ...) ===")
    result = await agent.graph.ainvoke(state)

    print(f"\n{'='*60}")
    print(f"written_files: {result.get('written_files', [])}")
    print(f"test_passed: {result.get('test_passed')}")
    print(f"iteration: {result.get('iteration', 0)}")
    print(f"tool_call_count: {result.get('tool_call_count', 0)}")

    test_output = (result.get("test_output", "") or "")
    if test_output:
        print(f"\ntest_output (마지막 500자):\n{test_output[-500:]}")

    print(f"\nexecution_log:")
    for entry in result.get("execution_log", []):
        print(f"  {entry}")

    print(f"\n=== Workspace 파일 ===")
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", "node_modules")]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), workspace)
            size = os.path.getsize(os.path.join(root, f))
            print(f"  {rel} ({size}B)")


if __name__ == "__main__":
    asyncio.run(main())
