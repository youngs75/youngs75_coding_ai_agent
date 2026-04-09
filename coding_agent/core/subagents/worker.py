"""SubAgent 워커 프로세스 엔트리포인트.

SubAgentProcessManager가 spawn한 자식 프로세스에서 실행된다.
에이전트를 초기화하고, 작업을 수행하고, 결과를 stdout JSON으로 출력한다.

실행: python -m coding_agent.core.subagents.worker \
        --agent-type coding_assistant --task-message-file /tmp/task.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from langchain_core.messages import HumanMessage

# stderr로만 로그 출력 (stdout은 JSON 결과 전용)
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[worker:%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# 에이전트 타입 → (모듈 경로, 클래스명, 설정 모듈, 설정 클래스, invoke 결과 추출 방법)
_AGENT_REGISTRY: dict[str, dict] = {
    "coding_assistant": {
        "agent_module": "coding_agent.agents.coding_assistant.agent",
        "agent_class": "CodingAssistantAgent",
        "config_module": "coding_agent.agents.coding_assistant.config",
        "config_class": "CodingConfig",
        "use_skill_registry": True,
        "use_create_factory": True,
        "invoke_input_fn": "_coding_input",
        "extract_fn": "_coding_extract",
    },
    "deep_research": {
        "agent_module": "coding_agent.agents.deep_research.agent",
        "agent_class": "DeepResearchAgent",
        "config_module": "coding_agent.agents.deep_research.config",
        "config_class": "ResearchConfig",
        "use_skill_registry": False,
        "use_create_factory": False,
        "invoke_input_fn": "_simple_input",
        "extract_fn": "_research_extract",
    },
    "simple_react": {
        "agent_module": "coding_agent.agents.simple_react.agent",
        "agent_class": "SimpleMCPReActAgent",
        "config_module": "coding_agent.agents.simple_react.config",
        "config_class": "SimpleReActConfig",
        "use_skill_registry": False,
        "use_create_factory": True,
        "invoke_input_fn": "_simple_input",
        "extract_fn": "_react_extract",
    },
}


# ── invoke 입력 생성 함수 ──


def _coding_input(message: str, task_plan: str | None) -> dict:
    effective = f"{message}\n\n{task_plan}" if task_plan else message
    return {
        "messages": [HumanMessage(content=effective)],
        "iteration": 0,
        "max_iterations": 11,
    }


def _simple_input(message: str, task_plan: str | None) -> dict:
    return {"messages": [HumanMessage(content=message)]}


# ── 결과 추출 함수 ──


def _coding_extract(result: dict) -> dict:
    code = result.get("generated_code") or ""
    if not code:
        msgs = result.get("messages", [])
        code = msgs[-1].content if msgs else ""
    written = result.get("written_files", [])
    return {
        "result": code,
        "written_files": written,
        "test_passed": result.get("test_passed", True),
        "exit_reason": result.get("exit_reason", ""),
    }


def _research_extract(result: dict) -> dict:
    return {"result": result.get("final_report", ""), "written_files": []}


def _react_extract(result: dict) -> dict:
    msgs = result.get("messages", [])
    text = msgs[-1].content if msgs else ""
    return {"result": text, "written_files": []}


_INPUT_FNS = {
    "_coding_input": _coding_input,
    "_simple_input": _simple_input,
}

_EXTRACT_FNS = {
    "_coding_extract": _coding_extract,
    "_research_extract": _research_extract,
    "_react_extract": _react_extract,
}


async def _run_agent(
    agent_type: str,
    task_message: str,
    task_plan: str | None = None,
) -> dict:
    """에이전트를 초기화하고 작업을 수행한다."""
    import importlib

    spec = _AGENT_REGISTRY.get(agent_type)
    if not spec:
        raise ValueError(f"알 수 없는 에이전트 타입: {agent_type}")

    # 에이전트 클래스 동적 import
    agent_mod = importlib.import_module(spec["agent_module"])
    agent_cls = getattr(agent_mod, spec["agent_class"])

    config_mod = importlib.import_module(spec["config_module"])
    config_cls = getattr(config_mod, spec["config_class"])

    # 에이전트 생성
    create_kwargs: dict = {"config": config_cls()}

    # 스킬 레지스트리 초기화 (coding_assistant)
    if spec.get("use_skill_registry"):
        import os
        from pathlib import Path

        from coding_agent.core.skills.loader import SkillLoader
        from coding_agent.core.skills.registry import SkillRegistry

        skill_registry = SkillRegistry()
        skills_dir = os.getenv(
            "SKILLS_DIR",
            str(Path(__file__).resolve().parent.parent.parent / "data" / "skills"),
        )
        if Path(skills_dir).is_dir():
            loader = SkillLoader(skills_dir)
            skill_registry = SkillRegistry(loader=loader)
            skill_registry.discover()
        create_kwargs["skill_registry"] = skill_registry

    if spec.get("use_create_factory"):
        agent = await agent_cls.create(**create_kwargs)
    else:
        agent = agent_cls(**create_kwargs)

    # invoke 입력 생성
    input_fn = _INPUT_FNS[spec["invoke_input_fn"]]
    invoke_input = input_fn(task_message, task_plan)

    # 실행
    result = await agent.graph.ainvoke(invoke_input)

    # 결과 추출
    extract_fn = _EXTRACT_FNS[spec["extract_fn"]]
    return extract_fn(result)


def _output_json(data: dict) -> None:
    """stdout에 JSON을 출력한다 (ProcessManager가 파싱)."""
    print(json.dumps(data, ensure_ascii=False), flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="SubAgent worker process")
    parser.add_argument("--agent-type", required=True, help="에이전트 타입")
    parser.add_argument("--task-message-file", required=True, help="태스크 메시지 파일 경로")
    parser.add_argument("--task-plan-file", default=None, help="태스크 계획 파일 경로")
    parser.add_argument("--parent-id", default="", help="부모 에이전트 ID")
    args = parser.parse_args()

    # 태스크 메시지 읽기
    with open(args.task_message_file, encoding="utf-8") as f:
        task_message = f.read()

    task_plan = None
    if args.task_plan_file:
        with open(args.task_plan_file, encoding="utf-8") as f:
            task_plan = f.read()

    start = time.time()

    try:
        logger.info("에이전트 시작: type=%s, parent=%s", args.agent_type, args.parent_id)
        extracted = await _run_agent(args.agent_type, task_message, task_plan)
        duration = time.time() - start

        _output_json({
            "status": "completed",
            "result": extracted.get("result"),
            "written_files": extracted.get("written_files", []),
            "test_passed": extracted.get("test_passed", True),
            "exit_reason": extracted.get("exit_reason", ""),
            "duration_s": round(duration, 2),
            "token_usage": {},
            "error": None,
        })
        logger.info("에이전트 완료: %.1fs", duration)

    except Exception as e:
        duration = time.time() - start
        logger.error("에이전트 실패: %s", e, exc_info=True)
        _output_json({
            "status": "failed",
            "result": None,
            "written_files": [],
            "duration_s": round(duration, 2),
            "token_usage": {},
            "error": str(e),
        })
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
