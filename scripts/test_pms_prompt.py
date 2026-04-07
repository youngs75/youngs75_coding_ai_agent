"""PMS 샘플 프롬프트 E2E 테스트.

Orchestrator → Planner → CodingAssistant(Phase 순차실행) 전체 파이프라인을
비대화형으로 실행하여 산출물과 Langfuse 트레이싱을 확인한다.

HITL(계획 승인)은 자동 승인으로 처리.
"""

import asyncio
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# 워크스페이스를 별도 디렉토리로 설정 (프로젝트 오염 방지)
WORKSPACE = os.getenv("PMS_WORKSPACE", "/tmp/pms_workspace")
os.makedirs(WORKSPACE, exist_ok=True)
os.environ["CODE_TOOLS_WORKSPACE"] = WORKSPACE

os.environ.setdefault("SKILLS_DIR", "./data/skills")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pms_test")

# ── 샘플 프롬프트 ──

PMS_PROMPT = """\
## Task

PMS (project manage system) 시스템을 구성하는 프로젝트.

## Process

1. PRD 파일을 만들고
2. PRD 파일을 기반으로 작업을 원자 단위 작업으로 분해할 것.
3. 작업에 대한 명세는 구체적이어야하며, 추상적인 문구를 배제하고 확실히 개발 방향을 명시할 것.
4. 개발 명세서를 Spec Driven Development 기반으로 도출할 것.
5. 위 내용으로 도출된 개발 명세서를 기반으로 개발 작업을 수행할 것.
 (단, Test Driven Development 방식으로 개발하는 것을 필히 준수해야함)

## 세부 요구사항
1. 사용자 : it 회사의 프로젝트을 수행하는 PM
2. 관리자 : it 회사의 임원 및 PMO 조직
3. 웹, 모바일에서 접속 가능
4. 사용자는 프로젝트 정보를 입력한다. (프로젝트명, 프로젝트코드, 고객사, 설계자, 개발자, 프로젝트 일정)
5. 관리자는 등록된 프로젝트와 일자를 관리한다.
6. 사용자가 사용하기 편하게 해야 한다.
7. 기본적으로 간트 차트 기능이 구현되어야 한다.
"""


async def run_pms_test():
    """Orchestrator 전체 파이프라인 실행 (HITL 자동 승인)."""
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from youngs75_a2a.agents.orchestrator.agent import OrchestratorAgent
    from youngs75_a2a.agents.orchestrator.config import OrchestratorConfig
    from youngs75_a2a.eval_pipeline.observability.callback_handler import (
        create_langfuse_handler,
        safe_flush,
    )

    # Langfuse 콜백 핸들러
    langfuse_handler = create_langfuse_handler()
    callbacks = [langfuse_handler] if langfuse_handler else []

    # Orchestrator 생성 (checkpointer 필수 — HITL을 위해)
    checkpointer = MemorySaver()
    orch_config = OrchestratorConfig()
    agent = await OrchestratorAgent.create(
        config=orch_config,
        checkpointer=checkpointer,
    )

    thread_id = f"pms_test_{int(time.time())}"
    run_config = {
        "configurable": {
            **orch_config.to_langgraph_configurable(),
            "thread_id": thread_id,
        },
        "callbacks": callbacks,
    }

    logger.info("=" * 60)
    logger.info("PMS 프롬프트 실행 시작")
    logger.info("워크스페이스: %s", WORKSPACE)
    logger.info("Thread ID: %s", thread_id)
    logger.info("=" * 60)

    start_time = time.time()

    # 1단계: 초기 실행 (classify → plan → interrupt)
    logger.info("[1/3] Orchestrator 초기 실행 (classify + plan)...")
    result = await agent.graph.ainvoke(
        {"messages": [HumanMessage(content=PMS_PROMPT)]},
        config=run_config,
    )

    # HITL 체크: interrupt 발생 여부 확인
    state = await agent.graph.aget_state(run_config)
    pending = state.tasks if hasattr(state, "tasks") else []

    if pending:
        # 계획이 생성됨 → interrupt에서 대기 중
        # state에서 task_plan 확인
        plan_text = state.values.get("task_plan", "")
        logger.info("[2/3] Planner 계획 수립 완료 (%d chars)", len(plan_text or ""))

        # 계획 내용 출력
        print("\n" + "=" * 60)
        print("📋 Planner가 수립한 계획:")
        print("=" * 60)
        if plan_text:
            print(plan_text[:3000])
            if len(plan_text) > 3000:
                print(f"\n... ({len(plan_text)} chars, 3000자만 표시)")
        print("=" * 60 + "\n")

        # 자동 승인 → delegate(Phase 순차실행) 진행
        logger.info("[3/3] 계획 자동 승인 → Phase 순차실행 시작...")
        result = await agent.graph.ainvoke(
            Command(resume=True),
            config=run_config,
        )
    else:
        logger.info("[2/3] Planner interrupt 없음 — 직접 실행 결과")

    elapsed = time.time() - start_time

    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 실행 결과:")
    print("=" * 60)

    # agent_response 추출
    final_state = await agent.graph.aget_state(run_config)
    values = final_state.values

    agent_response = values.get("agent_response", "")
    phase_results = values.get("phase_results", [])
    task_plan = values.get("task_plan", "")

    if agent_response:
        print("\n📝 에이전트 응답:")
        print(agent_response[:5000])
        if len(agent_response) > 5000:
            print(f"\n... ({len(agent_response)} chars)")

    if phase_results:
        print("\n📦 Phase 결과:")
        for pr in phase_results:
            icon = {"success": "✓", "failed": "✗", "skipped": "⊘"}.get(pr.get("status", ""), "?")
            print(f"  {icon} {pr.get('phase_id', '?')}: {pr.get('title', '')} [{pr.get('status')}]")
            for f in pr.get("written_files", []):
                print(f"    - {f}")
            if pr.get("error"):
                print(f"    ⚠ 오류: {pr['error'][:100]}")

    print(f"\n⏱ 총 소요 시간: {elapsed:.1f}초")

    # 워크스페이스 파일 목록
    print("\n📁 생성된 파일 (워크스페이스):")
    for root, dirs, files in os.walk(WORKSPACE):
        # .venv, __pycache__ 제외
        dirs[:] = [d for d in dirs if d not in (".venv", "__pycache__", "node_modules", ".git")]
        for f in sorted(files):
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, WORKSPACE)
            size = os.path.getsize(fpath)
            print(f"  {rel} ({size:,}B)")

    # Langfuse flush
    if langfuse_handler:
        safe_flush(langfuse_handler)
        print("\n🔍 Langfuse 트레이싱 전송 완료")
        print(f"   Trace: https://cloud.langfuse.com → pms_e2e_test")

    print("\n" + "=" * 60)

    return {
        "elapsed": elapsed,
        "phase_results": phase_results,
        "agent_response": agent_response,
        "task_plan": task_plan,
    }


if __name__ == "__main__":
    result = asyncio.run(run_pms_test())
    # 결과를 JSON으로도 저장
    summary_path = os.path.join(WORKSPACE, "_test_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "elapsed_s": result["elapsed"],
                "phases": len(result.get("phase_results", [])),
                "success_count": sum(
                    1 for p in result.get("phase_results", [])
                    if p.get("status") == "success"
                ),
                "response_length": len(result.get("agent_response", "")),
                "plan_length": len(result.get("task_plan", "")),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"요약 저장: {summary_path}")
