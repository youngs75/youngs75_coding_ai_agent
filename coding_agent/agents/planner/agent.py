"""Planner Agent — 태스크 분석 및 구현 계획 수립.

Claude Code Plan Agent 패턴:
- STRONG 모델로 아키텍처 설계 및 태스크 분해
- Read-only 도구로 기존 프로젝트 탐색
- 웹 검색으로 외부 API 문서 조사
- 구조화된 실행 계획(TaskPlan)을 출력하여 코딩 에이전트에 전달

그래프 플로우:
    analyze_task → [_route_after_analyze]
      ├─ needs_research  → research_external → explore_context → create_plan → END
      ├─ moderate/complex → explore_context → create_plan → END
      └─ simple          → create_plan → END
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from coding_agent.core.abort_controller import AbortController
from coding_agent.core.base_agent import BaseGraphAgent
from coding_agent.core.mcp_loader import MCPToolLoader
from coding_agent.core.middleware import (
    MemoryMiddleware,
    MiddlewareChain,
    ModelRequest as MWRequest,
    ResilienceMiddleware,
)
from coding_agent.core.tool_call_utils import tc_args, tc_id, tc_name

from .config import PlannerConfig
from .prompts import (
    ANALYZE_SYSTEM_PROMPT,
    EXPLORE_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    RESEARCH_SUMMARIZE_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from .schemas import PlannerState, TaskPlan, validate_task_plan

logger = logging.getLogger(__name__)


def _parse_follow_up_queries(summary: str) -> list[str]:
    """요약 텍스트에서 FOLLOW_UP_QUERIES 섹션의 쿼리를 추출한다."""
    marker = "FOLLOW_UP_QUERIES:"
    idx = summary.find(marker)
    if idx == -1:
        return []
    section = summary[idx + len(marker) :]
    queries = []
    for line in section.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            queries.append(line[2:].strip())
        elif line.startswith("* "):
            queries.append(line[2:].strip())
    return queries


def _remove_follow_up_section(summary: str) -> str:
    """요약 텍스트에서 FOLLOW_UP_QUERIES 섹션을 제거한다."""
    marker = "FOLLOW_UP_QUERIES:"
    idx = summary.find(marker)
    if idx == -1:
        return summary
    return summary[:idx].rstrip()


async def _execute_tool_safely(tool: Any, args: dict) -> str:
    """도구를 안전하게 실행하고 결과를 문자열로 반환한다."""
    try:
        result = await tool.ainvoke(args)
        return str(result)[:3000] if result else ""
    except Exception as e:
        return f"도구 실행 오류: {e}"


class PlannerAgent(BaseGraphAgent):
    """태스크 분석 및 구현 계획 전문 에이전트.

    analyze_task → [route] → (research_external →) explore_context → create_plan → END
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "ANALYZE": "analyze_task",
        "RESEARCH": "research_external",
        "EXPLORE": "explore_context",
        "CREATE_PLAN": "create_plan",
    }

    def __init__(
        self,
        *,
        config: PlannerConfig | None = None,
        model: BaseChatModel | None = None,
        **kwargs: Any,
    ) -> None:
        self._planner_config = config or PlannerConfig()
        self._mcp_loader = MCPToolLoader(self._planner_config.mcp_servers)
        self._web_search_loader = MCPToolLoader(
            self._planner_config.web_search_mcp_servers
        )
        self._tools: list[Any] = []
        self._web_search_tools: list[Any] = []
        self._explicit_model = model

        # AbortController + 미들웨어 체인
        self._abort_controller = AbortController()
        self._middleware_chain = MiddlewareChain([
            ResilienceMiddleware(abort_controller=self._abort_controller),
            MemoryMiddleware(),
        ])

        kwargs.pop("auto_build", None)
        super().__init__(
            config=self._planner_config,
            model=model,
            state_schema=PlannerState,
            agent_name="PlannerAgent",
            auto_build=False,
            **kwargs,
        )

    async def async_init(self) -> None:
        """MCP 도구를 비동기로 로드하고 read-only 필터링."""
        # 코드 탐색 도구 로드
        all_tools = await self._mcp_loader.load()
        allowed = set(self._planner_config.allowed_tools)
        self._tools = [t for t in all_tools if getattr(t, "name", None) in allowed]
        logger.info(
            "Planner 도구 %d개 로드 (read-only): %s",
            len(self._tools),
            [getattr(t, "name", "") for t in self._tools],
        )

        # 웹 검색 도구 로드 (실패 시 graceful skip)
        try:
            all_web_tools = await self._web_search_loader.load()
            web_allowed = set(self._planner_config.web_search_tools)
            self._web_search_tools = [
                t for t in all_web_tools if getattr(t, "name", None) in web_allowed
            ]
            logger.info(
                "Planner 웹 검색 도구 %d개 로드: %s",
                len(self._web_search_tools),
                [getattr(t, "name", "") for t in self._web_search_tools],
            )
        except Exception as e:
            logger.warning("웹 검색 MCP 로딩 실패 (research 단계 생략됨): %s", e)
            self._web_search_tools = []

    def _get_model(self) -> BaseChatModel:
        """STRONG 모델 반환."""
        return self._explicit_model or self._planner_config.get_model("planning")

    # ── 노드 구현 ──

    async def _analyze_task(self, state: PlannerState) -> dict[str, Any]:
        """사용자 요청의 복잡도를 분석한다.

        REASONING 티어 유지 — 계획 수립은 고성능 모델 필요.
        """
        # 턴 시작: abort 상태 리셋
        self._abort_controller.reset()

        user_request = state.get("user_request", "")
        if not user_request:
            # messages에서 추출
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    user_request = msg.content
                    break

        model = self._get_model()
        mw_request = MWRequest(
            system_message=ANALYZE_SYSTEM_PROMPT,
            messages=[HumanMessage(content=user_request)],
            metadata={
                "purpose": "planning",
                "request_timeout": self._planner_config.get_request_timeout("planning"),
            },
        )
        mw_response = await self._middleware_chain.invoke(mw_request, model)
        response = mw_response.message

        try:
            analysis = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            analysis = {
                "complexity": "moderate",
                "reason": "분석 실패 — moderate로 기본 처리",
                "estimated_files": 3,
                "needs_exploration": False,
                "needs_research": False,
                "research_queries": [],
            }

        return {
            "user_request": user_request,
            "task_plan": {
                "complexity": analysis.get("complexity", "moderate"),
                "_needs_research": analysis.get("needs_research", False),
                "_research_queries": analysis.get("research_queries", []),
            },
            "explored_context": [],
            "research_context": [],
            "messages": [response],
        }

    async def _research_external(self, state: PlannerState) -> dict[str, Any]:
        """웹 검색으로 외부 API 문서/정보를 조사한다.

        멀티라운드 리서치:
        1라운드: 분석 단계의 쿼리로 웹 검색 → 핵심 사실 요약
        2라운드: 미확인 항목이 있으면 요약이 제안한 후속 쿼리로 추가 검색 → 재요약
        """
        if not self._web_search_tools:
            return {"research_context": ["웹 검색 도구 없음 — 조사 생략"]}

        user_request = state.get("user_request", "")
        task_plan = state.get("task_plan", {})
        research_queries = task_plan.get("_research_queries", [])
        max_searches = self._planner_config.max_research_searches
        max_rounds = 2

        tool_descriptions = self._web_search_loader.get_tool_descriptions()
        model = self._get_model()
        search_model = model.bind_tools(self._web_search_tools)
        tools_by_name = {getattr(t, "name", None): t for t in self._web_search_tools}

        all_raw_results: list[str] = []
        research_entries: list[str] = []
        messages: list = []
        current_queries = research_queries

        for round_num in range(1, max_rounds + 1):
            if not current_queries:
                break

            # 검색 쿼리 힌트 구성
            query_hint = "\n\n## 검색 쿼리\n" + "\n".join(
                f"- {q}" for q in current_queries[:max_searches]
            )
            round_label = f"(라운드 {round_num})" if round_num > 1 else ""

            response = await search_model.ainvoke(
                [
                    SystemMessage(
                        content=RESEARCH_SYSTEM_PROMPT.format(
                            tool_descriptions=tool_descriptions,
                            max_searches=max_searches,
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"다음 작업에 필요한 외부 API/서비스 정보를 조사하세요 {round_label}:\n"
                            f"{user_request}{query_hint}"
                        )
                    ),
                ]
            )

            # 도구 호출 실행
            tool_calls = getattr(response, "tool_calls", None) or []
            messages.append(response)

            for call in tool_calls[:max_searches]:
                name = tc_name(call)
                args = tc_args(call)
                call_id = tc_id(call)

                if name in tools_by_name:
                    result = await _execute_tool_safely(tools_by_name[name], args)
                    query_label = args.get("query", args.get("q", ""))
                    research_entries.append(
                        f"[웹 검색: {query_label}]\n{result[:2000]}"
                    )
                    all_raw_results.append(result[:3000])
                else:
                    result = f"허용되지 않은 도구: {name}"

                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=call_id or f"call_{name}",
                        name=name or "unknown",
                    )
                )

            # 검색 결과 요약 + 후속 쿼리 추출
            if all_raw_results:
                combined_raw = "\n---\n".join(all_raw_results)
                summary_response = await model.ainvoke(
                    [
                        SystemMessage(content=RESEARCH_SUMMARIZE_PROMPT),
                        HumanMessage(
                            content=(
                                f"## 원래 요청\n{user_request}\n\n"
                                f"## 검색 결과\n{combined_raw[:6000]}"
                            )
                        ),
                    ]
                )
                summary_text = summary_response.content

                # FOLLOW_UP_QUERIES 파싱
                follow_up_queries = _parse_follow_up_queries(summary_text)

                if follow_up_queries and round_num < max_rounds:
                    # 미확인 항목 있음 → 후속 검색 진행
                    logger.info(
                        "리서치 라운드 %d: 미확인 항목 발견, 후속 쿼리 %d개 → 라운드 %d 진행",
                        round_num,
                        len(follow_up_queries),
                        round_num + 1,
                    )
                    current_queries = follow_up_queries
                    continue
                else:
                    # 모든 항목 확인됨 또는 최대 라운드 도달
                    # FOLLOW_UP_QUERIES 섹션 제거하고 요약만 저장
                    clean_summary = _remove_follow_up_section(summary_text)
                    research_entries.insert(
                        0, f"[조사 요약]\n{clean_summary}"
                    )
                    break

        return {
            "research_context": research_entries,
            "messages": messages,
        }

    async def _explore_context(self, state: PlannerState) -> dict[str, Any]:
        """Read-only 도구로 프로젝트 컨텍스트를 탐색한다."""
        if not self._tools:
            return {"explored_context": ["도구 없음 — 탐색 생략"]}

        user_request = state.get("user_request", "")
        tool_descriptions = self._mcp_loader.get_tool_descriptions()

        model = self._get_model()
        if self._tools:
            model = model.bind_tools(self._tools)

        # LLM 호출 전 abort 체크 (도구 바인딩된 모델은 middleware chain 미사용)
        self._abort_controller.check_or_raise()

        response = await model.ainvoke(
            [
                SystemMessage(
                    content=EXPLORE_SYSTEM_PROMPT.format(
                        tool_descriptions=tool_descriptions
                    )
                ),
                HumanMessage(
                    content=f"다음 작업을 위해 프로젝트를 탐색하세요:\n{user_request}"
                ),
            ]
        )

        # LLM 호출 후 abort 체크
        self._abort_controller.check_or_raise()

        # 도구 호출 실행 (최대 1 라운드)
        context_entries: list[str] = list(state.get("explored_context", []))
        tool_calls = getattr(response, "tool_calls", None) or []
        tools_by_name = {getattr(t, "name", None): t for t in self._tools}

        messages = [response]

        for call in tool_calls[:5]:  # 최대 5회
            # 도구 실행 전 abort 체크
            self._abort_controller.check_or_raise()

            name = tc_name(call)
            args = tc_args(call)
            call_id = tc_id(call)

            if name in tools_by_name:
                result = await _execute_tool_safely(tools_by_name[name], args)
                if name == "read_file":
                    path = args.get("path", "?")
                    context_entries.append(f"[{path}]\n{result[:2000]}")
                elif name == "list_directory":
                    context_entries.append(f"[디렉토리 구조]\n{result[:1500]}")
                elif name == "search_code":
                    context_entries.append(
                        f"[검색: {args.get('query', '')}]\n{result[:1500]}"
                    )
            else:
                result = f"허용되지 않은 도구: {name}"

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id or f"call_{name}",
                    name=name or "unknown",
                )
            )

        # 도구 실행 후 abort 체크
        self._abort_controller.check_or_raise()

        return {
            "explored_context": context_entries,
            "messages": messages,
        }

    async def _create_plan(self, state: PlannerState) -> dict[str, Any]:
        """구조화된 실행 계획을 생성한다."""
        user_request = state.get("user_request", "")
        explored_context = state.get("explored_context", [])
        research_context = state.get("research_context", [])

        context_str = (
            "\n\n".join(explored_context)
            if explored_context
            else "탐색된 컨텍스트 없음"
        )
        research_str = (
            "\n\n".join(research_context)
            if research_context
            else "조사된 외부 API 정보 없음"
        )

        model = self._get_model()
        mw_request = MWRequest(
            system_message=PLAN_SYSTEM_PROMPT.format(
                explored_context=context_str,
                research_context=research_str,
            ),
            messages=[HumanMessage(content=user_request)],
            metadata={
                "purpose": "planning",
                "request_timeout": self._planner_config.get_request_timeout("planning"),
            },
        )
        mw_response = await self._middleware_chain.invoke(mw_request, model)
        response = mw_response.message

        try:
            plan_data = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            # JSON 파싱 실패 시 마크다운 계획으로 폴백
            plan_data = {
                "complexity": state.get("task_plan", {}).get("complexity", "moderate"),
                "summary": "계획 생성 완료",
                "architecture": "",
                "file_structure": [],
                "phases": [
                    {
                        "id": "phase_1",
                        "title": "전체 구현",
                        "description": user_request,
                        "files": [],
                        "depends_on": [],
                        "instructions": response.content,
                    }
                ],
                "tech_stack": [],
                "constraints": [],
            }

        task_plan: TaskPlan = {
            "complexity": plan_data.get("complexity", "moderate"),
            "summary": plan_data.get("summary", ""),
            "architecture": plan_data.get("architecture", ""),
            "file_structure": plan_data.get("file_structure", []),
            "phases": plan_data.get("phases", []),
            "tech_stack": plan_data.get("tech_stack", []),
            "constraints": plan_data.get("constraints", []),
        }

        # Phase별 파일 수 하드 리밋 검증 (5개 초과 시 자동 분할)
        task_plan = validate_task_plan(task_plan)

        # 계획을 마크다운으로도 저장 (코딩 에이전트 프롬프트용)
        plan_text = self._format_plan_as_markdown(task_plan)

        return {
            "task_plan": task_plan,
            "plan_text": plan_text,
            "messages": [response],
        }

    @staticmethod
    def _format_plan_as_markdown(plan: TaskPlan) -> str:
        """TaskPlan을 마크다운 문자열로 포맷팅한다."""
        parts: list[str] = []
        parts.append("## 구현 계획\n")

        if plan.get("summary"):
            parts.append(f"**요약**: {plan['summary']}\n")

        if plan.get("architecture"):
            parts.append(f"### 아키텍처\n{plan['architecture']}\n")

        if plan.get("tech_stack"):
            parts.append("### 기술 스택\n- " + "\n- ".join(plan["tech_stack"]) + "\n")

        if plan.get("file_structure"):
            parts.append("### 파일 구조\n```")
            parts.append("\n".join(plan["file_structure"]))
            parts.append("```\n")

        phases = plan.get("phases", [])
        if phases:
            parts.append("### 구현 페이즈\n")
            for phase in phases:
                deps = (
                    f" (의존: {', '.join(phase.get('depends_on', []))})"
                    if phase.get("depends_on")
                    else ""
                )
                parts.append(
                    f"#### {phase.get('id', '?')}: {phase.get('title', '')}{deps}"
                )
                parts.append(f"{phase.get('description', '')}")
                if phase.get("files"):
                    parts.append(f"파일: {', '.join(phase['files'])}")
                if phase.get("instructions"):
                    parts.append(f"\n**지시사항**: {phase['instructions']}")
                parts.append("")

        if plan.get("constraints"):
            parts.append("### 제약 조건\n- " + "\n- ".join(plan["constraints"]))

        return "\n".join(parts)

    # ── 라우팅 ──

    def _route_after_analyze(self, state: PlannerState) -> str:
        """분석 후 다음 단계를 결정한다.

        라우팅 우선순위:
        1. 외부 API 리서치 필요 + 웹 검색 도구 가용 → RESEARCH
        2. 복잡도 moderate/complex + 코드 탐색 도구 가용 → EXPLORE
        3. 그 외 → CREATE_PLAN
        """
        task_plan = state.get("task_plan", {})
        complexity = task_plan.get("complexity", "simple")
        needs_research = task_plan.get("_needs_research", False)

        if needs_research and self._web_search_tools:
            return self.get_node_name("RESEARCH")

        if complexity in ("moderate", "complex") and self._tools:
            return self.get_node_name("EXPLORE")

        return self.get_node_name("CREATE_PLAN")

    # ── 그래프 구성 ──

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("ANALYZE"), self._analyze_task)
        graph.add_node(self.get_node_name("RESEARCH"), self._research_external)
        graph.add_node(self.get_node_name("EXPLORE"), self._explore_context)
        graph.add_node(self.get_node_name("CREATE_PLAN"), self._create_plan)

    def init_edges(self, graph: StateGraph) -> None:
        graph.set_entry_point(self.get_node_name("ANALYZE"))

        # analyze → (research or explore or create_plan)
        graph.add_conditional_edges(
            self.get_node_name("ANALYZE"),
            self._route_after_analyze,
        )

        # research → explore (리서치 후 항상 로컬 탐색도 수행)
        graph.add_edge(
            self.get_node_name("RESEARCH"),
            self.get_node_name("EXPLORE"),
        )

        # explore → create_plan
        graph.add_edge(
            self.get_node_name("EXPLORE"),
            self.get_node_name("CREATE_PLAN"),
        )

        # create_plan → END
        graph.add_edge(self.get_node_name("CREATE_PLAN"), END)
