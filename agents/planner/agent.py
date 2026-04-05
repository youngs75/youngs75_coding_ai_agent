"""Planner Agent — 태스크 분석 및 구현 계획 수립.

Claude Code Plan Agent 패턴:
- STRONG 모델로 아키텍처 설계 및 태스크 분해
- Read-only 도구로 기존 프로젝트 탐색
- 구조화된 실행 계획(TaskPlan)을 출력하여 코딩 에이전트에 전달

그래프 플로우:
    analyze_task → [needs_exploration?] → explore_context → create_plan → END
                                        → create_plan → END
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from youngs75_a2a.core.base_agent import BaseGraphAgent
from youngs75_a2a.core.mcp_loader import MCPToolLoader
from youngs75_a2a.core.tool_call_utils import tc_args, tc_id, tc_name

from .config import PlannerConfig
from .prompts import ANALYZE_SYSTEM_PROMPT, EXPLORE_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from .schemas import PlannerState, TaskPlan

logger = logging.getLogger(__name__)


async def _execute_tool_safely(tool: Any, args: dict) -> str:
    """도구를 안전하게 실행하고 결과를 문자열로 반환한다."""
    try:
        result = await tool.ainvoke(args)
        return str(result)[:3000] if result else ""
    except Exception as e:
        return f"도구 실행 오류: {e}"


class PlannerAgent(BaseGraphAgent):
    """태스크 분석 및 구현 계획 전문 에이전트.

    analyze_task → [explore?] → explore_context → create_plan → END
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "ANALYZE": "analyze_task",
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
        self._tools: list[Any] = []
        self._explicit_model = model

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
        all_tools = await self._mcp_loader.load()
        allowed = set(self._planner_config.allowed_tools)
        self._tools = [
            t for t in all_tools if getattr(t, "name", None) in allowed
        ]
        logger.info(
            "Planner 도구 %d개 로드 (read-only): %s",
            len(self._tools),
            [getattr(t, "name", "") for t in self._tools],
        )

    def _get_model(self) -> BaseChatModel:
        """STRONG 모델 반환."""
        return self._explicit_model or self._planner_config.get_model("planning")

    # ── 노드 구현 ──

    async def _analyze_task(self, state: PlannerState) -> dict[str, Any]:
        """사용자 요청의 복잡도를 분석한다."""
        user_request = state.get("user_request", "")
        if not user_request:
            # messages에서 추출
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    user_request = msg.content
                    break

        model = self._get_model()
        response = await model.ainvoke([
            SystemMessage(content=ANALYZE_SYSTEM_PROMPT),
            HumanMessage(content=user_request),
        ])

        try:
            analysis = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            analysis = {
                "complexity": "moderate",
                "reason": "분석 실패 — moderate로 기본 처리",
                "estimated_files": 3,
                "needs_exploration": False,
            }

        return {
            "user_request": user_request,
            "task_plan": {"complexity": analysis.get("complexity", "moderate")},
            "explored_context": [],
            "messages": [response],
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

        response = await model.ainvoke([
            SystemMessage(
                content=EXPLORE_SYSTEM_PROMPT.format(
                    tool_descriptions=tool_descriptions
                )
            ),
            HumanMessage(
                content=f"다음 작업을 위해 프로젝트를 탐색하세요:\n{user_request}"
            ),
        ])

        # 도구 호출 실행 (최대 1 라운드)
        context_entries: list[str] = list(state.get("explored_context", []))
        tool_calls = getattr(response, "tool_calls", None) or []
        tools_by_name = {
            getattr(t, "name", None): t for t in self._tools
        }

        messages = [response]

        for call in tool_calls[:5]:  # 최대 5회
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
                    context_entries.append(f"[검색: {args.get('query', '')}]\n{result[:1500]}")
            else:
                result = f"허용되지 않은 도구: {name}"

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=call_id or f"call_{name}",
                    name=name or "unknown",
                )
            )

        return {
            "explored_context": context_entries,
            "messages": messages,
        }

    async def _create_plan(self, state: PlannerState) -> dict[str, Any]:
        """구조화된 실행 계획을 생성한다."""
        user_request = state.get("user_request", "")
        explored_context = state.get("explored_context", [])

        context_str = (
            "\n\n".join(explored_context) if explored_context else "탐색된 컨텍스트 없음"
        )

        model = self._get_model()
        response = await model.ainvoke([
            SystemMessage(
                content=PLAN_SYSTEM_PROMPT.format(explored_context=context_str)
            ),
            HumanMessage(content=user_request),
        ])

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
                deps = f" (의존: {', '.join(phase.get('depends_on', []))})" if phase.get("depends_on") else ""
                parts.append(f"#### {phase.get('id', '?')}: {phase.get('title', '')}{deps}")
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

    def _should_explore(self, state: PlannerState) -> str:
        """탐색이 필요한지 판단한다."""
        task_plan = state.get("task_plan", {})
        complexity = task_plan.get("complexity", "simple")

        # moderate 이상이고 도구가 있으면 탐색
        if complexity in ("moderate", "complex") and self._tools:
            return self.get_node_name("EXPLORE")
        return self.get_node_name("CREATE_PLAN")

    # ── 그래프 구성 ──

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("ANALYZE"), self._analyze_task)
        graph.add_node(self.get_node_name("EXPLORE"), self._explore_context)
        graph.add_node(self.get_node_name("CREATE_PLAN"), self._create_plan)

    def init_edges(self, graph: StateGraph) -> None:
        graph.set_entry_point(self.get_node_name("ANALYZE"))

        # analyze → (explore or create_plan)
        graph.add_conditional_edges(
            self.get_node_name("ANALYZE"),
            self._should_explore,
        )

        # explore → create_plan
        graph.add_edge(
            self.get_node_name("EXPLORE"),
            self.get_node_name("CREATE_PLAN"),
        )

        # create_plan → END
        graph.add_edge(self.get_node_name("CREATE_PLAN"), END)
