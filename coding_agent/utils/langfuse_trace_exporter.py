"""Langfuse 트레이스 대화 내용 추출 유틸리티.

AI Coding Agent Harness가 하나의 요청에 대해 수행한 전체 프로세스를
Langfuse에서 추출하여 읽기 좋은 Markdown 형태로 변환합니다.

사용법:
    # 현재 E2E 세션의 전체 대화 추출 (HARNESS_SESSION_ID 환경변수 사용)
    python -m coding_agent.utils.langfuse_trace_exporter --session current

    # 세션 ID로 전체 대화 추출
    python -m coding_agent.utils.langfuse_trace_exporter --session harness-20260410-091847-323b038f

    # 특정 trace ID로 추출
    python -m coding_agent.utils.langfuse_trace_exporter --trace 65d71a9193e5...

    # 최근 N개 세션 목록 조회
    python -m coding_agent.utils.langfuse_trace_exporter --list-sessions 10

    # harness 세션만 필터링
    python -m coding_agent.utils.langfuse_trace_exporter --list-sessions 10 --harness-only

    # 파일로 출력
    python -m coding_agent.utils.langfuse_trace_exporter --session current -o output.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse

from coding_agent.eval_pipeline.settings import Settings
from coding_agent.utils.e2e_session import get_or_create_session_id

# 프로젝트 루트의 .env를 명시적으로 로드
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


# ── 데이터 모델 ──────────────────────────────────────────────


@dataclass
class Message:
    """LLM 대화의 개별 메시지."""

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class Generation:
    """하나의 LLM 호출 (GENERATION observation)."""

    observation_id: str
    model: str | None
    parent_name: str | None
    start_time: datetime | None
    input_messages: list[Message]
    output_message: Message | None
    usage: dict[str, Any] | None
    latency: float | None


@dataclass
class AgentStep:
    """에이전트 파이프라인의 한 단계 (CHAIN/AGENT observation)."""

    name: str
    observation_type: str
    parent_name: str | None
    children: list[str] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)


@dataclass
class TraceConversation:
    """하나의 trace에서 추출한 전체 대화."""

    trace_id: str
    trace_name: str | None
    session_id: str | None
    timestamp: datetime | None
    user_input: str | None
    agent_output: str | None
    total_cost: float | None
    steps: list[AgentStep] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)


# ── Langfuse 클라이언트 ─────────────────────────────────────


def _create_client() -> Langfuse:
    """Settings 기반으로 Langfuse 클라이언트를 생성합니다."""
    s = Settings()
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        host=s.langfuse_base_url or s.langfuse_host,
    )


def resolve_session_id(raw: str) -> str:
    """세션 ID를 해석한다.

    - ``"current"`` → ``HARNESS_SESSION_ID`` 환경변수에서 가져옴
    - 그 외 → 그대로 반환
    """
    if raw.lower() == "current":
        sid = os.environ.get("HARNESS_SESSION_ID")
        if not sid:
            print(
                "ERROR: HARNESS_SESSION_ID 환경변수가 설정되지 않았습니다.\n"
                "E2E 테스트 실행 후 사용하거나, 직접 세션 ID를 지정하세요.",
                file=sys.stderr,
            )
            sys.exit(1)
        return sid
    return raw


# ── 파싱 헬퍼 ───────────────────────────────────────────────


def _parse_message(msg: dict[str, Any]) -> Message:
    """LLM 메시지 dict를 Message로 변환합니다."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    if isinstance(content, list):
        # multimodal content blocks
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", json.dumps(block, ensure_ascii=False)))
            else:
                parts.append(str(block))
        content = "\n".join(parts)
    elif content is None:
        content = ""

    tool_calls = []
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", ""),
            }
        )

    return Message(
        role=role,
        content=str(content),
        tool_calls=tool_calls,
        tool_call_id=msg.get("tool_call_id"),
    )


def _extract_messages_from_input(inp: Any) -> list[Message]:
    """observation.input에서 메시지 목록을 추출합니다."""
    if inp is None:
        return []
    if isinstance(inp, dict) and "messages" in inp:
        return [_parse_message(m) for m in inp["messages"]]
    if isinstance(inp, list):
        return [_parse_message(m) for m in inp if isinstance(m, dict)]
    return []


def _extract_output_message(out: Any) -> Message | None:
    """observation.output에서 응답 메시지를 추출합니다."""
    if out is None:
        return None
    if isinstance(out, dict):
        return _parse_message(out)
    if isinstance(out, str):
        return Message(role="assistant", content=out)
    return None


def _extract_user_request(trace_input: Any, trace_output: Any) -> str | None:
    """trace 레벨에서 사용자 원본 요청을 추출합니다."""
    # trace output.messages 의 첫 human 메시지가 원본 요청인 경우가 많음
    if isinstance(trace_output, dict) and "messages" in trace_output:
        for msg in trace_output["messages"]:
            content = msg.get("content", "")
            role = ""
            # LangGraph state 메시지 형식
            if "type" in msg:
                role = msg["type"]
            elif "role" in msg:
                role = msg["role"]
            if role in ("human", "user") and content:
                return str(content)
    # trace input 자체가 문자열인 경우
    if isinstance(trace_input, str) and trace_input:
        return trace_input
    return None


def _extract_final_output(trace_output: Any) -> str | None:
    """trace 레벨에서 최종 에이전트 응답을 추출합니다."""
    if isinstance(trace_output, dict) and "messages" in trace_output:
        # 마지막 AI 메시지를 찾음
        for msg in reversed(trace_output["messages"]):
            role = msg.get("type", msg.get("role", ""))
            content = msg.get("content", "")
            if role in ("ai", "assistant") and content:
                return str(content)
    if isinstance(trace_output, str) and trace_output:
        return trace_output
    return None


# ── 핵심 추출 로직 ──────────────────────────────────────────


def extract_trace(lf: Langfuse, trace_id: str) -> TraceConversation:
    """하나의 trace에서 전체 대화를 추출합니다."""
    detail = lf.api.trace.get(trace_id)

    # trace 목록 API에서 기본 정보 (detail에도 있음)
    user_input = _extract_user_request(detail.input, detail.output)
    agent_output = _extract_final_output(detail.output)

    conversation = TraceConversation(
        trace_id=trace_id,
        trace_name=detail.name,
        session_id=detail.session_id,
        timestamp=detail.timestamp,
        user_input=user_input,
        agent_output=agent_output,
        total_cost=detail.total_cost,
    )

    # observation을 시간순 정렬
    observations = sorted(
        detail.observations,
        key=lambda o: o.start_time or o.end_time or detail.timestamp,
    )

    # observation ID → name 매핑 (부모 이름 조회용)
    id_to_name: dict[str, str] = {}
    for obs in observations:
        id_to_name[obs.id] = obs.name or obs.type or "unknown"

    for obs in observations:
        parent_name = id_to_name.get(obs.parent_observation_id or "", None)

        if obs.type == "GENERATION":
            gen = Generation(
                observation_id=obs.id,
                model=obs.model,
                parent_name=parent_name or obs.name,
                start_time=obs.start_time,
                input_messages=_extract_messages_from_input(obs.input),
                output_message=_extract_output_message(obs.output),
                usage=obs.usage_details or (
                    {"input": obs.usage.input, "output": obs.usage.output}
                    if obs.usage
                    else None
                ),
                latency=obs.latency,
            )
            conversation.generations.append(gen)

    return conversation


def extract_session(
    lf: Langfuse,
    session_id: str,
) -> list[TraceConversation]:
    """세션의 모든 trace에서 대화를 추출합니다."""
    # 세션의 모든 trace 가져오기
    traces = lf.api.trace.list(session_id=session_id, limit=100)
    conversations = []
    for t in sorted(traces.data, key=lambda x: x.timestamp or datetime.min):
        conv = extract_trace(lf, t.id)
        conversations.append(conv)
    return conversations


def list_sessions(
    lf: Langfuse,
    limit: int = 20,
    harness_only: bool = False,
) -> list[dict[str, Any]]:
    """최근 세션 목록을 조회합니다.

    Args:
        harness_only: True이면 ``harness-`` 프리픽스 세션만 반환
    """
    # harness_only 필터링을 위해 넉넉하게 가져옴
    fetch_limit = limit * 3 if harness_only else limit
    sessions = lf.api.sessions.list(limit=fetch_limit)
    result = []
    for s in sessions.data:
        if harness_only and not s.id.startswith("harness-"):
            continue
        # 세션별 trace 수 확인
        traces = lf.api.trace.list(session_id=s.id, limit=100)
        trace_names = [t.name for t in traces.data]
        result.append(
            {
                "session_id": s.id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "trace_count": len(traces.data),
                "trace_names": trace_names,
            }
        )
        if len(result) >= limit:
            break
    return result


def list_traces(
    lf: Langfuse,
    limit: int = 20,
    name: str | None = None,
) -> list[dict[str, Any]]:
    """최근 trace 목록을 조회합니다."""
    kwargs: dict[str, Any] = {"limit": limit}
    if name:
        kwargs["name"] = name
    traces = lf.api.trace.list(**kwargs)
    result = []
    for t in traces.data:
        user_input = _extract_user_request(t.input, t.output)
        result.append(
            {
                "trace_id": t.id,
                "name": t.name,
                "session_id": t.session_id,
                "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                "user_input": (user_input[:100] + "...") if user_input and len(user_input) > 100 else user_input,
                "total_cost": t.total_cost,
                "tags": t.tags,
            }
        )
    return result


# ── Markdown 포맷터 ─────────────────────────────────────────


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """도구 호출을 읽기 좋은 형태로 포맷합니다."""
    lines = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        args_raw = tc.get("arguments", "")
        # JSON arguments를 파싱해서 보기 좋게
        try:
            if isinstance(args_raw, str):
                args = json.loads(args_raw)
            else:
                args = args_raw
            args_str = json.dumps(args, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            args_str = str(args_raw)

        # 너무 긴 arguments는 축약
        if len(args_str) > 500:
            args_str = args_str[:500] + "\n  ... (truncated)"

        lines.append(f"**`{name}`**")
        lines.append(f"```json\n{args_str}\n```")
    return "\n".join(lines)


def _format_content(content: str, max_length: int = 2000) -> str:
    """메시지 content를 적절한 길이로 포맷합니다."""
    if not content:
        return "(empty)"
    if len(content) > max_length:
        return content[:max_length] + f"\n\n... (truncated, total {len(content)} chars)"
    return content


def format_conversation_markdown(
    conversations: list[TraceConversation],
    *,
    verbose: bool = False,
) -> str:
    """추출된 대화를 Claude Code가 읽기 좋은 Markdown으로 포맷합니다."""
    lines: list[str] = []

    # 헤더
    if conversations:
        first = conversations[0]
        lines.append(f"# Langfuse Trace Export")
        if first.session_id:
            lines.append(f"**Session**: `{first.session_id}`")
        lines.append(f"**Traces**: {len(conversations)}개")
        ts = first.timestamp
        if ts:
            lines.append(f"**시작 시각**: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        total_cost = sum(c.total_cost or 0 for c in conversations)
        if total_cost > 0:
            lines.append(f"**총 비용**: ${total_cost:.6f}")
        lines.append("")

    for conv_idx, conv in enumerate(conversations, 1):
        lines.append(f"---")
        lines.append(f"## Trace {conv_idx}: {conv.trace_name or 'unnamed'}")
        lines.append(f"- **ID**: `{conv.trace_id}`")
        if conv.timestamp:
            lines.append(f"- **시각**: {conv.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if conv.total_cost:
            lines.append(f"- **비용**: ${conv.total_cost:.6f}")
        lines.append(f"- **LLM 호출 수**: {len(conv.generations)}회")
        lines.append("")

        # 사용자 요청
        if conv.user_input:
            lines.append(f"### 사용자 요청")
            lines.append(f"```")
            lines.append(_format_content(conv.user_input, max_length=3000))
            lines.append(f"```")
            lines.append("")

        # 각 Generation (LLM 호출)을 시간순으로
        for gen_idx, gen in enumerate(conv.generations, 1):
            model_tag = f"`{gen.model}`" if gen.model else "unknown"
            parent_tag = f" ({gen.parent_name})" if gen.parent_name else ""
            lines.append(f"### Step {gen_idx}: LLM Call — {model_tag}{parent_tag}")

            # 사용량
            if gen.usage:
                input_tokens = gen.usage.get("input", 0)
                output_tokens = gen.usage.get("output", 0)
                if input_tokens or output_tokens:
                    lines.append(f"tokens: in={input_tokens} out={output_tokens}")
            if gen.latency:
                lines.append(f"latency: {gen.latency:.1f}s")
            lines.append("")

            if verbose:
                # 전체 input 메시지 표시
                for msg in gen.input_messages:
                    role_label = msg.role.upper()
                    lines.append(f"**[{role_label}]**")
                    if msg.content:
                        lines.append(_format_content(msg.content))
                    if msg.tool_calls:
                        lines.append(_format_tool_calls(msg.tool_calls))
                    if msg.tool_call_id:
                        lines.append(f"*(tool_call_id: {msg.tool_call_id})*")
                    lines.append("")
            else:
                # 간결 모드: system/user만 표시, tool 메시지는 요약
                shown_system = False
                for msg in gen.input_messages:
                    if msg.role == "system" and not shown_system:
                        # system prompt는 첫 번째만 축약 표시
                        lines.append(f"**[SYSTEM]** *(length: {len(msg.content)} chars)*")
                        lines.append(f"> {msg.content[:200]}...")
                        lines.append("")
                        shown_system = True
                    elif msg.role in ("user", "human"):
                        lines.append(f"**[USER]**")
                        lines.append(_format_content(msg.content))
                        lines.append("")
                    elif msg.role in ("assistant", "ai") and msg.tool_calls:
                        tool_names = [tc["name"] for tc in msg.tool_calls]
                        lines.append(f"**[ASSISTANT]** → tool calls: {', '.join(tool_names)}")
                        if verbose:
                            lines.append(_format_tool_calls(msg.tool_calls))
                        lines.append("")
                    elif msg.role == "tool":
                        if verbose or len(msg.content) < 300:
                            lines.append(f"**[TOOL]** *(id: {msg.tool_call_id})*")
                            lines.append(f"```\n{_format_content(msg.content, 500)}\n```")
                        else:
                            lines.append(f"**[TOOL]** *(id: {msg.tool_call_id}, {len(msg.content)} chars)*")
                        lines.append("")

            # Output
            if gen.output_message:
                out = gen.output_message
                lines.append(f"**[OUTPUT]**")
                if out.content:
                    lines.append(_format_content(out.content))
                if out.tool_calls:
                    lines.append("")
                    lines.append(f"**Tool Calls:**")
                    lines.append(_format_tool_calls(out.tool_calls))
                lines.append("")

        # 최종 응답
        if conv.agent_output:
            lines.append(f"### 최종 에이전트 응답")
            lines.append(_format_content(conv.agent_output, max_length=5000))
            lines.append("")

    return "\n".join(lines)


def format_sessions_list(sessions: list[dict[str, Any]]) -> str:
    """세션 목록을 Markdown 테이블로 포맷합니다."""
    lines = ["# Langfuse Sessions", ""]
    lines.append("| # | Session ID | Created | Traces | Agent Names |")
    lines.append("|---|-----------|---------|--------|-------------|")
    for i, s in enumerate(sessions, 1):
        names = ", ".join(set(s["trace_names"]))
        lines.append(
            f"| {i} | `{s['session_id']}` | {s['created_at'] or '-'} "
            f"| {s['trace_count']} | {names} |"
        )
    return "\n".join(lines)


def format_traces_list(traces: list[dict[str, Any]]) -> str:
    """trace 목록을 Markdown 테이블로 포맷합니다."""
    lines = ["# Langfuse Traces", ""]
    lines.append("| # | Trace ID | Name | Session | Time | Cost | User Input |")
    lines.append("|---|----------|------|---------|------|------|------------|")
    for i, t in enumerate(traces, 1):
        tid = t["trace_id"][:12] + "..."
        cost = f"${t['total_cost']:.4f}" if t["total_cost"] else "-"
        user_input = (t["user_input"] or "-")[:60]
        lines.append(
            f"| {i} | `{tid}` | {t['name']} | `{t['session_id'] or '-'}` "
            f"| {t['timestamp'] or '-'} | {cost} | {user_input} |"
        )
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Langfuse 트레이스에서 AI Coding Agent 대화 내용을 추출합니다.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session", "-s",
        help="세션 ID로 전체 대화 추출 ('current'로 HARNESS_SESSION_ID 환경변수 사용)",
    )
    group.add_argument("--trace", "-t", help="특정 trace ID로 대화 추출")
    group.add_argument(
        "--list-sessions", "-ls", type=int, nargs="?", const=20, metavar="N",
        help="최근 N개 세션 목록 조회 (기본: 20)",
    )
    group.add_argument(
        "--list-traces", "-lt", type=int, nargs="?", const=20, metavar="N",
        help="최근 N개 trace 목록 조회 (기본: 20)",
    )
    parser.add_argument("--output", "-o", help="출력 파일 경로 (미지정 시 stdout)")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="전체 메시지 내용 표시 (tool 메시지 포함)",
    )
    parser.add_argument("--name", help="trace 이름 필터 (--list-traces와 함께 사용)")
    parser.add_argument(
        "--harness-only", action="store_true",
        help="harness- 프리픽스 세션만 표시 (--list-sessions와 함께 사용)",
    )

    args = parser.parse_args()
    lf = _create_client()

    if args.list_sessions is not None:
        sessions = list_sessions(
            lf, limit=args.list_sessions, harness_only=args.harness_only,
        )
        output = format_sessions_list(sessions)
    elif args.list_traces is not None:
        traces = list_traces(lf, limit=args.list_traces, name=args.name)
        output = format_traces_list(traces)
    elif args.session:
        session_id = resolve_session_id(args.session)
        print(f"세션 ID: {session_id}", file=sys.stderr)
        conversations = extract_session(lf, session_id)
        output = format_conversation_markdown(conversations, verbose=args.verbose)
    elif args.trace:
        conversation = extract_trace(lf, args.trace)
        output = format_conversation_markdown([conversation], verbose=args.verbose)
    else:
        parser.print_help()
        return

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"출력 완료: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
