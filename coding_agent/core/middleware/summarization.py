"""LLM 기반 요약 컨텍스트 컴팩션 미들웨어.

DeepAgents/Claude Code 패턴:
- 토큰 임계치 초과 시 FAST 모델로 이전 대화를 요약
- 요약본은 원본 메시지를 대체하여 컨텍스트 절약
- 핵심 정보(요청, 생성된 파일, 에러, 결정사항)를 보존
- LLM 실패 시 규칙 기반 제거로 폴백
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from coding_agent.core.middleware.base import (
    AgentMiddleware,
    Handler,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

# 요약 시스템 프롬프트 (DeepAgents 패턴: 요약에 최대 4K 토큰)
_SUMMARIZE_SYSTEM_PROMPT = """\
당신은 AI 코딩 에이전트의 대화 기록을 압축하는 전문가입니다.

## 역할
대화에서 **중복과 반복을 제거**하되, 핵심 정보는 **모두 보존**하세요.
코드를 생성하는 에이전트가 이 요약만 보고도 이전에 무슨 일이 있었는지 완전히 파악할 수 있어야 합니다.

## 반드시 보존할 정보 (하나도 빠짐없이)
1. **사용자의 원래 요청** 전문
2. **생성된 파일 목록**과 각 파일의 핵심 구조:
   - 변수명, 함수명, 클래스명 (예: `bp = Blueprint('routes', ...)`, `api_bp` 등)
   - import 경로 (예: `from backend.routes import bp`)
   - 팩토리 함수 시그니처 (예: `create_app(config_name='development')`)
3. **에러 내용** 전문 (traceback, 에러 메시지)
4. **아키텍처 결정사항** (DB URI, Blueprint URL prefix, CORS 설정 등)
5. **파일 간 의존 관계** (어떤 파일이 어떤 모듈의 어떤 이름을 import하는지)

## 제거해도 되는 것
- 동일한 도구 호출의 반복 (예: write_file 호출이 7번이면 파일 목록으로 압축)
- 도구 결과 중 "OK: filename (N자, M줄)" 형태의 성공 확인 → 파일명만 보존
- LLM이 생성한 코드 전문 → 핵심 시그니처와 import 구문만 보존
- 시스템 프롬프트 반복

## 형식
마크다운으로 구조화하여 작성하세요. 한국어로 작성하세요.
"""


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    """메시지 리스트의 토큰 수를 대략 추정한다 (4 chars ≈ 1 token)."""
    total_chars = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total_chars += len(content)
        # 도구 호출 인자도 포함
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                total_chars += len(str(tc.get("args", {})))
    return total_chars // 4 + 1


def _messages_to_text(messages: list[BaseMessage], max_chars: int = 60_000) -> str:
    """메시지 리스트를 요약용 텍스트로 변환한다."""
    lines: list[str] = []
    total = 0
    for msg in messages:
        role = type(msg).__name__.replace("Message", "").upper()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        # 도구 호출 요약
        tool_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_names = [tc.get("name", "?") for tc in msg.tool_calls]
            tool_info = f" [도구 호출: {', '.join(tool_names)}]"

        # 긴 내용 축약
        if len(content) > 1000:
            content = content[:500] + f"... ({len(content)}자 생략) ..." + content[-300:]

        line = f"[{role}]{tool_info} {content}"
        if total + len(line) > max_chars:
            lines.append(f"... (이하 {len(messages) - len(lines)}개 메시지 생략)")
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


class SummarizationMiddleware(AgentMiddleware):
    """LLM 기반 요약 미들웨어.

    Phase 1: AIMessage의 tool_calls 인자가 max_tool_arg_chars를 초과하면 잘라냄.
    Phase 2: 토큰 임계치 초과 시 FAST 모델로 이전 대화를 요약하여 교체.

    Args:
        token_threshold: 요약 트리거 토큰 수
        keep_recent_messages: 요약 대상에서 제외할 최근 메시지 수
        max_tool_arg_chars: 도구 인자 최대 길이
        summarize_model: 요약에 사용할 LLM (None이면 규칙 기반 폴백)
    """

    def __init__(
        self,
        token_threshold: int = 100_000,
        keep_recent_messages: int = 6,
        max_tool_arg_chars: int = 2000,
        summarize_model: Any = None,
    ) -> None:
        self._token_threshold = token_threshold
        self._keep_recent = keep_recent_messages
        self._max_tool_arg_chars = max_tool_arg_chars
        self._model = summarize_model

    async def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Handler,
    ) -> ModelResponse:
        messages = list(request.messages)
        modified = False

        # Phase 1: tool_calls 인자 축소
        for i, msg in enumerate(messages):
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                args = tc.get("args", {})
                if not isinstance(args, dict):
                    continue
                for key, value in args.items():
                    if isinstance(value, str) and len(value) > self._max_tool_arg_chars:
                        if not modified:
                            messages = copy.deepcopy(messages)
                            modified = True
                        truncated = (
                            value[:200]
                            + f"... ({len(value)}자 생략)"
                        )
                        messages[i].tool_calls[
                            msg.tool_calls.index(tc)
                        ]["args"][key] = truncated
                        logger.debug(
                            "[Summarization] Phase1: tool_call arg '%s' 축소 (%d → %d chars)",
                            key, len(value), len(truncated),
                        )

        # Phase 2: 토큰 임계치 초과 시 LLM 요약 또는 규칙 기반 제거
        estimated = _estimate_tokens(messages)
        if estimated > self._token_threshold and len(messages) > self._keep_recent + 1:
            first_msg = messages[0]
            old_msgs = messages[1:-self._keep_recent]
            tail_msgs = messages[-self._keep_recent:]
            removed_count = len(old_msgs)

            summary_text = await self._summarize(old_msgs)

            summary = HumanMessage(content=(
                f"## 이전 대화 요약 ({removed_count}개 메시지 압축)\n\n"
                f"{summary_text}"
            ))

            messages = [first_msg, summary] + tail_msgs
            modified = True
            logger.info(
                "[Summarization] Phase2: 토큰 %d → 압축 (%d개 메시지 → 요약), 모델=%s",
                estimated, removed_count,
                "LLM" if self._model else "규칙기반",
            )

        if modified:
            return await handler(request.override(messages=messages))
        return await handler(request)

    async def _summarize(self, messages: list[BaseMessage]) -> str:
        """메시지 리스트를 요약한다. LLM 사용 가능 시 LLM, 아니면 규칙 기반."""
        if self._model:
            return await self._llm_summarize(messages)
        return self._rule_based_summarize(messages)

    async def _llm_summarize(self, messages: list[BaseMessage]) -> str:
        """FAST 모델로 대화 내용을 요약한다."""
        conversation_text = _messages_to_text(messages)

        try:
            response = await self._model.ainvoke([
                SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"다음 대화 기록에서 중복과 반복을 제거하고 핵심 정보를 보존하세요.\n\n"
                    f"{conversation_text}"
                )),
            ])
            summary = response.content if isinstance(response.content, str) else str(response.content)
            logger.info("[Summarization] LLM 요약 완료 (%d자)", len(summary))
            return summary
        except Exception as e:
            logger.warning("[Summarization] LLM 요약 실패, 규칙 기반 폴백: %s", e)
            return self._rule_based_summarize(messages)

    def _rule_based_summarize(self, messages: list[BaseMessage]) -> str:
        """LLM 없이 규칙 기반으로 핵심 정보를 추출한다 (폴백)."""
        lines: list[str] = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else ""
            if isinstance(msg, HumanMessage) and len(content) > 50:
                lines.append(f"- 사용자: {content[:200]}")
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_names = [tc.get("name", "?") for tc in msg.tool_calls]
                lines.append(f"- 도구 호출: {', '.join(tool_names)}")

        if not lines:
            return f"[{len(messages)}개 메시지 제거됨]"
        return "\n".join(lines[:10])
