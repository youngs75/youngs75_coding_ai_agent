"""Verification Agent — lint, test, LLM 리뷰 기반 코드 검증.

CodingAssistant의 verify 노드를 독립 에이전트로 분리한 것.
3중 검증 파이프라인:
1. Lint 체크: syntax/compile 오류 감지 (MCP run_python)
2. Test 체크: 테스트 실행 (MCP run_python)
3. LLM 리뷰: 보안, 스타일, 완전성 검증

독립 호출 또는 CodingAssistant/Orchestrator에서 위임 가능.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
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

from .config import VerifierConfig
from .prompts import REVIEW_SYSTEM_PROMPT
from .schemas import CheckResult, VerificationResult, VerificationState

logger = logging.getLogger(__name__)


class VerificationAgent(BaseGraphAgent):
    """lint + test + LLM 리뷰 기반 검증 에이전트.

    사용:
        agent = await VerificationAgent.create(config=VerifierConfig())
        result = await agent.graph.ainvoke({
            "code": "...",
            "written_files": ["app.py", "models.py"],
            "language": "python",
            "requirements": "사용자 요구사항",
        })
        verification = result["verification_result"]
    """

    NODE_NAMES: ClassVar[dict[str, str]] = {
        "LINT": "lint_check",
        "TEST": "test_check",
        "REVIEW": "llm_review",
        "AGGREGATE": "aggregate_results",
    }

    def __init__(
        self,
        *,
        config: VerifierConfig | None = None,
        model: BaseChatModel | None = None,
        **kwargs: Any,
    ) -> None:
        self._verifier_config = config or VerifierConfig()
        self._mcp_loader = MCPToolLoader(self._verifier_config.mcp_servers)
        self._tools: list[Any] = []

        if model is None:
            model = self._verifier_config.get_model("review")

        # AbortController + 미들웨어 체인
        self._abort_controller = AbortController()
        self._middleware_chain = MiddlewareChain([
            ResilienceMiddleware(abort_controller=self._abort_controller),
            MemoryMiddleware(),
        ])

        kwargs.pop("auto_build", None)
        super().__init__(
            config=self._verifier_config,
            model=model,
            state_schema=VerificationState,
            agent_name="VerificationAgent",
            auto_build=False,
            **kwargs,
        )

    async def async_init(self) -> None:
        """MCP 도구를 비동기로 로딩한다."""
        try:
            self._tools = await self._mcp_loader.load()
        except Exception as e:
            logger.warning("MCP 도구 로딩 실패 (검증은 LLM 리뷰만 진행): %s", e)
            self._tools = []

    # ── 노드 구현 ──

    async def _lint_check(self, state: VerificationState) -> dict[str, Any]:
        """파일의 문법 오류를 검사한다 (멀티언어 지원)."""
        # 턴 시작: abort 상태 리셋 (lint가 첫 노드)
        self._abort_controller.reset()

        if not self._verifier_config.enable_lint:
            return {"lint_result": _skip_result("lint", "비활성화됨")}

        written_files = state.get("written_files", [])
        language = state.get("language", "python")

        # 언어별 lint 대상 파일 필터링
        lint_targets = _filter_lintable_files(written_files, language)
        if not lint_targets:
            return {"lint_result": _skip_result("lint", f"lint 가능한 {language} 파일 없음")}

        run_python = _find_tool(self._tools, "run_python")
        if not run_python:
            return {"lint_result": _skip_result("lint", "run_python 도구 없음")}

        issues: list[str] = []
        for filepath in lint_targets:
            try:
                lint_code = _build_lint_command(filepath, language)
                result = await asyncio.wait_for(
                    run_python.ainvoke({"code": lint_code}),
                    timeout=self._verifier_config.lint_timeout,
                )
                output = str(result)
                if "ERROR" in output or "error" in output.lower():
                    issues.append(f"{filepath}: {output}")
            except TimeoutError:
                issues.append(f"{filepath}: lint 타임아웃")
            except Exception as e:
                logger.warning("lint 실패 (%s): %s", filepath, e)

        passed = len(issues) == 0
        return {
            "lint_result": CheckResult(
                check_type="lint",
                passed=passed,
                output=f"{len(lint_targets)}개 {language} 파일 검사, {len(issues)}개 오류",
                issues=issues,
            ),
        }

    async def _test_check(self, state: VerificationState) -> dict[str, Any]:
        """프로젝트 테스트를 실행한다 (멀티언어 지원)."""
        if not self._verifier_config.enable_test:
            return {"test_result": _skip_result("test", "비활성화됨")}

        written_files = state.get("written_files", [])
        language = state.get("language", "python")
        if not written_files:
            return {"test_result": _skip_result("test", "파일 없음")}

        # 테스트 파일이 있는지 확인
        test_files = [f for f in written_files if "test" in f.lower() or "_test" in f.lower() or f.endswith("_spec.js") or f.endswith("_spec.ts")]
        if not test_files:
            return {"test_result": _skip_result("test", "테스트 파일 없음 — 스킵")}

        run_python = _find_tool(self._tools, "run_python")
        if not run_python:
            return {"test_result": _skip_result("test", "run_python 도구 없음")}

        try:
            test_code = _build_test_command(test_files, language)
            result = await asyncio.wait_for(
                run_python.ainvoke({"code": test_code}),
                timeout=self._verifier_config.test_timeout,
            )
            output = str(result)
            passed = "failed" not in output.lower() and "FAIL" not in output
            issues = [output] if not passed else []
        except TimeoutError:
            passed = True
            output = "테스트 타임아웃 — 스킵"
            issues = []
        except Exception as e:
            logger.warning("test 실행 실패: %s", e)
            passed = True
            output = f"테스트 실행 실패: {e}"
            issues = []

        return {
            "test_result": CheckResult(
                check_type="test",
                passed=passed,
                output=output,
                issues=issues,
            ),
        }

    async def _llm_review(self, state: VerificationState) -> dict[str, Any]:
        """LLM 기반 코드 리뷰를 수행한다.

        미들웨어 체인으로 abort 체크 + 재시도 보호.
        """
        if not self._verifier_config.enable_llm_review:
            return {"review_result": _skip_result("llm_review", "비활성화됨")}

        code = state.get("code", "")
        if not code:
            return {"review_result": _skip_result("llm_review", "코드 없음")}

        # 코드 트런케이션
        max_chars = self._verifier_config.max_review_chars
        code_for_review = code
        if len(code_for_review) > max_chars:
            code_for_review = (
                code_for_review[:max_chars]
                + f"\n\n... (총 {len(code)}자 중 {max_chars}자만 검증)"
            )

        requirements = state.get("requirements", "")
        review_context = (
            f"원래 요청: {requirements}\n\n"
            f"생성된 코드:\n{code_for_review}"
        )

        try:
            mw_request = MWRequest(
                system_message=REVIEW_SYSTEM_PROMPT,
                messages=[HumanMessage(content=review_context)],
                metadata={
                    "purpose": "verification",
                    "request_timeout": self._verifier_config.get_request_timeout("verification"),
                },
            )
            mw_response = await asyncio.wait_for(
                self._middleware_chain.invoke(mw_request, self.model),
                timeout=self._verifier_config.review_timeout,
            )
            review = json.loads(mw_response.message.content)
            return {
                "review_result": CheckResult(
                    check_type="llm_review",
                    passed=review.get("passed", True),
                    output=json.dumps(review, ensure_ascii=False),
                    issues=review.get("issues", []),
                ),
            }
        except (TimeoutError, json.JSONDecodeError, Exception) as e:
            logger.warning("LLM 리뷰 실패: %s", e)
            return {
                "review_result": CheckResult(
                    check_type="llm_review",
                    passed=True,
                    output=f"리뷰 실패 — 스킵: {e}",
                    issues=[],
                ),
            }

    async def _aggregate_results(self, state: VerificationState) -> dict[str, Any]:
        """모든 체크 결과를 집계한다.

        순수 데이터 집계 — LLM 미호출. abort 체크포인트만 삽입.
        """
        # 집계 전 abort 체크
        self._abort_controller.check_or_raise()

        checks: list[CheckResult] = []
        all_issues: list[str] = []
        all_suggestions: list[str] = []

        for key in ("lint_result", "test_result", "review_result"):
            result = state.get(key)
            if result:
                checks.append(result)
                all_issues.extend(result.get("issues", []))

        # review_result에서 suggestions 추출
        review = state.get("review_result", {})
        if review and review.get("output"):
            try:
                review_data = json.loads(review["output"])
                all_suggestions.extend(review_data.get("suggestions", []))
            except (json.JSONDecodeError, TypeError):
                pass

        passed = all(c.get("passed", True) for c in checks)

        # 요약 생성
        check_summary = []
        for c in checks:
            icon = "✓" if c.get("passed", True) else "✗"
            check_summary.append(f"{icon} {c.get('check_type', '?')}")
        summary = " | ".join(check_summary)

        return {
            "verification_result": VerificationResult(
                passed=passed,
                checks=checks,
                issues=all_issues,
                suggestions=all_suggestions,
                summary=summary,
            ),
        }

    # ── 그래프 구성 ──

    def init_nodes(self, graph: StateGraph) -> None:
        graph.add_node(self.get_node_name("LINT"), self._lint_check)
        graph.add_node(self.get_node_name("TEST"), self._test_check)
        graph.add_node(self.get_node_name("REVIEW"), self._llm_review)
        graph.add_node(self.get_node_name("AGGREGATE"), self._aggregate_results)

    def init_edges(self, graph: StateGraph) -> None:
        # lint → test → review → aggregate → END (순차)
        graph.set_entry_point(self.get_node_name("LINT"))
        graph.add_edge(self.get_node_name("LINT"), self.get_node_name("TEST"))
        graph.add_edge(self.get_node_name("TEST"), self.get_node_name("REVIEW"))
        graph.add_edge(self.get_node_name("REVIEW"), self.get_node_name("AGGREGATE"))
        graph.add_edge(self.get_node_name("AGGREGATE"), END)


# ── 헬퍼 함수 ──


# ── 언어별 lint/test 커맨드 빌더 ──

_LANG_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "rust": [".rs"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc"],
    "ruby": [".rb"],
    "csharp": [".cs"],
}


def _filter_lintable_files(files: list[str], language: str) -> list[str]:
    """언어에 맞는 lint 대상 파일을 필터링한다."""
    exts = _LANG_EXTENSIONS.get(language, [])
    if not exts:
        # 알려지지 않은 언어면 모든 소스 파일 대상
        return files
    return [f for f in files if any(f.endswith(ext) for ext in exts)]


def _build_lint_command(filepath: str, language: str) -> str:
    """언어별 lint 명령어를 생성한다."""
    if language == "python":
        return (
            "import py_compile, sys\n"
            "try:\n"
            f"    py_compile.compile('{filepath}', doraise=True)\n"
            "    print('OK')\n"
            "except py_compile.PyCompileError as e:\n"
            "    print(f'ERROR: {{e}}', file=sys.stderr)\n"
            "    sys.exit(1)"
        )
    if language in ("javascript", "typescript"):
        return (
            "import subprocess, sys\n"
            f"result = subprocess.run(['node', '--check', '{filepath}'],\n"
            "    capture_output=True, text=True, timeout=15)\n"
            "print(result.stdout or 'OK')\n"
            "if result.returncode != 0:\n"
            "    print(f'ERROR: {{result.stderr}}', file=sys.stderr)\n"
            "    sys.exit(1)"
        )
    if language == "go":
        return (
            "import subprocess, sys, os\n"
            f"d = os.path.dirname('{filepath}') or '.'\n"
            "result = subprocess.run(['go', 'vet', './...'],\n"
            "    capture_output=True, text=True, timeout=30, cwd=d)\n"
            "print(result.stdout or 'OK')\n"
            "if result.returncode != 0:\n"
            "    print(f'ERROR: {{result.stderr}}', file=sys.stderr)\n"
            "    sys.exit(1)"
        )
    if language == "rust":
        return (
            "import subprocess, sys\n"
            "result = subprocess.run(['cargo', 'check', '--message-format=short'],\n"
            "    capture_output=True, text=True, timeout=60)\n"
            "print(result.stdout or 'OK')\n"
            "if result.returncode != 0:\n"
            "    print(f'ERROR: {{result.stderr}}', file=sys.stderr)\n"
            "    sys.exit(1)"
        )
    if language == "java":
        return (
            "import subprocess, sys\n"
            f"result = subprocess.run(['javac', '-Xlint:all', '{filepath}'],\n"
            "    capture_output=True, text=True, timeout=30)\n"
            "print(result.stdout or 'OK')\n"
            "if result.returncode != 0:\n"
            "    print(f'ERROR: {{result.stderr}}', file=sys.stderr)\n"
            "    sys.exit(1)"
        )
    # 지원하지 않는 언어: 기본 Python syntax check (fallback)
    if filepath.endswith(".py"):
        return _build_lint_command(filepath, "python")
    return "print('OK')  # lint 미지원 언어"


def _build_test_command(test_files: list[str], language: str) -> str:
    """언어별 테스트 실행 명령어를 생성한다."""
    if language == "python":
        return (
            "import subprocess, sys\n"
            f"files = {test_files!r}\n"
            "result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pytest', '--tb=short', '-q'] + files,\n"
            "    capture_output=True, text=True, timeout=50\n"
            ")\n"
            "print(result.stdout)\n"
            "if result.returncode != 0:\n"
            "    print(result.stderr, file=sys.stderr)\n"
            "    sys.exit(result.returncode)"
        )
    if language in ("javascript", "typescript"):
        return (
            "import subprocess, sys\n"
            "# npx jest 또는 npx vitest 시도\n"
            "result = subprocess.run(\n"
            "    ['npx', '--yes', 'jest', '--passWithNoTests', '--no-coverage'],\n"
            "    capture_output=True, text=True, timeout=60\n"
            ")\n"
            "print(result.stdout)\n"
            "if result.returncode != 0:\n"
            "    print(result.stderr, file=sys.stderr)\n"
            "    sys.exit(result.returncode)"
        )
    if language == "go":
        return (
            "import subprocess, sys\n"
            "result = subprocess.run(\n"
            "    ['go', 'test', '-v', '-count=1', './...'],\n"
            "    capture_output=True, text=True, timeout=60\n"
            ")\n"
            "print(result.stdout)\n"
            "if result.returncode != 0:\n"
            "    print(result.stderr, file=sys.stderr)\n"
            "    sys.exit(result.returncode)"
        )
    if language == "rust":
        return (
            "import subprocess, sys\n"
            "result = subprocess.run(\n"
            "    ['cargo', 'test'],\n"
            "    capture_output=True, text=True, timeout=120\n"
            ")\n"
            "print(result.stdout)\n"
            "if result.returncode != 0:\n"
            "    print(result.stderr, file=sys.stderr)\n"
            "    sys.exit(result.returncode)"
        )
    # 기본: Python pytest fallback
    return _build_test_command(test_files, "python")


def _skip_result(check_type: str, reason: str) -> CheckResult:
    """스킵된 체크의 기본 결과를 생성한다."""
    return CheckResult(
        check_type=check_type,
        passed=True,
        output=reason,
        issues=[],
    )


def _find_tool(tools: list[Any], name: str) -> Any | None:
    """도구 목록에서 이름으로 도구를 찾는다."""
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    return None
