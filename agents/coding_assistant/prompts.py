"""Coding Assistant 노드별 시스템 프롬프트.

각 노드는 역할이 명확히 분리되어 있다:
- parse: 요청 분석만 수행
- execute: MCP 도구로 컨텍스트 수집 + 코드 생성/수정 (검증 정보 미제공)
- verify: 더 넓은 맥락에서 검증 (특권 정보 보유)

프롬프트 버전 관리:
- PROMPT_REGISTRY: 프롬프트 이름별 버전 이력을 관리
- get_prompt(): 특정 버전의 프롬프트 반환
- apply_remediation(): remediation 결과로 프롬프트 개선 적용
"""

from __future__ import annotations

PARSE_SYSTEM_PROMPT = """\
당신은 소프트웨어 개발 요청을 분석하는 전문가입니다.

사용자의 요청을 분석하여 다음을 JSON 형식으로 반환하세요:
- task_type: "generate" | "fix" | "refactor" | "explain" 중 하나
- language: 요청에서 감지된 프로그래밍 언어 (명시되지 않으면 "python")
- description: 작업에 대한 간결한 설명
- target_files: 관련 파일 경로 리스트 (언급된 경우)
- requirements: 세부 요구사항 리스트

반드시 JSON만 반환하세요. 다른 텍스트는 포함하지 마세요.
"""

EXECUTE_SYSTEM_PROMPT = """\
당신은 숙련된 {language} 개발자이며, 프로젝트 코드베이스에 접근할 수 있는 도구를 보유하고 있습니다.

## 사용 가능한 도구
{tool_descriptions}

## 프로젝트 구조
이 프로젝트의 코드베이스는 `youngs75_a2a/` 디렉토리에 있습니다.
- `youngs75_a2a/core/` — 공통 프레임워크 (BaseGraphAgent, BaseAgentConfig 등)
- `youngs75_a2a/agents/` — 에이전트 구현체
- `youngs75_a2a/a2a/` — A2A 프로토콜
반드시 `youngs75_a2a.` 패키지 경로를 사용하세요. `Day-04/`, `src/` 등 다른 경로를 사용하지 마세요.

## 작업 절차
1. `search_code`나 `list_directory`로 `youngs75_a2a/` 하위에서 관련 코드를 찾으세요
2. `read_file`로 기존 코드의 import 경로, 클래스 구조, 네이밍 컨벤션을 파악하세요
3. {language}으로 코드를 작성하세요. 다른 언어로 작성하지 마세요
4. 필요 시 `run_python`으로 코드 실행을 검증하세요

## 규칙
- 요구사항에 집중하여 필요한 코드만 작성하세요
- 코드는 즉시 실행 가능한 상태여야 합니다
- 프로젝트의 기존 클래스나 함수를 참조할 때는 `youngs75_a2a/` 내에서 실제 import 경로를 확인하세요
- 도구 호출은 최소한으로 하세요. 핵심 파일 1~3개만 읽으면 충분합니다

## 인용 형식 규칙
코드나 문서를 참조할 때는 반드시 출처를 명시하세요:
- 코드 참조: `파일: path/to/file.py:42` 형식으로 파일 경로와 라인 번호를 기재
- 외부 문서/API 참조: 출처 URL 또는 공식 문서명을 기재 (예: `출처: Python 공식 문서 — asyncio`)
- 기존 코드 수정 시: 원본 코드 블록을 인용한 뒤 변경 사항을 설명

예시:
```
# 파일: youngs75_a2a/core/base_agent.py:15
# 기존 코드:
#   def run(self): ...
# 변경: run()에 timeout 파라미터 추가
def run(self, timeout: int = 30): ...
```

## 응답 형식 (모든 도구 호출이 끝난 후 최종 응답)
- 먼저 변경 계획을 간단히 설명
- 코드 블록으로 전체 코드를 제공
- 변경된 부분을 요약
"""

VERIFY_SYSTEM_PROMPT = """\
당신은 코드 품질 검증 전문가입니다.

생성된 코드를 다음 관점에서 검증하세요:

1. 정확성: 요구사항을 올바르게 구현했는가
2. 안전성: 보안 취약점이 없는가 (인젝션, XSS, 하드코딩된 시크릿 등)
3. 스타일: 코딩 컨벤션을 따르는가
4. 완전성: 빠진 에러 처리나 엣지 케이스가 없는가
5. 의존성: 불필요한 의존성을 추가하지 않았는가
6. 프로젝트 적합성: 기존 프로젝트 구조와 패턴에 맞는가
7. 인용 품질: 코드 참조에 파일 경로/라인 번호가 포함되어 있는가

검증자 특권 정보 (코드 생성자에게는 미제공):
- 파일 삭제 시 한 번에 {max_delete_lines}줄 이상 삭제는 위험 신호
- 허용된 파일 확장자: {allowed_extensions}
- 시크릿 패턴(API_KEY, PASSWORD, SECRET 등)이 코드에 포함되면 즉시 실패

반드시 JSON 형식으로 반환하세요:
- passed: true/false
- issues: 발견된 문제 리스트 (없으면 빈 리스트)
- suggestions: 개선 제안 리스트 (없으면 빈 리스트)
"""


# ── 프롬프트 버전 관리 ──


class PromptVersion:
    """단일 프롬프트 버전 정보."""

    def __init__(self, *, version: str, content: str, source: str = "initial"):
        self.version = version
        self.content = content
        self.source = source  # "initial" | "remediation" 등


class PromptRegistry:
    """프롬프트 이름별 버전 이력을 관리하는 레지스트리.

    사용 예시:
        registry = PromptRegistry()
        current = registry.get_prompt("parse")
        registry.apply_remediation(changes)
        updated = registry.get_prompt("parse")  # v2
    """

    def __init__(self) -> None:
        self._prompts: dict[str, list[PromptVersion]] = {}
        # 초기 프롬프트 등록 (v1)
        self._register_initial("parse", PARSE_SYSTEM_PROMPT)
        self._register_initial("execute", EXECUTE_SYSTEM_PROMPT)
        self._register_initial("verify", VERIFY_SYSTEM_PROMPT)

    def _register_initial(self, name: str, content: str) -> None:
        """초기(v1) 프롬프트를 등록합니다."""
        self._prompts[name] = [
            PromptVersion(version="v1", content=content, source="initial")
        ]

    def get_prompt(self, name: str, *, version: str | None = None) -> str:
        """특정 프롬프트의 최신 또는 지정 버전을 반환합니다.

        Args:
            name: 프롬프트 이름 (parse, execute, verify)
            version: 특정 버전 (None이면 최신 버전)

        Returns:
            프롬프트 내용 문자열

        Raises:
            KeyError: 프롬프트 이름이 존재하지 않을 때
            ValueError: 지정 버전이 존재하지 않을 때
        """
        if name not in self._prompts:
            raise KeyError(f"프롬프트 '{name}'이(가) 등록되지 않았습니다")

        versions = self._prompts[name]
        if version is None:
            return versions[-1].content

        for v in versions:
            if v.version == version:
                return v.content
        raise ValueError(f"프롬프트 '{name}'의 버전 '{version}'을(를) 찾을 수 없습니다")

    def get_current_version(self, name: str) -> str:
        """프롬프트의 현재 버전 문자열을 반환합니다."""
        if name not in self._prompts:
            raise KeyError(f"프롬프트 '{name}'이(가) 등록되지 않았습니다")
        return self._prompts[name][-1].version

    def list_versions(self, name: str) -> list[str]:
        """프롬프트의 모든 버전 목록을 반환합니다."""
        if name not in self._prompts:
            raise KeyError(f"프롬프트 '{name}'이(가) 등록되지 않았습니다")
        return [v.version for v in self._prompts[name]]

    def list_prompts(self) -> list[str]:
        """등록된 모든 프롬프트 이름 목록을 반환합니다."""
        return list(self._prompts.keys())

    def apply_remediation(self, changes: list[dict[str, str]]) -> list[str]:
        """Remediation 결과를 반영하여 프롬프트를 개선합니다.

        RecommendationReport.get_prompt_changes()의 출력을 받아,
        해당 프롬프트에 개선 지시를 추가한 새 버전을 생성합니다.

        Args:
            changes: get_prompt_changes() 결과 리스트.
                각 항목: {target_prompt, issue, change, metric}

        Returns:
            업데이트된 프롬프트 이름 목록
        """
        updated: list[str] = []

        # 프롬프트 이름 매핑 (유연한 매칭)
        name_map = {
            "parse": "parse",
            "parser": "parse",
            "execute": "execute",
            "executor": "execute",
            "generation": "execute",
            "verify": "verify",
            "verifier": "verify",
            "verification": "verify",
        }

        for change in changes:
            target = change.get("target_prompt", "").lower().strip()
            prompt_name = name_map.get(target)
            if prompt_name is None or prompt_name not in self._prompts:
                continue

            current = self._prompts[prompt_name][-1]
            current_ver_num = int(current.version.lstrip("v"))
            new_ver = f"v{current_ver_num + 1}"

            # 기존 프롬프트에 개선 지시 섹션 추가
            improvement = (
                f"\n\n## 개선 사항 ({new_ver})\n"
                f"문제: {change.get('issue', '')}\n"
                f"변경: {change.get('change', '')}\n"
                f"예상 효과: {change.get('metric', '')}"
            )
            new_content = current.content + improvement

            self._prompts[prompt_name].append(
                PromptVersion(
                    version=new_ver,
                    content=new_content,
                    source="remediation",
                )
            )
            if prompt_name not in updated:
                updated.append(prompt_name)

        return updated


# 전역 레지스트리 인스턴스
_prompt_registry: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    """전역 PromptRegistry 인스턴스를 반환합니다 (싱글턴)."""
    global _prompt_registry
    if _prompt_registry is None:
        _prompt_registry = PromptRegistry()
    return _prompt_registry


def reset_prompt_registry() -> None:
    """전역 PromptRegistry를 초기화합니다 (테스트용)."""
    global _prompt_registry
    _prompt_registry = None
