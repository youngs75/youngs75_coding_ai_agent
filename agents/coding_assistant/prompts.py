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

from youngs75_a2a.core.project_context import ProjectContextLoader


def inject_project_context(
    base_prompt: str,
    workspace: str | None = None,
    max_context_tokens: int = 4000,
) -> str:
    """시스템 프롬프트에 프로젝트 컨텍스트를 주입한다.

    workspace가 지정되면 ProjectContextLoader로 컨텍스트를 검색하여
    기본 프롬프트 뒤에 추가한다. 컨텍스트가 없으면 원본 프롬프트를 그대로 반환.

    Args:
        base_prompt: 기본 시스템 프롬프트
        workspace: 프로젝트 워크스페이스 경로 (None이면 주입하지 않음)
        max_context_tokens: 최대 컨텍스트 토큰 수

    Returns:
        프로젝트 컨텍스트가 주입된 시스템 프롬프트
    """
    if workspace is None:
        return base_prompt

    loader = ProjectContextLoader(workspace, max_context_tokens=max_context_tokens)
    section = loader.build_system_prompt_section()
    if not section:
        return base_prompt

    return base_prompt + section


PARSE_SYSTEM_PROMPT = """\
당신은 소프트웨어 개발 요청을 분석하는 전문가입니다.

사용자의 요청을 분석하여 다음을 JSON 형식으로 반환하세요:
- task_type: "generate" | "fix" | "refactor" | "explain" | "analyze" | "scaffold" 중 하나
  - scaffold: 새 프로젝트/앱 생성 (보일러플레이트, 프레임워크 셋업)
  - generate: 새로운 코드 작성 (기존 프로젝트에 기능 추가)
  - fix: 기존 코드의 버그 수정
  - refactor: 기존 코드 개선/리팩토링
  - explain: 코드나 개념 설명
  - analyze: 파일/프로젝트 분석 (파일 읽기가 필요한 작업)
- framework: scaffold일 때 사용할 프레임워크 조합 (예: "flask_vue", "react_express", "fastapi_react", "django_htmx"). 요청에서 명시된 백엔드/프론트엔드 기술을 조합하여 판단. scaffold가 아니면 빈 문자열.
- language: 요청에서 감지된 프로그래밍 언어 (명시되지 않으면 요청 내용에서 추론, 추론 불가 시 "python")
- description: 작업에 대한 간결한 설명
- target_files: 관련 파일 경로 리스트 (언급된 경우)
- requirements: 세부 요구사항 리스트

반드시 JSON만 반환하세요. 다른 텍스트는 포함하지 마세요.
"""

EXECUTE_SYSTEM_PROMPT = """\
당신은 숙련된 {language} 개발자입니다.

## 사용 가능한 도구
{tool_descriptions}

## 핵심 원칙: 도구 사용 최소화
- **새 코드 생성 (generate)**: 도구를 호출하지 말고 즉시 코드를 작성하세요. list_directory나 read_file은 불필요합니다.
- **기존 코드 수정 (fix/refactor)**: `read_file`로 대상 파일을 먼저 읽은 후 수정하세요.
- **파일 분석 (analyze)**: `read_file`로 해당 파일을 읽으세요.
- 도구는 **기존 파일을 읽거나 수정해야 할 때만** 사용하세요.
- `list_directory`는 파일 경로를 모를 때만 1회 호출하세요. 반복 호출하지 마세요.

## 금지 경로
`.claude/`, `.git/`, `__pycache__/` 내 파일은 생성/수정하지 마세요.

## 규칙
- 요구사항에 집중하여 필요한 코드만 작성하세요
- 코드는 즉시 실행 가능한 상태여야 합니다
- {language}으로 작성하세요
- **이전 Phase에서 생성된 파일이 언급된 경우**: 해당 파일을 `read_file`로 먼저 읽고, **필요한 부분만 수정**하세요. 파일 전체를 처음부터 다시 작성하지 마세요. 이전 phase의 코드가 유실됩니다.

## import/모듈 경로 규칙
- 지시사항에 명시된 import 경로가 있으면 그것을 우선 따르세요
- 같은 프로젝트 내 모든 파일에서 **동일한 import 스타일**을 일관되게 유지하세요
- 언어별 기본 규칙:
  - **Python**: `__init__.py`가 있으면 패키지 → 절대 import, 없으면 상대 import
  - **JavaScript/TypeScript**: `import {{ X }} from './module'` 상대 경로 사용
  - **Go**: `import "project/pkg/module"` (go.mod 기준)
  - **Rust**: `mod module;` 또는 `use crate::module;`
  - **Java**: `import com.project.package.Class;`

## 순환 Import 방지 (필수)
파일 간 import가 **DAG(방향 비순환 그래프)**를 형성하는지 반드시 확인하세요.
"A imports B, B imports A" 패턴이 감지되면 즉시 중간 모듈로 분리하세요.

- **Python**: 공유 객체(db, config 등)를 `extensions.py`로 분리하고, Blueprint/라우터는 app을 직접 import하지 말 것.
  Factory 패턴(`create_app()`)으로 blueprint를 등록하세요.
  ```
  extensions.py: db = SQLAlchemy()
  models.py: from extensions import db
  blueprints/api.py: from models import User  # app을 import하지 않음
  app.py: from extensions import db; app.register_blueprint(api_bp)
  ```
- **JavaScript/TypeScript**: barrel export(index.ts)에서 순환 re-export 주의. 공유 의존성은 별도 모듈로 분리.
- **Go**: 패키지 간 순환 import은 컴파일 에러. interface를 상위 패키지에 정의하고 구현을 하위 패키지에 배치.

파일 생성 전에 import 그래프를 먼저 설계하고, 순환이 없는지 검증한 후 코드를 작성하세요.

## 파일 저장 형식 (필수)
**각 파일은 반드시 별도의 코드 블록으로 분리**하고, 첫 줄에 파일 경로를 주석으로 명시하세요:
- Python/Ruby/Shell: `# filepath: path/to/file.py`
- JavaScript/TypeScript/Go/Rust/Java/C/C++: `// filepath: path/to/file.js`
- HTML/XML/Vue/Svelte: `<!-- filepath: path/to/file.html -->`
- CSS/SCSS: `/* filepath: path/to/style.css */`
- TOML: `# filepath: Cargo.toml`
- YAML: `# filepath: config.yaml`
이 형식을 따르면 코드가 자동으로 파일에 저장됩니다.
디렉토리 구조가 필요하면 경로에 포함하세요 (예: `src/main.rs`, `cmd/app/main.go`).
**하나의 코드 블록에 여러 파일을 합치지 마세요. 파일당 1개의 코드 블록을 사용하세요.**

## 의존성 관리
- 의존성 파일이 프로젝트에 포함된 경우, 코드에서 사용하는 **모든 외부 패키지**를 빠짐없이 추가하세요
- 언어별 의존성 파일: requirements.txt(Python), package.json(JS/TS), go.mod(Go), Cargo.toml(Rust), pom.xml/build.gradle(Java)
- 표준 라이브러리는 제외하세요

## 테스트 코드 필수 생성
- 코드를 생성할 때 **반드시 해당 코드에 대한 테스트 파일도 함께 생성**하세요
- 테스트는 핵심 로직(모델, API, 유틸 함수)에 대해 작성하세요. UI/템플릿에 대한 테스트는 선택입니다.
- 테스트 파일 위치: `tests/` 디렉토리 또는 소스 파일 옆 (예: `test_models.py`, `models.test.js`)
- 테스트는 **실제 실행 가능한 상태**여야 합니다. 생성 후 자동으로 실행됩니다.
- 테스트가 실패하면 코드 수정을 요청받게 되므로, 테스트가 통과하는 코드를 작성하세요.

## 응답 형식
- 먼저 변경 계획을 간단히 설명
- 코드 블록으로 전체 코드를 제공 (파일 경로 주석 포함)
- **테스트 코드 블록도 포함** (filepath 주석 포함)
- 변경된 부분을 요약
"""

VERIFY_SYSTEM_PROMPT = """\
당신은 코드 품질 검증 전문가입니다.

생성된 코드를 다음 관점에서 검증하세요:

1. 정확성: 요구사항을 올바르게 구현했는가
2. 안전성: 보안 취약점이 없는가 (인젝션, XSS, 하드코딩된 시크릿 등)
3. 스타일: 코딩 컨벤션을 따르는가
4. 완전성: 빠진 에러 처리나 엣지 케이스가 없는가
5. 의존성: 불필요한 의존성을 추가하지 않았는가, 코드에서 사용하는 패키지가 requirements.txt/package.json에 **모두 포함**되었는가
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
