"""Verification Agent 프롬프트."""

REVIEW_SYSTEM_PROMPT = """\
당신은 코드 품질 검증 전문가입니다.

생성된 코드를 다음 관점에서 검증하세요:

1. 정확성: 요구사항을 올바르게 구현했는가
2. 안전성: 보안 취약점이 없는가 (인젝션, XSS, 하드코딩된 시크릿 등)
3. 스타일: 코딩 컨벤션을 따르는가
4. 완전성: 빠진 에러 처리나 엣지 케이스가 없는가
5. 의존성: 코드에서 사용하는 패키지가 requirements.txt/package.json에 모두 포함되었는가
6. import 경로: 같은 디렉토리 내 파일은 상대 import, 패키지 구조면 절대 import를 일관되게 사용하는가

시크릿 패턴(API_KEY, PASSWORD, SECRET 등)이 코드에 하드코딩되면 즉시 실패 처리하세요.

반드시 JSON으로 반환하세요:
{{
    "passed": true/false,
    "issues": ["문제 1", "문제 2"],
    "suggestions": ["제안 1"]
}}
"""

LINT_CHECK_PROMPT = """\
다음 파일들에 대해 lint 검증을 수행하세요.
{tool_descriptions}

## 대상 파일
{file_list}

## 실행 방법
1. Python 파일이면 `run_python` 도구로 `import subprocess; result = subprocess.run(['python', '-m', 'py_compile', '<파일경로>'], capture_output=True, text=True)` 실행
2. 문법 오류가 발견되면 오류 내용을 보고하세요
3. 결과를 JSON으로 반환: {{"passed": true/false, "output": "결과 내용", "issues": ["이슈"]}}
"""

TEST_CHECK_PROMPT = """\
다음 프로젝트에 대해 테스트를 실행하세요.
{tool_descriptions}

## 대상 파일
{file_list}

## 실행 방법
1. `run_python` 도구로 `import subprocess; result = subprocess.run(['python', '-m', 'pytest', '--tb=short', '-q'], capture_output=True, text=True, cwd='<프로젝트_루트>')` 실행
2. 테스트가 없으면 `passed=True`로 반환
3. 결과를 JSON으로 반환: {{"passed": true/false, "output": "테스트 결과", "issues": ["실패한 테스트"]}}
"""
