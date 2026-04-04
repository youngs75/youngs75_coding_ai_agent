"""OpenAI 범용 클라이언트 모듈.

사용 예시:
    from youngs75_a2a.eval_pipeline.llm.openrouter import get_openrouter_client
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model="deepseek/deepseek-v3.2",
        messages=[{"role": "user", "content": "Hello!"}],
    )
"""

from __future__ import annotations

from openai import OpenAI


def get_openrouter_client() -> OpenAI:
    """OpenAI SDK 클라이언트를 생성합니다.

    OPENAI_API_KEY 환경변수에서 API 키를 읽습니다.
    함수명은 하위 호환성을 위해 유지합니다.

    Returns:
        OpenAI: OpenAI 클라이언트
    """
    return OpenAI()
