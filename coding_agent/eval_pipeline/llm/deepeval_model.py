"""DeepEval 전용 OpenAI 모델 래퍼.

DeepEval 라이브러리의 Synthesizer, Metric 등은 model= 인자로
DeepEvalBaseLLM 인스턴스를 받습니다. 이 모듈은 OpenAI API를
DeepEval이 이해할 수 있는 형태로 감싸는 어댑터 패턴을 구현합니다.

사용 예시:
    from coding_agent.eval_pipeline.llm.deepeval_model import get_deepeval_model
    model = get_deepeval_model()

    # Metric에 전달
    metric = AnswerRelevancyMetric(model=model, threshold=0.7)
"""

from __future__ import annotations

import json
import os

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class OpenAIModel(DeepEvalBaseLLM):
    """DeepEval 전용 OpenAI 모델 래퍼.

    DeepEvalBaseLLM을 상속하여 DeepEval 내부에서 사용하는
    generate(), a_generate(), get_model_name() 인터페이스를 구현합니다.

    Args:
        model_name: 사용할 모델명 (예: "deepseek/deepseek-v3.2"). None이면 환경변수/기본값 사용
        api_key: OpenAI API 키. None이면 OPENAI_API_KEY 환경변수 사용
    """

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        # Settings에서 모델명을 가져오되, 환경변수 OPENAI_MODEL_NAME을 먼저 확인
        if model_name is None:
            model_name = os.getenv("OPENAI_MODEL_NAME")
        if model_name is None:
            model_name = os.getenv("DEFAULT_MODEL", "qwen/qwen3.5-9b")
        self._model_name = model_name

        # OpenRouter 모델인 경우 OpenRouter API 사용
        provider = os.getenv("MODEL_PROVIDER", "openrouter")
        if provider == "openrouter" or "/" in self._model_name:
            self._client = OpenAI(
                api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            self._client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
            )

    def load_model(self):
        """모델 클라이언트를 반환합니다 (DeepEvalBaseLLM 인터페이스)."""
        return self._client

    def generate(self, prompt: str, schema=None):
        """동기 텍스트 생성."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""

        if schema:
            try:
                parsed = json.loads(content)
                return schema(**parsed)
            except Exception as exc:
                print(f"[WARN] Schema JSON parse failed ({schema.__name__}): {exc}")
                try:
                    return schema(response=content)
                except Exception:
                    return content
        return content

    async def a_generate(self, prompt: str, schema=None):
        """비동기 텍스트 생성 (동기 generate를 위임)."""
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        """모델 식별자를 반환합니다."""
        return self._model_name


# 하위 호환성을 위한 별칭
OpenRouterModel = OpenAIModel


def get_deepeval_model(model_name: str | None = None) -> OpenAIModel:
    """DeepEval 메트릭에 전달할 OpenAI 모델 인스턴스를 생성합니다."""
    return OpenAIModel(model_name=model_name)
