"""LLM 클라이언트 모듈.

Day3 프로젝트의 모든 LLM 호출은 OpenRouter를 통해 이루어집니다.

두 가지 클라이언트를 제공합니다:
    1. get_openrouter_client(): 범용 OpenAI SDK 클라이언트 (직접 API 호출용)
    2. get_deepeval_model(): DeepEval 전용 모델 래퍼 (메트릭/합성기용)
"""

from youngs75_a2a.eval_pipeline.llm.deepeval_model import get_deepeval_model
from youngs75_a2a.eval_pipeline.llm.openrouter import get_openrouter_client

__all__ = ["get_deepeval_model", "get_openrouter_client"]
