from __future__ import annotations

from unittest.mock import MagicMock, patch

from youngs75_a2a.eval_pipeline.llm.deepeval_model import OpenRouterModel


class TestOpenRouterModel:
    def test_model_name(self):
        """모델명이 올바르게 설정되는지 테스트합니다."""
        model = OpenRouterModel(model_name="openai/gpt-5.4", api_key="test-key")
        assert model.get_model_name() == "openai/gpt-5.4"

    def test_generate(self):
        """generate 메서드가 올바른 텍스트를 반환하는지 테스트합니다."""
        model = OpenRouterModel(model_name="test-model", api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]

        with patch.object(
            model._client.chat.completions, "create", return_value=mock_response
        ):
            result = model.generate("test prompt")
            assert result == "test response"

    def test_generate_with_schema(self):
        """generate 메서드가 schema를 적용하여 파싱하는지 테스트합니다."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str

        model = OpenRouterModel(model_name="test-model", api_key="test-key")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"answer": "hello"}'))
        ]

        with patch.object(
            model._client.chat.completions, "create", return_value=mock_response
        ):
            result = model.generate("test", schema=TestSchema)
            assert isinstance(result, TestSchema)
            assert result.answer == "hello"
