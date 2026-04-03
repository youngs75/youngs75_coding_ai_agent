from __future__ import annotations

from unittest.mock import MagicMock, patch

from youngs75_a2a.eval_pipeline.llm.deepeval_model import OpenRouterModel


class TestOpenRouterModel:
    def test_model_name(self):
        with patch("youngs75_a2a.eval_pipeline.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openrouter_api_key="test-key",
                openrouter_model_name="openai/gpt-5.4",
            )
            model = OpenRouterModel(model_name="openai/gpt-5.4", api_key="test-key")
            assert model.get_model_name() == "openai/gpt-5.4"

    def test_generate(self):
        with patch("youngs75_a2a.eval_pipeline.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openrouter_api_key="test-key",
                openrouter_model_name="openai/gpt-5.4",
            )
            model = OpenRouterModel(model_name="test-model", api_key="test-key")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]

            with patch.object(model._client.chat.completions, "create", return_value=mock_response):
                result = model.generate("test prompt")
                assert result == "test response"

    def test_generate_with_schema(self):
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str

        with patch("youngs75_a2a.eval_pipeline.llm.deepeval_model.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                openrouter_api_key="test-key",
                openrouter_model_name="openai/gpt-5.4",
            )
            model = OpenRouterModel(model_name="test-model", api_key="test-key")

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content='{"answer": "hello"}'))]

            with patch.object(model._client.chat.completions, "create", return_value=mock_response):
                result = model.generate("test", schema=TestSchema)
                assert isinstance(result, TestSchema)
                assert result.answer == "hello"
