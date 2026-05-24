import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from rag.generation.llm_generator import LLMGenerator


class TestLLMGenerator:
    def setup_method(self):
        """Reset class-level client before each test."""
        LLMGenerator._client = None
        LLMGenerator._client_config = None

    def test_singleton_client_shared(self):
        """Test that multiple LLMGenerator instances share the same client."""
        gen1 = LLMGenerator()
        gen2 = LLMGenerator()

        # Both should reference the same client
        assert gen1.client is gen2.client

    def test_singleton_client_config_same_when_same_params(self):
        """Test that client is reused when API key and base URL are the same."""
        gen1 = LLMGenerator(api_key="test-key-1", api_base="http://test1.com")
        gen2 = LLMGenerator(api_key="test-key-1", api_base="http://test1.com")

        assert gen1.client is gen2.client

    def test_singleton_client_recreated_when_config_changes(self):
        """Test that client is recreated when API key or base URL changes."""
        gen1 = LLMGenerator(api_key="test-key-1", api_base="http://test1.com")
        client1 = gen1.client

        gen2 = LLMGenerator(api_key="test-key-2", api_base="http://test2.com")

        # Client should be different
        assert gen2.client is not client1

    @pytest.mark.asyncio
    async def test_generate_with_retry_decorator(self):
        """Test that generate method works with the retry mechanism."""
        gen = LLMGenerator()

        # Mock the _call_with_retry method
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test answer"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        with patch.object(gen, '_call_with_retry', new_callable=AsyncMock, return_value=mock_response) as mock_call:
            result = await gen.generate(
                query="测试问题",
                contexts=[],
            )

            assert result["answer"] == "Test answer"
            assert result["usage"]["prompt_tokens"] == 10
            assert result["usage"]["completion_tokens"] == 20
            mock_call.assert_called_once()


class TestLLMSingletonImport:
    """Test class-level client singleton behavior."""

    def setup_method(self):
        LLMGenerator._client = None
        LLMGenerator._client_config = None

    def test_client_initially_none(self):
        """Test that class-level client starts as None."""
        LLMGenerator._client = None
        gen = LLMGenerator()
        # Client should be created on first access
        assert gen.client is not None

    def test_property_creates_client_if_none(self):
        """Test that accessing client property creates it if None."""
        LLMGenerator._client = None
        gen = LLMGenerator()
        client = gen.client
        assert client is not None
        # Same instance should be returned
        assert gen.client is client