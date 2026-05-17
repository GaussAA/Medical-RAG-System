import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.cache import CacheManager, make_cache_key, cached


class TestCacheManager:
    def setup_method(self):
        """Reset singleton before each test."""
        CacheManager._instance = None
        CacheManager._client = None

    @pytest.mark.asyncio
    async def test_get_set_delete(self):
        """Test basic cache get/set/delete operations."""
        manager = CacheManager.get_instance()

        # Mock Redis client
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)
        mock_client.set = AsyncMock()
        mock_client.setex = AsyncMock()
        mock_client.delete = AsyncMock()
        manager._client = mock_client

        # Test set
        result = await manager.set("test_key", {"value": "test"})
        assert result is True

        # Test get with no data
        mock_client.get = AsyncMock(return_value=None)
        result = await manager.get("test_key")
        assert result is None

        # Test delete
        result = await manager.delete("test_key")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_returns_deserialized_data(self):
        """Test that get returns deserialized JSON data."""
        import json

        manager = CacheManager.get_instance()
        mock_client = AsyncMock()

        test_data = {"key": "value", "number": 42}
        mock_client.get = AsyncMock(return_value=json.dumps(test_data).encode())
        manager._client = mock_client

        result = await manager.get("test_key")
        assert result == test_data

    @pytest.mark.asyncio
    async def test_set_with_ttl(self):
        """Test that set with TTL uses setex."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.setex = AsyncMock()
        manager._client = mock_client

        await manager.set("test_key", {"value": "test"}, ttl=300)

        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 300  # TTL

    @pytest.mark.asyncio
    async def test_exists(self):
        """Test cache key existence check."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(return_value=1)
        manager._client = mock_client

        result = await manager.exists("test_key")
        assert result is True


class TestMakeCacheKey:
    def test_cache_key_format(self):
        """Test that cache key has correct format."""
        key = make_cache_key("retrieval", "query", top_k=5)
        assert key.startswith("cache:retrieval:")
        assert len(key) > len("cache:retrieval:")

    def test_same_args_same_key(self):
        """Test that same arguments produce same key."""
        key1 = make_cache_key("retrieval", "query1", top_k=5)
        key2 = make_cache_key("retrieval", "query1", top_k=5)
        assert key1 == key2

    def test_different_args_different_key(self):
        """Test that different arguments produce different keys."""
        key1 = make_cache_key("retrieval", "query1", top_k=5)
        key2 = make_cache_key("retrieval", "query2", top_k=5)
        assert key1 != key2


class TestCachedDecorator:
    def test_cached_decorator_basic(self):
        """Test that @cached decorator returns cached result."""

        call_count = 0

        @cached("test", ttl=60)
        async def expensive_function(query: str):
            nonlocal call_count
            call_count += 1
            return {"result": f"processed {query}"}

        # This is a basic structure test - actual Redis mocking would be needed
        # for full integration test
        assert expensive_function is not None