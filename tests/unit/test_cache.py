import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.cache import CacheManager, CacheManagerSync, cached, make_cache_key

# ============================================================================
# TestMakeCacheKey
# ============================================================================


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

    def test_different_prefix_different_key(self):
        """Test that different prefixes produce different keys even with same args."""
        key1 = make_cache_key("retrieval", "query1", top_k=5)
        key2 = make_cache_key("llm", "query1", top_k=5)
        assert key1 != key2
        assert key1.startswith("cache:retrieval:")
        assert key2.startswith("cache:llm:")

    def test_empty_args(self):
        """Test that empty args produce a valid key."""
        key = make_cache_key("test")
        assert key.startswith("cache:test:")
        # Should not crash with no args
        assert len(key) > len("cache:test:")

    def test_kwargs_only(self):
        """Test that only kwargs (no positional args) works."""
        key = make_cache_key("test", key="value", number=42)
        assert key.startswith("cache:test:")

    def test_kwargs_ordering_independent(self):
        """Test that kwargs in different order produce the same key."""
        key1 = make_cache_key("test", a=1, b=2, c=3)
        key2 = make_cache_key("test", c=3, a=1, b=2)
        assert key1 == key2

    def test_nested_dict_args(self):
        """Test with nested dictionary arguments."""
        nested = {"level1": {"level2": {"key": "value"}}}
        key = make_cache_key("test", nested)
        assert key.startswith("cache:test:")

    def test_list_args(self):
        """Test with list arguments."""
        key = make_cache_key("test", [1, 2, 3, "a", "b", "c"])
        assert key.startswith("cache:test:")

    def test_complex_mixed_args(self):
        """Test with a mix of complex positional and keyword args."""
        key = make_cache_key(
            "test",
            {"nested": "dict"},
            [1, 2, 3],
            str_arg="hello",
            int_arg=42,
            bool_arg=True,
        )
        assert key.startswith("cache:test:")

    def test_special_characters(self):
        """Test with special characters in arguments."""
        key = make_cache_key("test", "line1\nline2", "tab\there")
        assert key.startswith("cache:test:")

    def test_unicode_characters(self):
        """Test with unicode characters in arguments."""
        key1 = make_cache_key("test", "你好世界", emoji="🎉")
        key2 = make_cache_key("test", "你好世界", emoji="🎉")
        assert key1 == key2
        assert key1.startswith("cache:test:")

    def test_boolean_and_none_args(self):
        """Test with boolean and None arguments."""
        key = make_cache_key("test", True, False, None, opt=None, flag=True)
        assert key.startswith("cache:test:")

    def test_numeric_args(self):
        """Test with various numeric types."""
        key = make_cache_key("test", 0, 42, -1, 3.14, float_val=-2.5)
        assert key.startswith("cache:test:")

    def test_non_json_serializable_fallback(self):
        """Test that non-JSON-serializable args fall back to str() via default=str."""
        key = make_cache_key("test", object(), lambda x: x)
        assert key.startswith("cache:test:")

    def test_consistent_non_serializable(self):
        """Test consistent output for same non-serializable input type."""

        # A custom class instance with the same str() output produces same key
        class MyObj:
            def __str__(self):
                return "MyObj-str"

        key1 = make_cache_key("test", MyObj())
        key2 = make_cache_key("test", MyObj())
        assert key1 == key2


# ============================================================================
# TestCacheManager
# ============================================================================


class TestCacheManager:
    def setup_method(self):
        """Reset singleton before each test."""
        CacheManager._instance = None
        CacheManager._client = None

    # --- Basic operations (existing tests) ---

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

    # --- Extended operations ---

    @pytest.mark.asyncio
    async def test_set_without_ttl(self):
        """Test that set without TTL uses set (not setex)."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        manager._client = mock_client

        await manager.set("test_key", {"value": "test"})

        mock_client.set.assert_called_once()
        mock_client.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_exception_returns_none(self):
        """Test that get returns None when Redis raises an exception."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = await manager.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_exception_returns_false(self):
        """Test that set returns False when Redis raises an exception."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.set = AsyncMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = await manager.set("test_key", {"value": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_exception_returns_false(self):
        """Test that delete returns False when Redis raises an exception."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = await manager.delete("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_key_not_found(self):
        """Test that exists returns False when key does not exist."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(return_value=0)
        manager._client = mock_client

        result = await manager.exists("nonexistent_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_exception_returns_false(self):
        """Test that exists returns False when Redis raises an exception."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.exists = AsyncMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = await manager.exists("test_key")
        assert result is False

    @pytest.mark.asyncio
    async def test_close_disconnects_and_clears_client(self):
        """Test that close() closes the Redis connection and clears the client."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()
        manager._client = mock_client

        await manager.close()

        mock_client.close.assert_called_once()
        assert manager._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        """Test that close() does not error when there is no client."""
        manager = CacheManager.get_instance()
        manager._client = None

        # Should not raise
        await manager.close()
        assert manager._client is None

    def test_get_instance_singleton(self):
        """Test that get_instance returns the same instance."""
        CacheManager._instance = None
        mgr1 = CacheManager.get_instance()
        mgr2 = CacheManager.get_instance()
        assert mgr1 is mgr2

    @pytest.mark.asyncio
    async def test_set_large_value(self):
        """Test setting a large nested value in cache."""
        manager = CacheManager.get_instance()
        mock_client = AsyncMock()
        mock_client.set = AsyncMock()
        manager._client = mock_client

        large_value = {"key": "val", "nested": {"items": list(range(100))}}
        result = await manager.set("large_key", large_value)
        assert result is True
        mock_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_client_creates_redis_client(self):
        """Test that _ensure_client creates a Redis client when none exists."""
        manager = CacheManager.get_instance()
        # Ensure _client is None so _ensure_client has to create it
        manager._client = None

        mock_redis_client = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.database.redis.host = "localhost"
        mock_settings.database.redis.port = 6379
        mock_settings.database.redis.db = 0
        mock_settings.database.redis.password = ""

        with (
            patch("app.core.cache.get_settings", return_value=mock_settings),
            patch("app.core.cache.redis.Redis", return_value=mock_redis_client),
        ):
            client = await manager._ensure_client()

        assert client is mock_redis_client
        assert manager._client is mock_redis_client


# ============================================================================
# TestCacheManagerSync
# ============================================================================


class TestCacheManagerSync:
    def setup_method(self):
        """Reset singleton before each test."""
        CacheManagerSync._instance = None
        CacheManagerSync._client = None

    def test_get_instance_singleton(self):
        """Test that get_instance returns the same instance."""
        mgr1 = CacheManagerSync.get_instance()
        mgr2 = CacheManagerSync.get_instance()
        assert mgr1 is mgr2

    def test_get_set_delete(self):
        """Test basic cache get/set/delete operations."""
        manager = CacheManagerSync.get_instance()

        mock_client = MagicMock()
        mock_client.get = MagicMock(return_value=None)
        mock_client.set = MagicMock()
        mock_client.setex = MagicMock()
        mock_client.delete = MagicMock()
        manager._client = mock_client

        # Test set
        result = manager.set("test_key", {"value": "test"})
        assert result is True
        mock_client.set.assert_called_once()

        # Test get returns None for no data
        mock_client.get = MagicMock(return_value=None)
        result = manager.get("test_key")
        assert result is None

        # Test delete
        result = manager.delete("test_key")
        assert result is True
        mock_client.delete.assert_called_once_with("test_key")

    def test_get_returns_deserialized_data(self):
        """Test that get returns deserialized JSON data."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()

        test_data = {"key": "value", "number": 42}
        mock_client.get = MagicMock(return_value=json.dumps(test_data).encode())
        manager._client = mock_client

        result = manager.get("test_key")
        assert result == test_data

    def test_set_with_ttl(self):
        """Test that set with TTL uses setex."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()
        mock_client.setex = MagicMock()
        manager._client = mock_client

        manager.set("test_key", {"value": "test"}, ttl=300)

        mock_client.setex.assert_called_once()
        call_args = mock_client.setex.call_args
        assert call_args[0][1] == 300  # TTL

    def test_set_without_ttl(self):
        """Test that set without TTL uses set (not setex)."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()
        mock_client.set = MagicMock()
        manager._client = mock_client

        manager.set("test_key", {"value": "test"})

        mock_client.set.assert_called_once()
        mock_client.setex.assert_not_called()

    def test_get_exception_returns_none(self):
        """Test that get returns None when Redis raises an exception."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()
        mock_client.get = MagicMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = manager.get("test_key")
        assert result is None

    def test_set_exception_returns_false(self):
        """Test that set returns False when Redis raises an exception."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()
        mock_client.set = MagicMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = manager.set("test_key", {"value": "test"})
        assert result is False

    def test_delete_exception_returns_false(self):
        """Test that delete returns False when Redis raises an exception."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()
        mock_client.delete = MagicMock(side_effect=ConnectionError("redis down"))
        manager._client = mock_client

        result = manager.delete("test_key")
        assert result is False

    def test_delete_nonexistent_key(self):
        """Test deleting a key that does not exist (Redis delete is idempotent)."""
        manager = CacheManagerSync.get_instance()
        mock_client = MagicMock()
        # Redis delete returns 0 for non-existent keys but does not raise
        mock_client.delete = MagicMock(return_value=0)
        manager._client = mock_client

        result = manager.delete("nonexistent_key")
        assert result is True
        mock_client.delete.assert_called_once_with("nonexistent_key")

    def test_ensure_client_creates_redis_client(self):
        """Test that _ensure_client creates a Redis client when none exists."""
        manager = CacheManagerSync.get_instance()
        # Ensure _client is None so _ensure_client has to create it
        manager._client = None

        mock_redis_client = MagicMock()
        mock_settings = MagicMock()
        mock_settings.database.redis.host = "localhost"
        mock_settings.database.redis.port = 6379
        mock_settings.database.redis.db = 0
        mock_settings.database.redis.password = ""

        with (
            patch("app.core.cache.get_settings", return_value=mock_settings),
            patch("app.core.cache.redis.Redis", return_value=mock_redis_client),
        ):
            client = manager._ensure_client()

        assert client is mock_redis_client
        assert manager._client is mock_redis_client


# ============================================================================
# TestCachedDecorator
# ============================================================================


class TestCachedDecorator:
    def setup_method(self):
        """Reset singleton before each test."""
        CacheManager._instance = None
        CacheManager._client = None

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

    @pytest.mark.asyncio
    async def test_cached_cache_hit(self):
        """Test that cached decorator returns cached value on cache hit."""

        call_count = 0
        cached_value = {"result": "from_cache", "score": 0.95}

        @cached("test", ttl=60)
        async def expensive_function(query: str):
            nonlocal call_count
            call_count += 1
            return {"result": f"processed {query}"}

        # Mock the cache manager
        mock_manager = MagicMock()
        mock_manager.get = AsyncMock(return_value=cached_value)
        mock_manager.set = AsyncMock()

        with patch("app.core.cache.CacheManager.get_instance", return_value=mock_manager):
            result = await expensive_function("test_query")

        assert result == cached_value
        assert call_count == 0  # Function was NOT called (cache hit)
        mock_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_cache_miss(self):
        """Test that cached decorator calls function and caches result on miss."""
        call_count = 0
        expected_result = {"result": "from_function"}

        @cached("test", ttl=60)
        async def expensive_function(query: str):
            nonlocal call_count
            call_count += 1
            return expected_result

        # Mock the cache manager
        mock_manager = MagicMock()
        mock_manager.get = AsyncMock(return_value=None)  # cache miss
        mock_manager.set = AsyncMock()

        with patch("app.core.cache.CacheManager.get_instance", return_value=mock_manager):
            result = await expensive_function("test_query")

        assert result == expected_result
        assert call_count == 1  # Function was called
        # Result should be cached
        mock_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_none_result_not_cached(self):
        """Test that None results from function are not cached."""
        call_count = 0

        @cached("test", ttl=60)
        async def returns_none(query: str):
            nonlocal call_count
            call_count += 1
            return None

        mock_manager = MagicMock()
        mock_manager.get = AsyncMock(return_value=None)  # cache miss
        mock_manager.set = AsyncMock()

        with patch("app.core.cache.CacheManager.get_instance", return_value=mock_manager):
            result = await returns_none("test_query")

        assert result is None
        assert call_count == 1  # Function was called
        # None results should NOT be cached
        mock_manager.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_cache_key_uniqueness(self):
        """Test that different function args produce different cache lookups."""
        call_count = 0
        _first_result = {"result": "first"}
        _second_result = {"result": "second"}

        @cached("test", ttl=60)
        async def func(query: str):
            nonlocal call_count
            call_count += 1
            return {"result": query}

        mock_manager = MagicMock()
        mock_manager.get = AsyncMock(return_value=None)  # always miss
        mock_manager.set = AsyncMock()

        with patch("app.core.cache.CacheManager.get_instance", return_value=mock_manager):
            _r1 = await func("query_a")
            _r2 = await func("query_b")

        # Both calls should have produced different cache keys
        assert call_count == 2
        assert mock_manager.get.call_count == 2
        key1 = mock_manager.get.call_args_list[0][0][0]
        key2 = mock_manager.get.call_args_list[1][0][0]
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_cached_preserves_function_metadata(self):
        """Test that @cached preserves function name and docstring via functools.wraps."""

        @cached("test", ttl=60)
        async def my_function(query: str):
            """My docstring."""
            return query

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."
