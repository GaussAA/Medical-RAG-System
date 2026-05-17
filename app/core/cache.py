"""
Redis-based caching layer for Medical RAG System.

Provides:
- CacheManager: Singleton class for Redis operations
- @cached decorator: Function result caching with TTL
"""

import functools
import hashlib
import json
from typing import Any, Callable

import redis.asyncio as redis
from loguru import logger

from config.settings import get_settings


class CacheManager:
    """
    Redis cache manager singleton.

    Provides get/set/delete operations with JSON serialization.
    """

    _instance: "CacheManager | None" = None
    _client: redis.Redis | None = None

    def __init__(self):
        self._client = None

    @classmethod
    def get_instance(cls) -> "CacheManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _ensure_client(self) -> redis.Redis:
        """Ensure Redis client is connected."""
        if self._client is None:
            settings = get_settings()
            redis_conf = settings.database.redis

            self._client = redis.Redis(
                host=redis_conf.host,
                port=redis_conf.port,
                db=redis_conf.db,
                password=redis_conf.password if redis_conf.password else None,
                decode_responses=False,  # We handle decoding ourselves
            )
        return self._client

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            client = await self._ensure_client()
            data = await client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        max_size: int | None = None,
    ) -> bool:
        """
        Set value in cache with optional TTL and max size.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds
            max_size: Not implemented (reserved for future use)
        """
        try:
            client = await self._ensure_client()
            data = json.dumps(value)

            if ttl:
                await client.setex(key, ttl, data)
            else:
                await client.set(key, data)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            client = await self._ensure_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            client = await self._ensure_client()
            return await client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


def make_cache_key(prefix: str, *args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_data = {
        "args": args,
        "kwargs": kwargs,
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return f"cache:{prefix}:{key_hash}"


def cached(
    prefix: str,
    ttl: int = 300,
    max_size: int | None = None,
):
    """
    Decorator to cache function results in Redis.

    Args:
        prefix: Cache key prefix (e.g., "retrieval", "llm")
        ttl: Time to live in seconds (default 300)
        max_size: Not implemented (reserved)

    Usage:
        @cached("retrieval", ttl=300)
        async def search(query: str, top_k: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = CacheManager.get_instance()

            # Generate cache key
            cache_key = make_cache_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result

            # Call function and cache result
            logger.debug(f"Cache miss for {cache_key}")
            result = await func(*args, **kwargs)

            # Only cache successful results (not None)
            if result is not None:
                await cache_manager.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


# Synchronous versions for non-async contexts
class CacheManagerSync:
    """
    Synchronous cache manager using Redis blocking client.
    For use in non-async contexts.
    """

    _instance: "CacheManagerSync | None" = None
    _client: redis.Redis | None = None

    def __init__(self):
        self._client = None

    @classmethod
    def get_instance(cls) -> "CacheManagerSync":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_client(self) -> redis.Redis:
        if self._client is None:
            settings = get_settings()
            redis_conf = settings.database.redis

            self._client = redis.Redis(
                host=redis_conf.host,
                port=redis_conf.port,
                db=redis_conf.db,
                password=redis_conf.password if redis_conf.password else None,
                decode_responses=False,
            )
        return self._client

    def get(self, key: str) -> Any | None:
        try:
            client = self._ensure_client()
            data = client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        max_size: int | None = None,
    ) -> bool:
        try:
            client = self._ensure_client()
            data = json.dumps(value)

            if ttl:
                client.setex(key, ttl, data)
            else:
                client.set(key, data)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            client = self._ensure_client()
            client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False