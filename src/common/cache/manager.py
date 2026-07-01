"""
Redis-based caching layer for Medical RAG System.

Provides:
- CacheManager: Singleton with Redis connection pool
- @cached decorator: result caching with TTL
- make_cache_key: deterministic cache key from args

ponytail: connection pool is managed automatically by redis.asyncio.client.Redis
when created without a pool, redis-py 6+ creates a single-connection client.
We use a shared pool so concurrent requests don't contend on one connection.
"""

import functools
import hashlib
import json
from collections.abc import Callable
from typing import Any

import redis.asyncio as redis
from loguru import logger

from src.common.config.settings import get_settings


class CacheManager:
    """Redis cache manager with connection pool.

    Singleton. Uses a shared connection pool for all operations.
    Gracefully degrades when Redis is unavailable.
    """

    _instance: "CacheManager | None" = None
    _pool: redis.ConnectionPool | None = None
    _client: redis.Redis | None = None

    def __init__(self):
        self._pool = None
        self._client = None

    @classmethod
    def get_instance(cls) -> "CacheManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _ensure_client(self) -> redis.Redis | None:
        """Lazy-init a Redis client backed by a shared connection pool."""
        if self._client is not None:
            return self._client
        try:
            settings = get_settings()
            rc = settings.database.redis

            self._pool = redis.ConnectionPool(
                host=rc.host,
                port=rc.port,
                db=rc.db,
                password=rc.password if rc.password else None,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
                max_connections=20,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            await self._client.ping()
            logger.info("Redis connection pool established")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            self._pool = None
            self._client = None
        return self._client

    async def get(self, key: str) -> Any | None:
        try:
            client = await self._ensure_client()
            if client is None:
                return None
            data = await client.get(key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self, key: str, value: Any, ttl: int | None = None
    ) -> bool:
        try:
            client = await self._ensure_client()
            if client is None:
                return False
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
        try:
            client = await self._ensure_client()
            if client is None:
                return False
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        try:
            client = await self._ensure_client()
            if client is None:
                return False
            return await client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False

    async def close(self) -> None:
        if self._pool:
            try:
                await self._pool.disconnect()
            except Exception:
                pass
            self._pool = None
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
):
    """Decorator to cache function results in Redis.

    Gracefully degrades: if Redis is unavailable, the function
    executes normally without caching.

    Args:
        prefix: Cache key prefix (e.g., "retrieval", "llm")
        ttl: Time to live in seconds (default 300)

    Usage:
        @cached("retrieval", ttl=300)
        async def search(query: str, top_k: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = CacheManager.get_instance()

            cache_key = make_cache_key(prefix, *args, **kwargs)

            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result

            result = await func(*args, **kwargs)

            if result is not None:
                await cache_manager.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator
