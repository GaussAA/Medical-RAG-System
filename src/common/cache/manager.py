"""
Redis-based caching layer for Medical RAG System.

Provides:
- CacheManager: Singleton with Redis connection pool and circuit-breaker retry
- @cached decorator: result caching with TTL
- make_cache_key: deterministic cache key from args

Design decisions:
- Circuit breaker with time-based half-open: after a failure, wait 30s before retry
- No permanent down state — the manager periodically probes Redis for recovery
- All cache miss/failure paths gracefully degrade to real-time computation
"""

import functools
import hashlib
import json
import random
import time
from collections.abc import Callable
from typing import Any

import redis.asyncio as redis
from loguru import logger

from src.common.config.settings import get_settings
from src.common.monitoring.metrics import (
    CACHE_FAILURES,
    CACHE_HITS,
    CACHE_MISSES,
)

# Default retry interval: how long to wait before re-attempting Redis connection
_DEFAULT_RETRY_SECONDS = 30
# Default TTL jitter ratio: ±10% random deviation to prevent cache stampede
_DEFAULT_TTL_JITTER = 0.1


def _apply_ttl_jitter(ttl: int, jitter_ratio: float = _DEFAULT_TTL_JITTER) -> int:
    """Apply random jitter to a TTL value.

    Returns a TTL in the range [ttl * (1 - jitter_ratio), ttl * (1 + jitter_ratio)].
    This prevents many cache entries from expiring at the same instant
    (cache stampede / thundering herd).

    Args:
        ttl: Base TTL in seconds
        jitter_ratio: Fraction of TTL to randomize (default 0.1 = ±10%)

    Returns:
        Randomized TTL, minimum 1 second
    """
    if jitter_ratio <= 0:
        return ttl
    delta = max(1, int(ttl * jitter_ratio))
    return max(1, ttl + random.randint(-delta, delta))


class CacheManager:
    """Redis cache manager with connection pool and circuit-breaker retry.

    Singleton. Uses a shared connection pool for all operations.
    Gracefully degrades when Redis is unavailable, with automatic retry.

    Circuit-breaker states:
    - **Closed** (normal): ``_down_since is None``, all operations go to Redis
    - **Open** (failing): ``_down_since`` set, calls return None for
      ``_retry_interval`` seconds without hitting the network
    - **Half-Open** (probing): after ``_retry_interval`` elapses, the next call
      attempts to reconnect; success → Closed, failure → Open again
    """

    _instance: "CacheManager | None" = None

    def __init__(self):
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._down_since: float | None = None  # time.monotonic() of last failure
        self._retry_interval: float = _DEFAULT_RETRY_SECONDS

    @classmethod
    def get_instance(cls) -> "CacheManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Circuit-breaker state machine
    # ------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        """Check whether we should skip Redis calls entirely.

        Returns True in the **Open** state — a recent failure and the retry
        interval has not yet elapsed.  Returns False in **Closed** or
        **Half-Open** states, allowing a connection attempt.
        """
        if self._down_since is None:
            return False  # Closed
        elapsed = time.monotonic() - self._down_since
        if elapsed < self._retry_interval:
            logger.debug(
                f"Cache circuit open ({elapsed:.0f}s since last failure), "
                f"next retry in {self._retry_interval - elapsed:.0f}s"
            )
            return True
        # Half-Open: enough time has passed, allow a reconnection probe
        logger.debug("Cache circuit half-open, attempting reconnection")
        return False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> redis.Redis | None:
        """Lazy-init a Redis client backed by a shared connection pool.

        Circuit-breaker with time-based half-open retry.
        After a failure, waits ``_retry_interval`` seconds before re-attempting.
        """
        if self._is_circuit_open():
            CACHE_FAILURES.labels(reason="circuit_open").inc()
            return None

        # If we already have a client, verify it is still alive
        if self._client is not None:
            try:
                await self._client.ping()  # type: ignore[misc]
                # If we were in a down state but now recovered, log it
                if self._down_since is not None:
                    downtime = time.monotonic() - self._down_since
                    self._down_since = None
                    logger.bind(event="cache_circuit_state", state="closed", downtime_s=round(downtime, 1)).info(
                        f"Circuit CLOSED (was open for {downtime:.0f}s) — Redis recovered, cache re-enabled"
                    )
                return self._client
            except Exception:
                logger.warning("Redis client ping failed, reconnecting...")
                await self._cleanup()

        # Fresh connection attempt
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
            await self._client.ping()  # type: ignore[misc]

            # Success — close the circuit
            was_down = self._down_since is not None
            self._down_since = None
            if was_down:
                logger.bind(event="cache_circuit_state", state="closed", downtime_s=None).info(
                    "Circuit CLOSED — Redis reconnected, cache re-enabled"
                )
            else:
                logger.info("Redis connection pool established")

        except Exception as e:
            self._down_since = time.monotonic()
            logger.bind(event="cache_circuit_state", state="open", retry_after_s=self._retry_interval).warning(
                f"Circuit OPEN — Redis unavailable, caching disabled (retry in {self._retry_interval}s): {e}"
            )
            await self._cleanup()

        return self._client

    async def _cleanup(self) -> None:
        """Tear down the connection pool and client."""
        if self._pool:
            try:
                await self._pool.disconnect()
            except Exception:
                pass
        self._pool = None
        self._client = None

    # ------------------------------------------------------------------
    # Public API — each operation respects the circuit-breaker
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        try:
            client = await self._ensure_client()
            if client is None:
                CACHE_MISSES.inc()
                return None
            data = await client.get(key)
            if data is None:
                CACHE_MISSES.inc()
                return None
            CACHE_HITS.inc()
            return json.loads(data)
        except Exception as e:
            CACHE_FAILURES.labels(reason="get").inc()
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        try:
            client = await self._ensure_client()
            if client is None:
                return False
            data = json.dumps(value)
            if ttl:
                jittered_ttl = _apply_ttl_jitter(ttl)
                await client.setex(key, jittered_ttl, data)
            else:
                await client.set(key, data)
            return True
        except Exception as e:
            CACHE_FAILURES.labels(reason="set").inc()
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
            CACHE_FAILURES.labels(reason="delete").inc()
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        try:
            client = await self._ensure_client()
            if client is None:
                return False
            return await client.exists(key) > 0
        except Exception as e:
            CACHE_FAILURES.labels(reason="exists").inc()
            logger.warning(f"Cache exists error for key {key}: {e}")
            return False

    async def close(self) -> None:
        await self._cleanup()

    async def reconnect_now(self) -> bool:
        """Force an immediate reconnection attempt.

        Useful for cache warmup after Redis recovery: bypasses the circuit-breaker
        retry interval and attempts to re-establish the connection right now.

        Returns:
            True if the connection was successfully established, False otherwise.
        """
        # Force reset the circuit breaker state so _ensure_client will try
        self._down_since = None
        await self._cleanup()
        client = await self._ensure_client()
        return client is not None


# ======================================================================
# Utility functions
# ======================================================================


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
