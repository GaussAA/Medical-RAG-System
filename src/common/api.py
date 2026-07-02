"""Health check API endpoints."""
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sqlalchemy import text

from src.common.config import get_settings
from src.common.database import get_session_factory

router = APIRouter(prefix="/api/v1", tags=["health"])


class DependencyStatus(BaseModel):
    status: str
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(description="Overall health status: healthy, degraded, or unhealthy")
    timestamp: str
    dependencies: dict[str, DependencyStatus] | None = None


async def check_postgresql() -> dict[str, Any]:
    """Check PostgreSQL connectivity."""
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_qdrant() -> dict[str, Any]:
    """Check Qdrant vector database connectivity."""
    try:
        settings = get_settings()
        client = QdrantClient(url=settings.database.qdrant.url)
        client.get_collections()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_redis() -> dict[str, Any]:
    """Check Redis connectivity using the CacheManager instance."""
    try:
        from src.common.cache import CacheManager

        cache_mgr = CacheManager.get_instance()
        alive = await cache_mgr.reconnect_now()
        if alive:
            return {"status": "healthy"}
        # Fallback temp client for accurate diagnostic
        settings = get_settings()
        import redis.asyncio as redis

        r = redis.Redis(
            host=settings.database.redis.host,
            port=settings.database.redis.port,
            db=settings.database.redis.db,
            password=settings.database.redis.password,
        )
        await r.ping()  # type: ignore[misc]
        await r.aclose()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_bm25() -> dict[str, Any]:
    """Check BM25 index file is accessible and non-empty."""
    try:
        from pathlib import Path
        from src.common.config import get_settings

        path = Path(get_settings().rag.retrieval.bm25_persist_path)
        if not path.exists():
            return {"status": "degraded", "error": "BM25 index file not found (will be created on first document)"}
        size = path.stat().st_size
        if size == 0:
            return {"status": "degraded", "error": "BM25 index file is empty"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/health", response_model=HealthResponse)
async def health_check(check_dependencies: bool = False) -> HealthResponse:
    """Enhanced health check endpoint with optional dependency checks."""
    response: dict[str, Any] = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }

    if check_dependencies:
        pg_status = await check_postgresql()
        qdrant_status = await check_qdrant()
        redis_status = await check_redis()
        bm25_status = await check_bm25()

        response["dependencies"] = {
            "postgresql": DependencyStatus(**pg_status),
            "qdrant": DependencyStatus(**qdrant_status),
            "redis": DependencyStatus(**redis_status),
            "bm25": DependencyStatus(**bm25_status),
        }

        deps = [pg_status, qdrant_status, redis_status, bm25_status]
        if any(d["status"] == "unhealthy" for d in deps):
            response["status"] = "degraded"

    return HealthResponse(**response)
