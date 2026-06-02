from datetime import datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sqlalchemy import text

from app.core.database import get_session_factory
from config.settings import get_settings

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
    """Check Redis connectivity using async client."""
    try:
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

        response["dependencies"] = {
            "postgresql": DependencyStatus(**pg_status),
            "qdrant": DependencyStatus(**qdrant_status),
            "redis": DependencyStatus(**redis_status),
        }

        deps = [pg_status, qdrant_status, redis_status]
        if any(d["status"] == "unhealthy" for d in deps):
            response["status"] = "degraded"

    return HealthResponse(**response)
