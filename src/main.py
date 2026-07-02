"""
FastAPI application entry point (new architecture).

This is the new entry point for the refactored application.
During migration, old app/main.py remains as fallback.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.agent import api as agent_api
from src.common import api as health_api
from src.common.config import get_settings
from src.common.di import create_container
from src.common.logging import setup_logging
from src.common.monitoring import metrics_router
from src.conversation import api as conversation_api

# Route imports (from new vertical slices)
from src.documents import api as documents_api
from src.evaluation import api as evaluation_api


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup on start, cleanup on shutdown."""
    settings = get_settings()
    logger.info(f"Starting {settings.app.name} v{settings.app.version}")

    # Initialize logging
    setup_logging()

    # Create DI container
    container = await create_container()
    app.state.container = container

    # 预热 embedding + reranker 模型到 GPU（幂等，热重载不会重复申请显存）
    try:
        if container.rag_engine is not None:
            status = await container.rag_engine.warmup_models()
            if not all(status.values()):
                logger.warning("Some models failed to load, system may have reduced functionality")
        else:
            logger.warning("RAGAgent not initialized, skipping model warmup")
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await container.close()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Medical Knowledge Base RAG Q&A System",
        lifespan=lifespan,
    )

    # Rate limiter
    try:
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    except Exception as e:
        logger.warning(f"Rate limiter setup failed: {e}")

    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    @app.get("/")
    async def root():
        """Root endpoint — redirects to API documentation."""
        return RedirectResponse(url="/docs")

    # Register routes from vertical slices
    app.include_router(agent_api.router)
    app.include_router(documents_api.router)
    app.include_router(conversation_api.router)
    app.include_router(evaluation_api.router)
    app.include_router(health_api.router)
    app.include_router(metrics_router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
