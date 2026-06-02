from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

# Rate limiter instance
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.deps import limiter
from app.api.routes import documents, evaluation, query, sessions  # noqa: F401
from app.api.routes import health as health_router
from app.api.routes import metrics as metrics_router
from app.core.database import (
    _ensure_engine_initialized,
    close_engine,
    get_session_factory,
)
from app.services.document import DocumentService
from app.services.session import SessionManager
from config.settings import get_settings


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
    settings = get_settings()
    print(f"Starting {settings.app.name} v{settings.app.version}")

    # 启动时从数据库恢复数据到内存
    startup_session = None
    try:
        # Ensure engine is initialized before creating session
        _ensure_engine_initialized()
        factory = get_session_factory()
        startup_session = factory()

        # 恢复文档列表
        doc_service = DocumentService(async_session=startup_session)
        await doc_service._load_documents_from_db()

        # 恢复会话列表
        sess_manager = SessionManager(async_session=startup_session)
        await sess_manager._load_sessions_from_db()

        # 将服务实例存到 app state 供后续使用
        app.state.document_service = doc_service
        app.state.session_manager = sess_manager
        app.state.startup_session = startup_session

        logger.info("Data recovery from database completed")

    except Exception as e:
        logger.error(f"Failed to recover data from database: {e}")

    yield

    # 关闭数据库连接
    print("Shutting down...")
    if hasattr(app.state, "document_service") and app.state.document_service:
        await app.state.document_service.close()
        logger.debug("Document service closed")
    if hasattr(app.state, "session_manager") and app.state.session_manager:
        await app.state.session_manager.close()
        logger.debug("Session manager closed")
    if hasattr(app.state, "startup_session") and app.state.startup_session:
        await app.state.startup_session.close()
        logger.debug("Startup session closed")
    await close_engine()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Medical Knowledge Base RAG Q&A System",
        lifespan=lifespan,
    )

    # Add rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

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

    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(sessions.router)
    app.include_router(evaluation.router)
    app.include_router(metrics_router.router)
    app.include_router(health_router.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )
