from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import query, documents, sessions  # noqa: F401
from app.api.routes import metrics as metrics_router
from app.api.routes import health as health_router
from app.core.database import close_engine, get_session_factory, _ensure_engine_initialized
from app.services.document import DocumentService
from app.services.session import SessionManager
from config.settings import get_settings


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
    await close_engine()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Medical Knowledge Base RAG Q&A System",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )

    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(sessions.router)
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
