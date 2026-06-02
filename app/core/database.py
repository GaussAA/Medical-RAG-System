import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import get_settings

_engine = None
_async_session_factory = None
_init_lock = asyncio.Lock()


async def get_engine():
    """Get or create async database engine (singleton with thread-safe initialization)."""
    global _engine
    if _engine is None:
        async with _init_lock:
            if _engine is None:  # double-check after acquiring lock
                settings = get_settings()
                _engine = create_async_engine(
                    settings.database.postgresql.url,
                    pool_size=settings.database.postgresql.pool_size,
                    max_overflow=settings.database.postgresql.max_overflow,
                    echo=False,
                )
                # Update session factory bind if it exists
                if _async_session_factory is not None:
                    _async_session_factory.configure(bind=_engine)
    return _engine


def get_session_factory():
    """Get or create async session factory (singleton).

    Returns a factory bound to the engine. This will synchronously
    initialize the engine if not already done.
    """
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=None,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    # Ensure engine is initialized and bind is configured
    if _engine is not None and not _async_session_factory.kw.get("bind"):
        _async_session_factory.configure(bind=_engine)
    return _async_session_factory


def _ensure_engine_initialized():
    """Synchronously ensure engine is initialized (for startup)."""
    global _engine
    if _engine is None:
        # For sync context, we need to initialize engine directly
        settings = get_settings()
        _engine = create_async_engine(
            settings.database.postgresql.url,
            pool_size=settings.database.postgresql.pool_size,
            max_overflow=settings.database.postgresql.max_overflow,
            echo=False,
        )
        if _async_session_factory is not None:
            _async_session_factory.configure(bind=_engine)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    异步上下文管理器，用于获取数据库会话。
    用法:
        async with get_session() as session:
            await session.execute(...)
            await session.commit()
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_standalone_session():
    """
    Standalone session context manager for use with services that accept async_session.
    Ensures session is properly closed after use.

    用法:
        async with get_standalone_session() as session:
            service = DocumentService(async_session=session)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        await session.close()


async def close_engine():
    """关闭数据库引擎（应用 shutdown 时调用）"""
    global _engine, _async_session_factory
    if _engine is not None:
        # dispose 会关闭池中所有连接，catch RuntimeError 以防 event loop 已关闭
        try:
            await _engine.dispose()
        except (RuntimeError, Exception):
            pass
        _engine = None
        _async_session_factory = None
