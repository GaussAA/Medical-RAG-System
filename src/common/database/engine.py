"""
Database engine management - migrated from app/core/database.py.
Provides async engine singleton, session factory, and session context managers.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.common.config.settings import get_settings

_engine = None
_async_session_factory = None
_init_lock = asyncio.Lock()


async def get_engine():
    """Get or create async database engine (singleton with thread-safe initialization)."""
    global _engine
    if _engine is None:
        async with _init_lock:
            if _engine is None:
                settings = get_settings()
                _engine = create_async_engine(
                    settings.database.postgresql.url,
                    pool_size=settings.database.postgresql.pool_size,
                    max_overflow=settings.database.postgresql.max_overflow,
                    echo=False,
                )
                if _async_session_factory is not None:
                    _async_session_factory.configure(bind=_engine)
    return _engine


def get_session_factory():
    """Get or create async session factory (singleton)."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=None,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    if _engine is not None and not _async_session_factory.kw.get("bind"):
        _async_session_factory.configure(bind=_engine)
    return _async_session_factory


def _ensure_engine_initialized():
    """Synchronously ensure engine is initialized (for startup)."""
    global _engine
    if _engine is None:
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
    """Async context manager for database sessions.

    Usage:
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
    """Standalone session context manager.

    Ensures session is properly closed after use.

    Usage:
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
    """Close database engine (called on application shutdown)."""
    global _engine, _async_session_factory
    if _engine is not None:
        try:
            await _engine.dispose()
        except (RuntimeError, Exception):
            pass
        _engine = None
        _async_session_factory = None
