"""
Lightweight Dependency Injection Container.

Uses Python @dataclass pattern with lazy initialization.
No third-party DI framework required.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.cache.manager import CacheManager
from src.common.config.settings import Settings, get_settings
from src.common.database.engine import (
    _ensure_engine_initialized,
    close_engine,
    get_session_factory,
)
from src.common.database.models import Base  # noqa: F401 - ensure models are loaded
from src.common.safety.checker import SafetyChecker
from src.conversation.manager import SessionManager
from src.documents.indexer import RetrievalIndexer
from src.documents.processor import DocumentProcessor
from src.documents.service import DocumentService
from src.documents.store import DocumentStore
from src.query.confidence import ConfidenceEvaluator
from src.query.engine import RAGEngine


@dataclass
class Container:
    """Application dependency container.

    All services are assembled here and injected into the application.
    This eliminates hidden ``new XXX()`` calls within service classes.
    """

    settings: Settings = field(default_factory=get_settings)
    cache: CacheManager = field(default_factory=CacheManager.get_instance)
    db_session_factory: Callable[[], AsyncSession] | None = None
    document_service: DocumentService | None = None
    rag_engine: RAGEngine | None = None
    session_manager: SessionManager | None = None
    safety_checker: SafetyChecker | None = None
    confidence_evaluator: ConfidenceEvaluator | None = None
    document_processor: DocumentProcessor | None = None
    document_store: DocumentStore | None = None
    retrieval_indexer: RetrievalIndexer | None = None

    async def close(self) -> None:
        """Close all managed resources."""
        if self.cache:
            await self.cache.close()
        if self.session_manager:
            await self.session_manager.close()
        if self.document_service:
            await self.document_service.close()
        if self.document_store:
            await self.document_store.close_session()
        await close_engine()


async def create_container() -> Container:
    """Create and wire the application dependency container.

    This factory ensures all dependencies are resolved in the correct order.
    Returns a fully wired container ready for use.
    """
    settings = get_settings()

    # 1. Initialize database
    _ensure_engine_initialized()
    factory = get_session_factory()

    # 2. Initialize cache (Redis, gracefully degrades)
    cache = CacheManager.get_instance()

    # 3. Init async db session for services
    session = factory()

    # 4. Build infrastructure services
    safety_checker = SafetyChecker()
    confidence_evaluator = ConfidenceEvaluator()
    session = factory()

    # 5. Build session manager (needed by RAGEngine)
    session_manager = SessionManager(async_session=session, cache_manager=cache)

    # 6. Build RAG engine with injected dependencies
    rag_engine = RAGEngine(
        safety_checker=safety_checker,
        session_manager=session_manager,
    )

    # 7. Build document slice
    document_processor = DocumentProcessor()
    document_store = DocumentStore(async_session=session)
    retrieval_indexer = RetrievalIndexer()

    # 8. Build document service with explicit DI injection
    document_service = DocumentService(
        rag_engine=rag_engine,
        processor=document_processor,
        store=document_store,
        indexer=retrieval_indexer,
        async_session=session,
    )

    container = Container(
        settings=settings,
        cache=cache,
        db_session_factory=factory,
        document_service=document_service,
        rag_engine=rag_engine,
        session_manager=session_manager,
        safety_checker=safety_checker,
        confidence_evaluator=confidence_evaluator,
        document_processor=document_processor,
        document_store=document_store,
        retrieval_indexer=retrieval_indexer,
    )

    return container
