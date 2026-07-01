"""
FastAPI dependency injection functions and shared limiter.

These replace the legacy app.api.deps module.
All dependencies are resolved from the container stored in app.state.
"""

import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.common.cache.manager import CacheManager
from src.common.config.settings import Settings
from src.common.safety.checker import SafetyChecker
from src.conversation.manager import SessionManager
from src.documents.service import DocumentService
from src.query.confidence import ConfidenceEvaluator
from src.query.engine import RAGEngine

# Shared rate limiter instance
limiter = Limiter(key_func=get_remote_address)


def get_container(request: Request):
    """Get the DI container from app state."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("DI container not initialized. Ensure lifespan runs before handling requests.")
    return container


async def get_rag_engine(request: Request) -> RAGEngine:
    """FastAPI dependency: get RAGEngine from container."""
    container = get_container(request)
    if container.rag_engine is None:
        raise RuntimeError("RAGEngine not initialized in container.")
    return container.rag_engine


async def get_document_service(request: Request) -> DocumentService:
    """FastAPI dependency: get DocumentService from container."""
    container = get_container(request)
    if container.document_service is None:
        raise RuntimeError("DocumentService not initialized in container.")
    return container.document_service


async def get_session_manager(request: Request) -> SessionManager:
    """FastAPI dependency: get SessionManager from container."""
    container = get_container(request)
    if container.session_manager is None:
        raise RuntimeError("SessionManager not initialized in container.")
    return container.session_manager


async def get_safety_checker(request: Request) -> SafetyChecker:
    """FastAPI dependency: get SafetyChecker from container."""
    container = get_container(request)
    if container.safety_checker is None:
        raise RuntimeError("SafetyChecker not initialized in container.")
    return container.safety_checker


async def get_confidence_evaluator(request: Request) -> ConfidenceEvaluator:
    """FastAPI dependency: get ConfidenceEvaluator from container."""
    container = get_container(request)
    if container.confidence_evaluator is None:
        raise RuntimeError("ConfidenceEvaluator not initialized in container.")
    return container.confidence_evaluator


async def get_cache_manager(request: Request) -> CacheManager:
    """FastAPI dependency: get CacheManager from container."""
    container = get_container(request)
    return container.cache


async def get_settings_dep(request: Request) -> Settings:
    """FastAPI dependency: get Settings from container."""
    container = get_container(request)
    return container.settings


async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key from X-API-Key header."""
    expected_key = os.environ.get("API_KEY", os.environ.get("RAG_API_KEY", ""))
    if not expected_key:
        return "dev_mode"
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# Type aliases for FastAPI dependency injection
RAGEngineDep = Annotated[RAGEngine, Depends(get_rag_engine)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
SessionManagerDep = Annotated[SessionManager, Depends(get_session_manager)]
SafetyCheckerDep = Annotated[SafetyChecker, Depends(get_safety_checker)]
ConfidenceEvaluatorDep = Annotated[ConfidenceEvaluator, Depends(get_confidence_evaluator)]
CacheManagerDep = Annotated[CacheManager, Depends(get_cache_manager)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
APIKeyDep = Annotated[str, Depends(verify_api_key)]
