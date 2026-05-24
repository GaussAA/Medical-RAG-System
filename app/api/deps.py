from typing import Annotated
import asyncio
import os

from fastapi import Depends, Header, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.confidence import ConfidenceEvaluator
from app.core.rag_engine import RAGEngine
from app.core.safety import SafetyChecker

# Rate limiter instance - must be defined before any route imports
limiter = Limiter(key_func=get_remote_address)

_rag_engine: RAGEngine | None = None
_safety_checker: SafetyChecker | None = None
_confidence_evaluator: ConfidenceEvaluator | None = None
_deps_lock = asyncio.Lock()


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        return RAGEngine()
    return _rag_engine


async def get_rag_engine_async() -> RAGEngine:
    """Thread-safe async version of get_rag_engine."""
    global _rag_engine
    if _rag_engine is None:
        async with _deps_lock:
            if _rag_engine is None:
                _rag_engine = RAGEngine()
    return _rag_engine


def get_safety_checker() -> SafetyChecker:
    global _safety_checker
    if _safety_checker is None:
        _safety_checker = SafetyChecker()
    return _safety_checker


def get_confidence_evaluator() -> ConfidenceEvaluator:
    global _confidence_evaluator
    if _confidence_evaluator is None:
        _confidence_evaluator = ConfidenceEvaluator()
    return _confidence_evaluator


async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Verify API key from X-API-Key header."""
    expected_key = os.environ.get("API_KEY", os.environ.get("RAG_API_KEY", ""))
    if not expected_key:
        # If no API key configured, allow access (legacy mode for development)
        return "dev_mode"
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


RAGEngineDep = Annotated[RAGEngine, Depends(get_rag_engine)]
SafetyCheckerDep = Annotated[SafetyChecker, Depends(get_safety_checker)]
ConfidenceEvaluatorDep = Annotated[ConfidenceEvaluator, Depends(get_confidence_evaluator)]
APIKeyDep = Annotated[str, Depends(verify_api_key)]
