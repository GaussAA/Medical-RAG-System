"""Agent slice — orchestration layer for RAG (future Agent Framework).

Public API — other modules MUST import from here.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.common.models import QueryRequest, RetrievedNode
from src.conversation import SessionManagerPort


class RAGAgentPort(Protocol):
    """Interface: what agent provides to other modules."""

    async def process_document(self, nodes: list[RetrievedNode]) -> bool: ...

    async def query(
        self, request: QueryRequest, session_manager: SessionManagerPort | None = None, trace_id: str | None = None
    ) -> Any: ...

    async def query_stream(
        self,
        request: QueryRequest,
        session_manager: SessionManagerPort | None = None,
        trace_id: str | None = None,
    ) -> Any: ...


# Concrete implementations
from src.agent.confidence import ConfidenceEvaluator  # noqa: E402, F401
from src.agent.rag_agent import RAGAgent  # noqa: E402, F401

__all__ = [
    "RAGAgentPort",
    "RAGAgent",
    "ConfidenceEvaluator",
]
