"""Retrieval slice — hybrid vector + BM25 retrieval with query boosting.

Public API — other modules MUST import from here.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.common.models import RetrievedNode


class HybridRetrieverPort(Protocol):
    """Interface: what retrieval provides to other modules."""

    bm25_retriever: Any

    async def add_documents(self, nodes: list[RetrievedNode]) -> None: ...

    async def delete_documents(self, ids: list[str]) -> None: ...

    async def delete_documents_atomic(self, doc_id: str, chunk_ids: list[str] | None = None) -> dict[str, Any]: ...

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> list[RetrievedNode]: ...


# Concrete implementations
from src.retrieval.base import BaseRetriever  # noqa: E402, F401
from src.retrieval.bm25 import BM25Retriever  # noqa: E402, F401
from src.retrieval.boosting import QueryBoosting  # noqa: E402, F401
from src.retrieval.hybrid import HybridRetriever  # noqa: E402, F401
from src.retrieval.reranker.cross_encoder import Reranker  # noqa: E402, F401
from src.retrieval.vector import VectorRetriever  # noqa: E402, F401

__all__ = [
    "HybridRetrieverPort",
    "HybridRetriever",
    "VectorRetriever",
    "BM25Retriever",
    "BaseRetriever",
    "QueryBoosting",
    "Reranker",
]
