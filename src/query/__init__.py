"""Query slice — public API.

Other modules MUST import from here instead of query.internal modules.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.common.models import RetrievedNode


class RAGEnginePort(Protocol):
    """Interface: what query provides to documents and other modules."""

    async def process_document(self, nodes: list[RetrievedNode]) -> bool: ...

    async def query(self, request: Any, session_manager: Any | None = None, trace_id: str | None = None) -> Any: ...


class LLMGeneratorPort(Protocol):
    """Interface: what query provides for LLM generation."""

    async def generate(self, query: str, contexts: list[Any], **kwargs: Any) -> dict[str, Any]: ...


class HybridRetrieverPort(Protocol):
    """Interface: what query provides for hybrid retrieval."""

    bm25_retriever: Any  # BM25Retriever instance for index rebuild access

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
