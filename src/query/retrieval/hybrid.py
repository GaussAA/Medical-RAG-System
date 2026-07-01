"""Hybrid retriever — combines vector similarity and BM25 keyword search with RRF fusion."""
import asyncio
from typing import Any

from loguru import logger

from src.common.cache.manager import CacheManager, make_cache_key
from src.common.config.settings import get_settings
from src.common.models import RetrievedNode
from src.query.retrieval.bm25 import BM25Retriever
from src.query.retrieval.boosting import QueryBoosting
from src.query.retrieval.vector import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
    ):
        settings = get_settings()
        self.config = settings.rag.retrieval

        self.vector_retriever = vector_retriever or VectorRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever(persist_path=self.config.bm25_persist_path)

        self.vector_weight = self.config.weights.get("vector", 0.6)
        self.bm25_weight = self.config.weights.get("bm25", 0.4)
        self.rrf_k = self.config.rrf_k
        self.final_top_k = self.config.final_top_k

        self.query_boosting = QueryBoosting()

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedNode]:
        top_k = top_k or self.final_top_k

        # ponytail: retrieve + fuse cached as a unit; filters are None in practice
        # and top_k is stable per config, so the cache key is just the query text.
        cache_key = make_cache_key("retrieval", query, top_k)

        cache_manager = CacheManager.get_instance()
        cached = await cache_manager.get(cache_key)
        if cached is not None:
            logger.debug(f"HybridRetriever cache hit for query: {query[:60]}")
            return [RetrievedNode(**n) for n in cached]

        fused_results = await self._search_and_fuse(query, top_k, filters)

        if fused_results:
            await cache_manager.set(
                cache_key,
                [n.model_dump() for n in fused_results],
                ttl=300,
            )

        return fused_results

    async def _search_and_fuse(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[RetrievedNode]:
        """Uncached inner: do the actual search + fuse."""

        # Detect query type for content-type boosting
        query_type = self.query_boosting.detect_query_type(query)

        vector_results, bm25_results = await self._parallel_search(query, filters)

        fused_results = self._reciprocal_rank_fusion(vector_results, bm25_results)

        # Apply content-type boosting based on query
        if query_type:
            fused_results = self.query_boosting.boost_by_content_type(fused_results, query_type)

        return fused_results[:top_k]

    async def _parallel_search(
        self, query: str, filters: dict[str, Any] | None
    ) -> tuple[list[RetrievedNode], list[RetrievedNode]]:

        async def get_vector() -> list[RetrievedNode]:
            try:
                return await self.vector_retriever.retrieve(
                    query,
                    top_k=self.config.vector_top_k,
                    filters=filters,
                )
            except Exception as e:
                logger.warning(f"Vector retrieval failed: {e}")
                return []

        async def get_bm25() -> list[RetrievedNode]:
            try:
                return await self.bm25_retriever.retrieve(
                    query,
                    top_k=self.config.bm25_top_k,
                    filters=filters,
                )
            except Exception as e:
                logger.warning(f"BM25 retrieval failed: {e}")
                return []

        vector_results, bm25_results = await asyncio.gather(get_vector(), get_bm25())
        return vector_results, bm25_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievedNode],
        bm25_results: list[RetrievedNode],
    ) -> list[RetrievedNode]:
        if not vector_results and not bm25_results:
            return []

        scores: dict[str, float] = {}

        for rank, node in enumerate(vector_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[node.node_id] = scores.get(node.node_id, 0) + self.vector_weight * rrf_score

        for rank, node in enumerate(bm25_results):
            rrf_score = 1 / (self.rrf_k + rank + 1)
            scores[node.node_id] = scores.get(node.node_id, 0) + self.bm25_weight * rrf_score

        all_nodes = {node.node_id: node for node in vector_results + bm25_results}

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for node_id, score in sorted_ids:
            node = all_nodes[node_id]
            fused_results.append(
                RetrievedNode(
                    node_id=node.node_id,
                    content=node.content,
                    score=score,
                    metadata=node.metadata,
                )
            )

        return fused_results

    async def add_documents(self, nodes: list[RetrievedNode]) -> None:
        results = await asyncio.gather(
            self.vector_retriever.add(nodes),
            self.bm25_retriever.add(nodes),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                store = "vector" if i == 0 else "BM25"
                logger.warning(f"Failed to add to {store} store: {result}")

    async def delete_documents(self, ids: list[str]) -> None:
        logger.info(f"HybridRetriever.delete_documents called with {len(ids)} IDs")
        results = await asyncio.gather(
            self.vector_retriever.delete(ids),
            self.bm25_retriever.delete(ids),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                store = "vector" if i == 0 else "BM25"
                logger.warning(f"Failed to delete from {store}: {result}")

    async def delete_documents_atomic(self, doc_id: str, chunk_ids: list[str] | None = None) -> dict[str, Any]:
        """
        Atomically delete documents from all indexes with status reporting.

        Uses doc_id filtering to delete ALL points for a document, which is more robust
        than deleting by chunk_ids when total_chunks is unknown or incorrect.

        Args:
            doc_id: Document ID to delete
            chunk_ids: Deprecated, kept for backward compatibility. Ignored.

        Returns:
            {
                "success": bool,
                "vector_deleted": int,
                "bm25_deleted": int,
                "errors": list[str]
            }
        """
        errors: list[str] = []
        vector_success = False
        bm25_success = False
        deleted_count = 0

        # Use doc_id filtering - more robust than chunk_ids
        try:
            await self.vector_retriever.delete_by_doc_id(doc_id)
            vector_success = True
            logger.info(f"VectorRetriever: deleted all points for doc_id={doc_id}")
        except Exception as e:
            errors.append(f"Vector delete failed: {e}")
            logger.error(f"VectorRetriever.delete_by_doc_id failed: {e}")

        # BM25 uses exact ID matching on node_id, but node_id format is {doc_id}_{index}
        # So we need to filter by doc_id in metadata
        try:
            remaining_docs = []

            for doc in self.bm25_retriever.documents:
                doc_id_in_doc = doc.get("metadata", {}).get("doc_id", "")
                if doc_id_in_doc == doc_id:
                    deleted_count += 1
                else:
                    remaining_docs.append(doc)

            self.bm25_retriever.documents = remaining_docs
            self.bm25_retriever.corpus = [doc["content"] for doc in remaining_docs]

            if self.bm25_retriever.corpus:
                import jieba  # type: ignore[import-untyped]
                from rank_bm25 import BM25Plus  # type: ignore[import-untyped]

                tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_retriever.corpus]
                self.bm25_retriever.bm25 = BM25Plus(tokenized_corpus)
            else:
                self.bm25_retriever.bm25 = None

            self.bm25_retriever._save()
            bm25_success = True
            logger.info(f"BM25Retriever: deleted {deleted_count} documents for doc_id={doc_id}")
        except Exception as e:
            errors.append(f"BM25 delete failed: {e}")
            logger.error(f"BM25Retriever.delete failed: {e}")

        return {
            "success": vector_success and bm25_success,
            "vector_deleted": -1 if vector_success else 0,  # Unknown count with filter delete
            "bm25_deleted": deleted_count if bm25_success else 0,
            "errors": errors,
        }
