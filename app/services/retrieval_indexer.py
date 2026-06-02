import uuid as uuid_lib
from typing import Any

from loguru import logger

from rag.retrieval.hybrid_retriever import HybridRetriever


class RetrievalIndexer:
    """Handles vector and BM25 index operations."""

    def __init__(self, hybrid_retriever: HybridRetriever | None = None):
        self.hybrid_retriever = hybrid_retriever or HybridRetriever()

    async def add_documents(self, nodes: list) -> bool:
        """Add documents to vector and BM25 indexes."""
        try:
            await self.hybrid_retriever.add_documents(nodes)
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            return False

    async def delete_documents(self, doc_id: str, total_chunks: int) -> bool:
        """Delete documents from vector and BM25 indexes."""
        try:
            chunk_ids = [str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, f"{doc_id}_{i}")) for i in range(total_chunks)]
            await self.hybrid_retriever.delete_documents(chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from index")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from index: {e}")
            return False

    async def delete_documents_atomic(self, doc_id: str, total_chunks: int) -> dict[str, Any]:
        """
        Delete documents from all indexes with full status reporting.

        Uses doc_id filtering for more robust deletion.

        Returns:
            {
                "success": bool,
                "vector_deleted": int,
                "bm25_deleted": int,
                "errors": list[str],
                "chunk_ids": list[str],
                "chunk_count": int
            }
        """
        # Log warning about total_chunks being ignored but still pass for backward compat
        if total_chunks is not None and total_chunks > 0:
            logger.info(
                f"delete_documents_atomic: total_chunks={total_chunks} passed but ignored, using doc_id filter instead"
            )

        result = await self.hybrid_retriever.delete_documents_atomic(doc_id)
        result["chunk_ids"] = []  # Unknown since we use filter now
        result["chunk_count"] = total_chunks or 0
        return result

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ):
        """Search across vector and BM25 indexes."""
        return await self.hybrid_retriever.search(query, top_k=top_k, filters=filters)

    async def rebuild_bm25_from_qdrant(self) -> dict[str, Any]:
        """
        Rebuild BM25 index from Qdrant payload data.

        This is useful when BM25 index is corrupted or lost but Qdrant
        still contains all the document content in payloads.

        Returns:
            {
                "success": bool,
                "documents_rebuilt": int,
                "errors": list[str]
            }
        """
        from qdrant_client import QdrantClient

        try:
            from config.settings import get_settings

            settings = get_settings()
            qdrant_client = QdrantClient(url=settings.database.qdrant.url)
        except Exception as e:
            logger.error(f"Failed to create Qdrant client: {e}")
            return {"success": False, "documents_rebuilt": 0, "errors": [str(e)]}

        errors: list[str] = []
        rebuilt_count = 0

        try:
            # Get all points from Qdrant
            scroll_result = qdrant_client.scroll(
                collection_name=settings.database.qdrant.collection,
                limit=10000,
                with_payload=True,
            )

            records = scroll_result[0] if scroll_result else []

            if not records:
                logger.info("No points found in Qdrant, nothing to rebuild")
                return {"success": True, "documents_rebuilt": 0, "errors": []}

            # Build nodes from Qdrant payload
            from app.models.schemas import RetrievedNode

            nodes: list[RetrievedNode] = []
            for point in records:
                payload = point.payload or {}
                content = payload.get("content", "")
                node_id = payload.get("node_id", str(point.id))

                if content:
                    nodes.append(
                        RetrievedNode(
                            node_id=node_id,
                            content=content,
                            score=1.0,
                            metadata=payload,
                        )
                    )

            # Clear existing BM25 and rebuild
            self.hybrid_retriever.bm25_retriever.clear()

            # Add all nodes to BM25
            for node in nodes:
                await self.hybrid_retriever.bm25_retriever.add([node])

            rebuilt_count = len(nodes)
            logger.info(f"BM25 rebuild complete: {rebuilt_count} documents")

            return {
                "success": True,
                "documents_rebuilt": rebuilt_count,
                "errors": errors,
            }

        except Exception as e:
            logger.error(f"Failed to rebuild BM25 from Qdrant: {e}")
            return {
                "success": False,
                "documents_rebuilt": rebuilt_count,
                "errors": [str(e)],
            }
