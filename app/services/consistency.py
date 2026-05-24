import uuid as uuid_lib
from typing import Any

from loguru import logger

from app.core.database import get_session_factory
from app.models.database import Document
from app.models.schemas import (
    ConsistencyCheckItem,
    ConsistencyCheckResponse,
    OrphanCleanupResponse,
)
from app.services.document_store import DocumentStore
from config.settings import get_settings
from rag.retrieval.hybrid_retriever import HybridRetriever


class ConsistencyChecker:
    """Service to check and repair consistency across all three stores."""

    def __init__(self):
        self.store = DocumentStore()
        self.hybrid_retriever = HybridRetriever()

    async def check_all_consistency(self, repair: bool = False) -> ConsistencyCheckResponse:
        """
        Check consistency for all documents.

        Compares:
        - PostgreSQL: documents table + chunks table
        - Qdrant: counts via vector_retriever (filtered by doc_id)
        - BM25: counts via bm25_retriever documents list
        """
        details: list[ConsistencyCheckItem] = []
        consistent_count = 0
        inconsistent_count = 0
        repair_actions: list[dict[str, Any]] = []

        factory = get_session_factory()
        async with factory() as session:
            from sqlalchemy import select

            result = await session.execute(select(Document).order_by(Document.created_at.desc()))
            docs = list(result.scalars())

        total = len(docs)

        for doc in docs:
            doc_id = str(doc.id)
            total_chunks = doc.total_chunks or 0

            expected_chunk_ids = set(
                str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, f"{doc_id}_{i}"))
                for i in range(total_chunks)
            )

            _pg_chunks_result, pg_chunk_count = await self.store.get_chunks(
                doc_id, page=1, page_size=10000
            )

            qdrant_count = await self._count_in_qdrant(doc_id)
            bm25_count = self._count_in_bm25(doc_id)

            issues: list[str] = []
            in_pg = True
            in_qdrant = qdrant_count > 0
            in_bm25 = bm25_count > 0

            if total_chunks == 0:
                if not in_qdrant and not in_bm25:
                    status = "consistent"
                    consistent_count += 1
                else:
                    status = "partial"
                    inconsistent_count += 1
                    if in_qdrant:
                        issues.append(f"Qdrant has {qdrant_count} chunks but PG has 0")
                    if in_bm25:
                        issues.append(f"BM25 has {bm25_count} chunks but PG has 0")
            elif in_qdrant and in_bm25 and pg_chunk_count == total_chunks:
                status = "consistent"
                consistent_count += 1
            elif not in_qdrant and not in_bm25 and pg_chunk_count == 0:
                status = "consistent"
                consistent_count += 1
            else:
                inconsistent_count += 1
                if in_qdrant and not in_bm25:
                    status = "partial"
                    issues.append(f"Qdrant has {qdrant_count} chunks but BM25 has {bm25_count}")
                elif in_bm25 and not in_qdrant:
                    status = "partial"
                    issues.append(f"BM25 has {bm25_count} chunks but Qdrant has {qdrant_count}")
                elif not in_qdrant and not in_bm25 and pg_chunk_count > 0:
                    status = "orphaned_in_db"
                    issues.append(f"PostgreSQL has {pg_chunk_count} chunks but no index entries")
                elif in_qdrant and in_bm25 and pg_chunk_count != total_chunks:
                    status = "partial"
                    issues.append(
                        f"Chunk counts mismatch: PG={pg_chunk_count}, Qdrant={qdrant_count}, BM25={bm25_count}"
                    )
                else:
                    status = "partial"

            details.append(
                ConsistencyCheckItem(
                    doc_id=doc_id,
                    in_postgresql=in_pg,
                    in_qdrant=in_qdrant,
                    in_bm25=in_bm25,
                    pg_chunk_count=pg_chunk_count,
                    qdrant_chunk_count=qdrant_count,
                    bm25_chunk_count=bm25_count,
                    status=status,
                    issues=issues,
                )
            )

            if repair and status != "consistent":
                if in_qdrant and not in_bm25:
                    repair_actions.append(
                        {
                            "action": "delete_from_qdrant",
                            "doc_id": doc_id,
                            "chunk_ids": list(expected_chunk_ids),
                            "reason": "Orphaned in Qdrant (no BM25)",
                        }
                    )
                elif in_bm25 and not in_qdrant:
                    repair_actions.append(
                        {
                            "action": "delete_from_bm25",
                            "doc_id": doc_id,
                            "chunk_ids": list(expected_chunk_ids),
                            "reason": "Orphaned in BM25 (no Qdrant)",
                        }
                    )
                elif not in_qdrant and not in_bm25 and pg_chunk_count > 0:
                    repair_actions.append(
                        {
                            "action": "delete_from_postgresql",
                            "doc_id": doc_id,
                            "reason": "No index entries but PG has chunks",
                        }
                    )

        return ConsistencyCheckResponse(
            total_documents=total,
            consistent_count=consistent_count,
            inconsistent_count=inconsistent_count,
            details=details,
            repair_actions=repair_actions,
        )

    async def cleanup_orphans(self) -> OrphanCleanupResponse:
        """
        Remove orphaned entries from Qdrant and BM25.

        An orphan is an entry in Qdrant/BM25 that doesn't have a corresponding
        document in PostgreSQL.
        """
        cleaned_qdrant = 0
        cleaned_bm25 = 0
        errors: list[dict[str, str]] = []

        factory = get_session_factory()
        async with factory() as session:
            from sqlalchemy import select

            result = await session.execute(select(Document.id))
            pg_doc_ids = {str(row[0]) for row in result.fetchall()}

        logger.info(f"PostgreSQL has {len(pg_doc_ids)} documents")

        # Clean BM25 orphans
        bm25_orphan_ids: list[str] = []
        for doc_data in self.hybrid_retriever.bm25_retriever.documents:
            doc_id = doc_data.get("metadata", {}).get("doc_id")
            if doc_id and doc_id not in pg_doc_ids:
                doc_id_value = doc_data.get("id")
                if doc_id_value:
                    bm25_orphan_ids.append(doc_id_value)

        if bm25_orphan_ids:
            try:
                await self.hybrid_retriever.bm25_retriever.delete(bm25_orphan_ids)
                cleaned_bm25 = len(bm25_orphan_ids)
                logger.info(f"Cleaned {cleaned_bm25} orphaned entries from BM25")
            except Exception as e:
                errors.append({"store": "bm25", "error": str(e)})
                logger.error(f"Failed to clean BM25 orphans: {e}")

        # Clean Qdrant orphans - need to scan all points and filter by doc_id not in PG
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import PointIdsList

            settings = get_settings()
            qdrant_client = QdrantClient(url=settings.database.qdrant.url)

            # Get all points to find orphaned doc_ids
            # Use scroll API to get all points (returns tuple: (records, offset))
            scroll_result = qdrant_client.scroll(
                collection_name=settings.database.qdrant.collection,
                limit=10000,
                with_payload=True,
            )

            orphaned_point_ids: list[str] = []
            records = scroll_result[0] if scroll_result else []
            for point in records:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if doc_id and doc_id not in pg_doc_ids:
                    orphaned_point_ids.append(str(point.id))

            if orphaned_point_ids:
                qdrant_client.delete(
                    collection_name=settings.database.qdrant.collection,
                    points_selector=PointIdsList(points=orphaned_point_ids),
                )
                cleaned_qdrant = len(orphaned_point_ids)
                logger.info(f"Cleaned {cleaned_qdrant} orphaned entries from Qdrant")

        except Exception as e:
            errors.append({"store": "qdrant", "error": str(e)})
            logger.error(f"Failed to clean Qdrant orphans: {e}")

        return OrphanCleanupResponse(
            cleaned_from_qdrant=cleaned_qdrant,
            cleaned_from_bm25=cleaned_bm25,
            errors=errors,
        )

    async def _count_in_qdrant(self, doc_id: str) -> int:
        """Count chunks in Qdrant for a specific doc_id."""
        try:
            results = await self.hybrid_retriever.vector_retriever.retrieve(
                "",
                top_k=10000,
                filters={"doc_id": doc_id},
            )
            return len(results)
        except Exception as e:
            logger.warning(f"Failed to count Qdrant entries for {doc_id}: {e}")
            return 0

    def _count_in_bm25(self, doc_id: str) -> int:
        """Count chunks in BM25 for a specific doc_id."""
        count = 0
        for doc_data in self.hybrid_retriever.bm25_retriever.documents:
            if doc_data.get("metadata", {}).get("doc_id") == doc_id:
                count += 1
        return count
