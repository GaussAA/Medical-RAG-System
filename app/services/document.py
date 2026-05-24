import hashlib
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.rag_engine import RAGEngine
from app.models.schemas import Chunk as SchemaChunk, DocumentStatus
from app.services.document_processor import DocumentProcessor
from app.services.document_store import DocumentStore
from app.services.retrieval_indexer import RetrievalIndexer


class DocumentService:
    """Document lifecycle management using composition of specialized classes."""

    def __init__(
        self,
        rag_engine: RAGEngine | None = None,
        processor: DocumentProcessor | None = None,
        store: DocumentStore | None = None,
        indexer: RetrievalIndexer | None = None,
        async_session: Any = None,
    ):
        self.processor = processor if processor is not None else DocumentProcessor()
        self.store = store if store is not None else DocumentStore(async_session=async_session)
        self.indexer = indexer if indexer is not None else RetrievalIndexer()
        self.rag_engine = rag_engine if rag_engine is not None else RAGEngine()
        self.documents: dict[str, dict[str, Any]] = {}
        self._owns_session = async_session is not None  # Track if WE own the session

    async def __aenter__(self) -> "DocumentService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures session cleanup."""
        await self.close()

    async def close(self) -> None:
        """Close database session if we own it."""
        if self._owns_session and self.store and self.store.async_session is not None:
            await self.store.async_session.close()
            self.store.async_session = None
            self._owns_session = False

    async def process_document(
        self,
        file_path: str | Path,
        title: str | None = None,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        file_path = Path(file_path)

        # Check for duplicate by MD5
        file_md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
        existing_doc = await self._find_document_by_md5(file_md5)
        if existing_doc:
            logger.info(f"Document with MD5 {file_md5} already exists: {existing_doc['id']}")
            existing_doc["status"] = "already_exists"
            return existing_doc

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        self.documents[doc_id] = {
            "id": doc_id,
            "title": title or file_path.stem,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix[1:],
            "status": "processing",
            "total_chunks": 0,
        }

        try:
            # Check if document already exists in DB by ID
            existing = await self.store.get_document(doc_id)
            if existing:
                logger.info(f"Document {doc_id} already exists in database, updating status")
                await self.store.update_document_status(doc_id, status="processing")
            else:
                # Create document record in PostgreSQL
                await self.store.create_document(
                    doc_id=doc_id,
                    file_path=str(file_path),
                    title=title or file_path.stem,
                    file_md5=file_md5,
                )

            # Parse document with heading tree extraction
            logger.info(f"Parsing document: {file_path}")
            parsed_doc, heading_tree = await self.processor.parse_with_headings(file_path)
            logger.info(f"Document parsed, text length: {len(parsed_doc.text_content)}, headings: {len(heading_tree)}")

            # Save heading tree to PostgreSQL
            heading_ids = await self.store.save_headings(doc_id, heading_tree)

            # Save processed text
            self.processor.save_processed_text(file_path, parsed_doc.text_content)

            # Build heading tree dict for chunking metadata
            heading_tree_dict = {}
            for h in heading_tree:
                heading_tree_dict[h["level"]] = h["title"]

            # Chunk document with heading context
            chunks = self.processor.chunk(
                parsed_doc.text_content,
                metadata={
                    "doc_id": doc_id,
                    "source_file": file_path.name,
                    "heading_tree": heading_tree_dict,
                    "tables": [t.model_dump() for t in parsed_doc.tables],
                },
            )
            logger.info(f"Document chunked into {len(chunks)} chunks")

            # Create RetrievedNode list for indexing
            retrieved_nodes = self.processor.create_retrieved_nodes(doc_id, chunks, file_path.name)

            if retrieved_nodes:
                logger.info(f"Storing {len(retrieved_nodes)} chunks to vector database")
                success = await self.rag_engine.process_document(retrieved_nodes)
                if not success:
                    logger.warning("GPU vectorization failed, falling back to CPU")
                    await self.indexer.add_documents(retrieved_nodes)
                else:
                    logger.info("Chunks stored successfully (GPU accelerated)")

            # Update document status
            self.documents[doc_id]["status"] = "completed"
            self.documents[doc_id]["total_chunks"] = len(chunks)

            # Persist to PostgreSQL (chunks with heading associations)
            await self.store.update_document_status(
                doc_id,
                status="completed",
                total_chunks=len(chunks),
            )
            await self.store.save_chunks(doc_id, chunks, heading_ids)

            return self.documents[doc_id]

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}")
            self.documents[doc_id]["status"] = "failed"
            self.documents[doc_id]["error_message"] = str(e)
            await self.store.update_document_status(doc_id, status="failed")
            raise

    async def _find_document_by_md5(self, file_md5: str) -> dict[str, Any] | None:
        """Find existing document by MD5 hash."""
        doc = await self.store.find_by_md5(file_md5)
        if doc:
            return {
                "id": str(doc.id),
                "title": doc.title,
                "file_name": doc.file_name,
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "status": doc.status,
                "total_chunks": doc.total_chunks,
                "total_pages": doc.total_pages,
            }
        return None

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.documents.get(doc_id)

    async def init_document(self, doc_id: str, file_path: str, title: str) -> None:
        """Initialize document entry in PostgreSQL."""
        await self.store.create_document(
            doc_id=doc_id,
            file_path=file_path,
            title=title,
        )

        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "file_name": Path(file_path).name,
            "file_path": file_path,
            "file_type": Path(file_path).suffix[1:],
            "status": "processing",
            "total_chunks": 0,
        }

    def get_document_status(self, doc_id: str) -> DocumentStatus | None:
        doc = self.documents.get(doc_id)
        if not doc:
            return None

        return DocumentStatus(
            id=doc["id"],
            title=doc["title"],
            status=doc["status"],
            total_chunks=doc.get("total_chunks"),
            error_message=doc.get("error_message"),
        )

    def list_documents(self) -> list[DocumentStatus]:
        return [
            DocumentStatus(
                id=doc["id"],
                title=doc["title"],
                status=doc["status"],
                total_chunks=doc.get("total_chunks"),
            )
            for doc in self.documents.values()
        ]

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from all stores atomically.

        Deletion order: PostgreSQL (source of truth) -> Qdrant -> BM25

        If PostgreSQL succeeds but Qdrant/BM25 fail, we log the inconsistency
        but do NOT rollback PostgreSQL (document is deleted, index will be
        fixed by consistency check/cleanup).

        If PostgreSQL fails, no further action is taken.
        """
        doc = self.documents.get(doc_id)
        if not doc:
            logger.warning(f"delete_document: doc {doc_id} not found in documents")
            return False

        logger.info(
            f"delete_document: found doc {doc_id} with total_chunks={doc.get('total_chunks')}"
        )

        # Handle None total_chunks - try to get from database or use 0
        total_chunks = doc.get("total_chunks")
        if total_chunks is None:
            # Try to get from database
            db_doc = await self.store.get_document(doc_id)
            if db_doc and db_doc.total_chunks:
                total_chunks = db_doc.total_chunks
                logger.info(f"Retrieved total_chunks={total_chunks} from database")
            else:
                total_chunks = 0
                logger.warning(f"total_chunks is None for doc {doc_id}, using 0")

        # STEP 1: Delete from PostgreSQL first (source of truth)
        pg_success = await self.store.delete_document(doc_id)
        if not pg_success:
            logger.error(f"Failed to delete document {doc_id} from PostgreSQL")
            return False

        logger.info(f"PostgreSQL: deleted document {doc_id}")

        # STEP 2 & 3: Delete from indexes (Qdrant then BM25)
        # These are best-effort after PostgreSQL is committed
        index_result = await self.indexer.delete_documents_atomic(doc_id, total_chunks)

        if not index_result["success"]:
            logger.error(
                f"Index deletion partial failure for doc {doc_id}: "
                f"vector_deleted={index_result['vector_deleted']}, "
                f"bm25_deleted={index_result['bm25_deleted']}, "
                f"errors={index_result['errors']}"
            )
        else:
            logger.info(f"Indexes: deleted {total_chunks} chunks from Qdrant and BM25")

        # STEP 4: Delete original file
        file_path = Path(doc["file_path"])
        try:
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {file_path}")

        # STEP 5: Delete processed file
        original_name = Path(doc["file_path"]).stem
        processed_file_path = Path(f"data/processed/{original_name}.txt")
        try:
            processed_file_path.unlink()
            logger.info(f"Deleted processed file: {processed_file_path}")
        except FileNotFoundError:
            logger.warning(f"Processed file not found for deletion: {processed_file_path}")

        # STEP 6: Remove from in-memory dict ONLY after all deletions complete
        del self.documents[doc_id]

        # STEP 7: Return status - note if there was index inconsistency
        if not index_result["success"]:
            logger.warning(
                f"Document {doc_id} deleted but index inconsistency detected. "
                f"Run consistency check to repair."
            )

        return True

    async def _load_documents_from_db(self):
        """Load documents from database on startup."""
        if self.documents:
            return

        docs, _ = await self.store.list_documents()
        for doc in docs:
            self.documents[str(doc.id)] = {
                "id": str(doc.id),
                "title": doc.title,
                "file_name": doc.file_name,
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "status": doc.status,
                "total_chunks": doc.total_chunks,
                "total_pages": doc.total_pages,
            }
        logger.info(f"Loaded {len(self.documents)} documents from database")

    def get_chunks(self, doc_id: str) -> list[SchemaChunk]:
        return []

    def get_in_memory_documents(self) -> dict[str, dict[str, Any]]:
        """Return in-memory document dict (for debug/testing)."""
        return self.documents.copy()
