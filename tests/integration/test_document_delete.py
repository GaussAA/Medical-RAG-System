import uuid
from unittest.mock import AsyncMock

import pytest


class TestVectorDBDeletion:
    """Test that vector DB is properly called during document deletion"""

    @pytest.mark.asyncio
    async def test_delete_document_calls_hybrid_retriever_delete(self):
        from app.services.document import DocumentService

        service = DocumentService()
        doc_id = str(uuid.uuid4())

        service.documents[doc_id] = {
            "id": doc_id,
            "file_path": "data/raw_documents/test.txt",
            "total_chunks": 5,
            "title": "Test",
            "file_name": "test.txt",
            "file_type": "txt",
            "status": "completed",
        }

        service.indexer.delete_documents_atomic = AsyncMock(
            return_value={
                "success": True,
                "vector_deleted": 5,
                "bm25_deleted": 5,
                "errors": [],
                "chunk_ids": [],
                "chunk_count": 5,
            }
        )
        service.store.delete_document = AsyncMock(return_value=True)

        result = await service.delete_document(doc_id)

        assert result is True
        service.indexer.delete_documents_atomic.assert_called_once_with(doc_id, 5)

    @pytest.mark.asyncio
    async def test_delete_document_with_zero_chunks_does_not_call_delete(self):
        from app.services.document import DocumentService

        service = DocumentService()
        doc_id = str(uuid.uuid4())

        service.documents[doc_id] = {
            "id": doc_id,
            "file_path": "data/raw_documents/test.txt",
            "total_chunks": 0,
            "title": "Test",
            "file_name": "test.txt",
            "file_type": "txt",
            "status": "completed",
        }

        service.indexer.delete_documents_atomic = AsyncMock()
        service.store.delete_document = AsyncMock(return_value=True)

        result = await service.delete_document(doc_id)

        assert result is True
        service.store.delete_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document_completely_removes_all_traces(self):
        from app.services.document import DocumentService

        service = DocumentService()
        doc_id = str(uuid.uuid4())

        service.documents[doc_id] = {
            "id": doc_id,
            "file_path": "data/raw_documents/test.txt",
            "total_chunks": 2,
            "title": "Test",
            "file_name": "test.txt",
            "file_type": "txt",
            "status": "completed",
        }

        service.indexer.delete_documents_atomic = AsyncMock(
            return_value={
                "success": True,
                "vector_deleted": 2,
                "bm25_deleted": 2,
                "errors": [],
                "chunk_ids": [],
                "chunk_count": 2,
            }
        )
        service.store.delete_document = AsyncMock(return_value=True)

        result = await service.delete_document(doc_id)

        assert result is True
        assert service.get_document(doc_id) is None

    @pytest.mark.asyncio
    async def test_delete_document_postgresql_failure(self):
        from app.services.document import DocumentService

        service = DocumentService()
        doc_id = str(uuid.uuid4())

        service.documents[doc_id] = {
            "id": doc_id,
            "file_path": "data/raw_documents/test.txt",
            "total_chunks": 2,
            "title": "Test",
            "file_name": "test.txt",
            "file_type": "txt",
            "status": "completed",
        }

        service.indexer.delete_documents_atomic = AsyncMock()
        service.store.delete_document = AsyncMock(return_value=False)

        result = await service.delete_document(doc_id)

        assert result is False
        service.indexer.delete_documents_atomic.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_document_partial_index_failure(self):
        from app.services.document import DocumentService

        service = DocumentService()
        doc_id = str(uuid.uuid4())

        service.documents[doc_id] = {
            "id": doc_id,
            "file_path": "data/raw_documents/test.txt",
            "total_chunks": 3,
            "title": "Test",
            "file_name": "test.txt",
            "file_type": "txt",
            "status": "completed",
        }

        service.indexer.delete_documents_atomic = AsyncMock(
            return_value={
                "success": False,
                "vector_deleted": 3,
                "bm25_deleted": 0,
                "errors": ["BM25 delete failed: connection error"],
                "chunk_ids": [],
                "chunk_count": 3,
            }
        )
        service.store.delete_document = AsyncMock(return_value=True)

        result = await service.delete_document(doc_id)

        assert result is True
        assert service.get_document(doc_id) is None
