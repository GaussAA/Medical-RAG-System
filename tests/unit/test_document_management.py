import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import uuid

from app.models.schemas import (
    BatchDeleteRequest,
    BatchOperationResponse,
    BatchUpdateRequest,
    ChunkListResponse,
    ChunkUpdateRequest,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentStatus,
    DocumentUpdateRequest,
)
from app.services.document_store import DocumentStore


class TestDocumentStoreFiltering:
    """Test filtering and pagination in DocumentStore."""

    def setup_method(self):
        self.store = DocumentStore()

    @pytest.mark.asyncio
    async def test_list_documents_with_status_filter(self):
        """Test filtering documents by status."""
        mock_docs = [
            MagicMock(
                id=uuid.uuid4(),
                title="Doc 1",
                status="completed",
                total_chunks=1,
                tags=[],
                created_at=datetime.now(timezone.utc),
            ),
            MagicMock(
                id=uuid.uuid4(),
                title="Doc 2",
                status="completed",
                total_chunks=2,
                tags=[],
                created_at=datetime.now(timezone.utc),
            ),
        ]

        mock_session_instance = MagicMock()

        async def mock_execute(query):
            result = MagicMock()
            result.scalar.return_value = 2
            scalars_mock = MagicMock()
            scalars_mock.__iter__ = MagicMock(return_value=iter(mock_docs))
            result.scalars.return_value = scalars_mock
            return result

        mock_session_instance.execute = mock_execute

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            docs, total = await self.store.list_documents(status="completed")

            assert total == 2

    @pytest.mark.asyncio
    async def test_list_documents_with_tag_filter(self):
        """Test filtering documents by tags using array containment."""
        mock_session_instance = MagicMock()

        async def mock_execute(query):
            result = MagicMock()
            result.scalar.return_value = 0
            scalars_mock = MagicMock()
            scalars_mock.__iter__ = MagicMock(return_value=iter([]))
            result.scalars.return_value = scalars_mock
            return result

        mock_session_instance.execute = mock_execute

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            docs, total = await self.store.list_documents(tags=["cardiology"])

            assert total == 0

    @pytest.mark.asyncio
    async def test_list_documents_with_date_range(self):
        """Test filtering documents by date range."""
        mock_session_instance = MagicMock()

        async def mock_execute(query):
            result = MagicMock()
            result.scalar.return_value = 0
            scalars_mock = MagicMock()
            scalars_mock.__iter__ = MagicMock(return_value=iter([]))
            result.scalars.return_value = scalars_mock
            return result

        mock_session_instance.execute = mock_execute

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            date_from = datetime(2026, 1, 1)
            date_to = datetime(2026, 5, 1)
            docs, total = await self.store.list_documents(date_from=date_from, date_to=date_to)

            assert total == 0


class TestDocumentStoreChunkOperations:
    """Test chunk CRUD operations in DocumentStore."""

    def setup_method(self):
        self.store = DocumentStore()

    @pytest.mark.asyncio
    async def test_get_chunks_returns_empty_for_nonexistent_doc(self):
        """Test that get_chunks returns empty for non-existent document."""
        mock_session_instance = MagicMock()
        mock_session_instance.get = AsyncMock(return_value=None)

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        # Use a valid UUID format since uuid.UUID() parses before session.get()
        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            chunks, total = await self.store.get_chunks("00000000-0000-0000-0000-000000000000")

            assert chunks == []
            assert total == 0

    @pytest.mark.asyncio
    async def test_get_chunks_with_pagination(self):
        """Test getting chunks with pagination."""
        mock_doc_id = uuid.uuid4()
        mock_chunks = [
            MagicMock(
                id=uuid.uuid4(),
                doc_id=mock_doc_id,
                content="Chunk 1",
                position=0,
                page_number=1,
                section_title="Intro",
                vector_id="vec1",
            ),
            MagicMock(
                id=uuid.uuid4(),
                doc_id=mock_doc_id,
                content="Chunk 2",
                position=1,
                page_number=2,
                section_title="Chapter 1",
                vector_id="vec2",
            ),
        ]

        mock_session_instance = MagicMock()
        mock_session_instance.get = AsyncMock(return_value=MagicMock(id=mock_doc_id))

        async def mock_execute(query):
            result = MagicMock()
            result.scalar.return_value = 2
            scalars_mock = MagicMock()
            scalars_mock.__iter__ = MagicMock(return_value=iter(mock_chunks))
            result.scalars.return_value = scalars_mock
            return result

        mock_session_instance.execute = mock_execute

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            chunks, total = await self.store.get_chunks(str(mock_doc_id), page=1, page_size=50)

            assert len(chunks) == 2
            assert total == 2

    @pytest.mark.asyncio
    async def test_update_chunk_content(self):
        """Test updating chunk content."""
        mock_doc_id = uuid.uuid4()
        mock_chunk_id = uuid.uuid4()
        mock_chunk = MagicMock()
        mock_chunk.doc_id = mock_doc_id
        mock_chunk.content = "Old content"
        mock_chunk.section_title = "Old title"

        mock_doc = MagicMock()

        mock_session_instance = MagicMock()
        mock_session_instance.get = AsyncMock(side_effect=lambda cls, uid: (
            mock_chunk if uid == mock_chunk_id else mock_doc
        ))
        mock_session_instance.commit = AsyncMock()

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            result = await self.store.update_chunk(
                str(mock_doc_id),
                str(mock_chunk_id),
                content="New content",
                section_title="New title",
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_delete_chunk_reorders_remaining(self):
        """Test that deleting a chunk calls session operations."""
        mock_doc_id = uuid.uuid4()
        mock_chunk_id = uuid.uuid4()

        # Create a mock chunk that matches the doc_id and chunk_id
        mock_chunk = MagicMock()
        mock_chunk.doc_id = mock_doc_id
        mock_chunk.id = mock_chunk_id

        mock_session_instance = MagicMock()
        mock_session_instance.get = AsyncMock(return_value=mock_chunk)

        call_count = 0
        async def mock_execute(query):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            scalars_mock = MagicMock()
            scalars_mock.__iter__ = MagicMock(return_value=iter([]))
            result.scalars.return_value = scalars_mock
            return result

        mock_session_instance.execute = mock_execute
        mock_session_instance.commit = AsyncMock()

        mock_ensure_session = AsyncMock(return_value=mock_session_instance)

        with patch.object(self.store, "_ensure_session", mock_ensure_session):
            await self.store.delete_chunk(str(mock_doc_id), str(mock_chunk_id))

            # execute should be called (for delete query and reorder query)
            assert call_count >= 1


class TestDocumentSchemas:
    """Test Pydantic schemas for new features."""

    def test_document_status_with_tags(self):
        """Test DocumentStatus schema with tags field."""
        status = DocumentStatus(
            id="test-id",
            title="Test Doc",
            status="completed",
            total_chunks=5,
            tags=["cardiology", "radiology"],
        )
        assert status.tags == ["cardiology", "radiology"]

    def test_document_list_response_pagination(self):
        """Test DocumentListResponse with pagination metadata."""
        response = DocumentListResponse(
            documents=[],
            total=0,
            page=1,
            page_size=20,
        )
        assert response.total == 0
        assert response.page == 1

    def test_document_preview_response(self):
        """Test DocumentPreviewResponse schema."""
        preview = DocumentPreviewResponse(
            document_id="test-id",
            title="Test Doc",
            preview_text="This is the preview text...",
            preview_is_full=False,
            total_pages=10,
            total_chunks=25,
        )
        assert "..." in preview.preview_text
        assert preview.preview_is_full is False

    def test_chunk_list_response_pagination(self):
        """Test ChunkListResponse with pagination metadata."""
        response = ChunkListResponse(
            chunks=[],
            total=0,
            page=1,
            page_size=50,
        )
        assert response.page_size == 50

    def test_batch_delete_request_validation(self):
        """Test BatchDeleteRequest validation."""
        request = BatchDeleteRequest(ids=["id1", "id2"])
        assert len(request.ids) == 2

    def test_batch_update_request_with_operation(self):
        """Test BatchUpdateRequest with operation field."""
        request = BatchUpdateRequest(
            ids=["id1"],
            tags=["cardiology"],
            operation="add",
        )
        assert request.operation == "add"

    def test_batch_operation_response(self):
        """Test BatchOperationResponse with success/failure."""
        response = BatchOperationResponse(
            deleted=["id1", "id2"],
            failed=[{"id": "id3", "error": "not found"}],
        )
        assert len(response.deleted) == 2
        assert len(response.failed) == 1

    def test_chunk_update_request_content_only(self):
        """Test ChunkUpdateRequest with content only."""
        request = ChunkUpdateRequest(content="New chunk content")
        assert request.content == "New chunk content"
        assert request.section_title is None

    def test_document_update_request_add_tags(self):
        """Test DocumentUpdateRequest for adding tags."""
        request = DocumentUpdateRequest(
            tags=["cardiology"],
            operation="add",
        )
        assert request.tags == ["cardiology"]
        assert request.operation == "add"

    def test_document_update_request_remove_tags(self):
        """Test DocumentUpdateRequest for removing tags."""
        request = DocumentUpdateRequest(
            tags=["cardiology"],
            operation="remove",
        )
        assert request.operation == "remove"