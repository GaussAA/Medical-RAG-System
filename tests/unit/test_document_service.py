import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.document import DocumentService
from app.services.document_processor import DocumentProcessor
from app.services.document_store import DocumentStore
from app.services.retrieval_indexer import RetrievalIndexer
from app.models.schemas import Chunk, ChunkMetadata, ParsedDocument


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_init_creates_chunker(self):
        assert self.processor.chunker is not None

    def test_chunk_returns_list(self):
        text = "这是糖尿病的诊断标准。"
        metadata = {"doc_id": "test", "source_file": "test.txt"}
        chunks = self.processor.chunk(text, metadata=metadata)
        assert isinstance(chunks, list)

    def test_create_retrieved_nodes(self):
        chunks = [
            Chunk(
                chunk_id="chunk-1",
                content="糖尿病诊断标准",
                metadata=ChunkMetadata(char_count=6),
            )
        ]
        nodes = self.processor.create_retrieved_nodes("doc-1", chunks, "test.txt")
        assert len(nodes) == 1
        assert nodes[0].node_id == "chunk-1"
        assert nodes[0].content == "糖尿病诊断标准"

    def test_save_processed_text(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("测试内容")
        result = self.processor.save_processed_text(test_file, "保存的文本内容")
        assert result.parent.name == "processed"
        assert result.name == "test.txt"


class TestDocumentStore:
    """Test DocumentStore class."""

    def setup_method(self):
        self.store = DocumentStore()

    def test_init(self):
        assert self.store.async_session is None

    @pytest.mark.asyncio
    async def test_ensure_session_creates_session(self):
        session = await self.store._ensure_session()
        assert session is not None

    @pytest.mark.asyncio
    async def test_find_by_md5_returns_none_for_nonexistent(self):
        result = await self.store.find_by_md5("nonexistent_md5")
        assert result is None


class TestRetrievalIndexer:
    """Test RetrievalIndexer class."""

    def setup_method(self):
        self.indexer = RetrievalIndexer()

    def test_init_creates_hybrid_retriever(self):
        assert self.indexer.hybrid_retriever is not None

    @pytest.mark.asyncio
    async def test_delete_documents_generates_correct_ids(self):
        doc_id = "test-doc-123"
        total_chunks = 3

        self.indexer.hybrid_retriever.delete_documents = AsyncMock()

        await self.indexer.delete_documents(doc_id, total_chunks)

        self.indexer.hybrid_retriever.delete_documents.assert_called_once()
        called_ids = self.indexer.hybrid_retriever.delete_documents.call_args[0][0]
        import uuid

        expected_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{i}")) for i in range(3)]
        assert called_ids == expected_ids


class TestDocumentService:
    """Test refactored DocumentService with composition."""

    def setup_method(self):
        self.service = DocumentService()

    def test_init_creates_sub_services(self):
        assert isinstance(self.service.processor, DocumentProcessor)
        assert isinstance(self.service.store, DocumentStore)
        assert isinstance(self.service.indexer, RetrievalIndexer)

    def test_get_document_returns_none_when_not_found(self):
        result = self.service.get_document("nonexistent")
        assert result is None

    def test_list_documents_empty(self):
        docs = self.service.list_documents()
        assert len(docs) == 0

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self):
        result = await self.service.delete_document("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_process_document_detects_duplicate_by_md5(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("糖尿病测试内容")

        mock_doc = MagicMock()
        mock_doc.id = "existing-doc"
        mock_doc.title = "Existing Doc"
        mock_doc.file_name = "test.txt"
        mock_doc.file_path = str(test_file)
        mock_doc.file_type = "txt"
        mock_doc.status = "completed"
        mock_doc.total_chunks = 1
        mock_doc.total_pages = 1

        self.service.store.find_by_md5 = AsyncMock(return_value=mock_doc)

        result = await self.service.process_document(str(test_file), title="Test")

        assert result["status"] == "already_exists"
        assert result["id"] == "existing-doc"

    @pytest.mark.asyncio
    async def test_process_document_success(self, tmp_path):
        test_file = tmp_path / "test.md"
        test_file.write_text("糖尿病是一种慢性代谢性疾病。", encoding="utf-8")

        mock_doc = MagicMock()
        mock_doc.id = "new-doc"
        mock_doc.title = "Test Doc"
        mock_doc.file_name = "test.md"
        mock_doc.file_path = str(test_file)
        mock_doc.file_type = "md"
        mock_doc.status = "processing"
        mock_doc.total_chunks = None
        mock_doc.total_pages = None

        self.service.store.create_document = AsyncMock(return_value=mock_doc)
        self.service.store.get_document = AsyncMock(return_value=None)  # Document doesn't exist yet
        self.service.store.update_document_status = AsyncMock()
        self.service.store.save_headings = AsyncMock(return_value={})
        self.service.store.save_chunks = AsyncMock()

        # Patch parse_document_with_headings in rag.parser module
        with patch("rag.parser.parse_document_with_headings") as mock_parse:
            mock_parse.return_value = (
                ParsedDocument(
                    text_content="糖尿病是一种慢性代谢性疾病。",
                    metadata={"page_count": 1},
                ),
                [],  # Empty heading_tree
            )

            with patch.object(self.service.processor, "chunk") as mock_chunk:
                mock_chunk.return_value = [
                    Chunk(
                        chunk_id="chunk-1",
                        content="糖尿病是一种慢性代谢性疾病。",
                        token_count=10,
                        metadata=ChunkMetadata(char_count=15),
                    )
                ]

                with patch.object(
                    self.service.rag_engine, "process_document", new_callable=AsyncMock
                ) as mock_proc:
                    mock_proc.return_value = True

                    result = await self.service.process_document(
                        str(test_file), title="糖尿病指南", doc_id="test-doc-process"
                    )

                    assert result["status"] == "completed"
                    assert result["total_chunks"] == 1

    @pytest.mark.asyncio
    async def test_delete_document_calls_sub_services(self, tmp_path):
        doc_id = "doc-to-delete"
        self.service.documents[doc_id] = {
            "id": doc_id,
            "file_path": str(tmp_path / "test.txt"),
            "total_chunks": 2,
            "title": "Test",
            "file_name": "test.txt",
            "file_type": "txt",
            "status": "completed",
        }

        # New order: store.delete_document first (PostgreSQL), then indexer.delete_documents_atomic
        self.service.store.delete_document = AsyncMock(return_value=True)
        self.service.indexer.delete_documents_atomic = AsyncMock(
            return_value={
                "success": True,
                "vector_deleted": 2,
                "bm25_deleted": 2,
                "errors": [],
                "chunk_ids": ["test-id-0", "test-id-1"],
                "chunk_count": 2,
            }
        )

        result = await self.service.delete_document(doc_id)

        assert result is True
        # PostgreSQL is called first (source of truth)
        self.service.store.delete_document.assert_called_once_with(doc_id)
        # Then indexer
        self.service.indexer.delete_documents_atomic.assert_called_once()


class TestDocumentServiceComposition:
    """Test that DocumentService properly composes sub-services."""

    def test_custom_sub_services(self):
        custom_processor = DocumentProcessor()
        custom_store = DocumentStore()
        custom_indexer = RetrievalIndexer()

        service = DocumentService(
            processor=custom_processor,
            store=custom_store,
            indexer=custom_indexer,
        )

        assert service.processor is custom_processor
        assert service.store is custom_store
        assert service.indexer is custom_indexer
