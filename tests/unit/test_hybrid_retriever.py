import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import RetrievedNode
from rag.retrieval.hybrid_retriever import HybridRetriever


class TestHybridRetrieverQueryDetection:
    """Test HybridRetriever query type detection and boosting."""

    def setup_method(self):
        self.retriever = HybridRetriever()

    def _make_node(self, node_id: str, content: str, content_type: str = "text", score: float = 1.0) -> RetrievedNode:
        return RetrievedNode(
            node_id=node_id,
            content=content,
            score=score,
            metadata={"content_type": content_type},
        )

    def test_detect_table_query(self):
        """Query about tables should detect 'table' query type."""
        assert self.retriever.query_boosting.detect_query_type("表1的数据是什么") == "table"
        assert self.retriever.query_boosting.detect_query_type("表格中的结果") == "table"
        assert self.retriever.query_boosting.detect_query_type("表10的诊断标准") == "table"
        assert self.retriever.query_boosting.detect_query_type("table data") == "table"

    def test_detect_list_query(self):
        """Query about lists should detect 'list' query type."""
        assert self.retriever.query_boosting.detect_query_type("列出所有药物") == "list"
        assert self.retriever.query_boosting.detect_query_type("列表中的项目") == "list"
        assert self.retriever.query_boosting.detect_query_type("哪些药物在列表") == "list"

    def test_detect_drug_query(self):
        """Drug-related queries should detect 'list' content type."""
        assert self.retriever.query_boosting.detect_query_type("剂量是多少") == "list"
        assert self.retriever.query_boosting.detect_query_type("每次服用的用量") == "list"
        assert self.retriever.query_boosting.detect_query_type("每日服用次数") == "list"
        assert self.retriever.query_boosting.detect_query_type("不良反应有哪些") == "list"

    def test_detect_no_query_type(self):
        """Regular queries should return None."""
        assert self.retriever.query_boosting.detect_query_type("糖尿病是什么") is None
        assert self.retriever.query_boosting.detect_query_type("如何治疗高血压") is None

    def test_boost_by_content_type_table(self):
        """Table query should boost table content."""
        nodes = [
            self._make_node("1", "普通文本", "text", score=0.8),
            self._make_node("2", "| 表1 | 数据 |", "table", score=0.7),
            self._make_node("3", "更多文本", "text", score=0.6),
        ]

        boosted = self.retriever.query_boosting.boost_by_content_type(nodes, "table")

        assert boosted[0].node_id == "2"  # Table should be first (0.7 * 1.3 = 0.91)
        assert boosted[1].node_id in ["1", "3"]  # Others follow

    def test_boost_by_content_type_list(self):
        """List query should boost list content."""
        nodes = [
            self._make_node("1", "段落内容", "text", score=0.8),
            self._make_node("2", "- 列表项", "list", score=0.7),
            self._make_node("3", "其他内容", "text", score=0.6),
        ]

        boosted = self.retriever.query_boosting.boost_by_content_type(nodes, "list")

        assert boosted[0].node_id == "2"  # List should be first (0.7 * 1.3 = 0.91)

    def test_boost_preserves_all_nodes(self):
        """Boosting should not drop any nodes."""
        nodes = [
            self._make_node("1", "文本", "text"),
            self._make_node("2", "| 表格 |", "table"),
            self._make_node("3", "列表项", "list"),
        ]

        boosted = self.retriever.query_boosting.boost_by_content_type(nodes, "table")

        assert len(boosted) == 3

    def test_boost_with_no_matching_type(self):
        """If no node matches the type, all nodes are returned."""
        nodes = [
            self._make_node("1", "文本", "text"),
            self._make_node("2", "更多文本", "text"),
        ]

        boosted = self.retriever.query_boosting.boost_by_content_type(nodes, "table")

        assert len(boosted) == 2


class TestHybridRetrieverRRF:
    """Test HybridRetriever reciprocal rank fusion."""

    def setup_method(self):
        with patch('rag.retrieval.hybrid_retriever.get_settings') as mock_settings:
            mock_settings.return_value.rag.retrieval.bm25_persist_path = "bm25_test.json"
            mock_settings.return_value.rag.retrieval.weights = {"vector": 0.6, "bm25": 0.4}
            mock_settings.return_value.rag.retrieval.rrf_k = 60
            mock_settings.return_value.rag.retrieval.final_top_k = 10
            mock_settings.return_value.rag.retrieval.vector_top_k = 20
            mock_settings.return_value.rag.retrieval.bm25_top_k = 20
            self.retriever = HybridRetriever()
            self.retriever.vector_retriever = MagicMock()
            self.retriever.bm25_retriever = MagicMock()

    def test_rrf_empty_results(self):
        """Should return empty list when both results are empty."""
        result = self.retriever._reciprocal_rank_fusion([], [])
        assert result == []

    def test_rrf_vector_only(self):
        """Should return vector results when BM25 is empty."""
        vector = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="content2", score=0.8, metadata={}),
        ]
        result = self.retriever._reciprocal_rank_fusion(vector, [])
        assert len(result) == 2
        assert result[0].node_id == "1"

    def test_rrf_bm25_only(self):
        """Should return BM25 results when vector is empty."""
        bm25 = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="content2", score=0.8, metadata={}),
        ]
        result = self.retriever._reciprocal_rank_fusion([], bm25)
        assert len(result) == 2
        assert result[0].node_id == "1"

    def test_rrf_fuses_scores(self):
        """Should fuse scores from both retrievers using RRF."""
        vector = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="content2", score=0.8, metadata={}),
        ]
        bm25 = [
            RetrievedNode(node_id="2", content="content2", score=0.9, metadata={}),
            RetrievedNode(node_id="3", content="content3", score=0.7, metadata={}),
        ]
        result = self.retriever._reciprocal_rank_fusion(vector, bm25)

        assert len(result) == 3
        assert result[0].node_id == "2"
        result_ids = [r.node_id for r in result]
        assert "1" in result_ids
        assert "3" in result_ids

    def test_rrf_preserves_metadata(self):
        """Should preserve metadata in fused results."""
        vector = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={"doc_id": "doc-1"}),
        ]
        bm25 = []
        result = self.retriever._reciprocal_rank_fusion(vector, bm25)
        assert result[0].metadata["doc_id"] == "doc-1"


class TestHybridRetrieverParallelSearch:
    """Test HybridRetriever parallel search."""

    def setup_method(self):
        with patch('rag.retrieval.hybrid_retriever.get_settings') as mock_settings:
            mock_settings.return_value.rag.retrieval.bm25_persist_path = "bm25_test.json"
            mock_settings.return_value.rag.retrieval.weights = {"vector": 0.6, "bm25": 0.4}
            mock_settings.return_value.rag.retrieval.rrf_k = 60
            mock_settings.return_value.rag.retrieval.final_top_k = 10
            mock_settings.return_value.rag.retrieval.vector_top_k = 20
            mock_settings.return_value.rag.retrieval.bm25_top_k = 20
            self.retriever = HybridRetriever()

    @pytest.mark.asyncio
    async def test_parallel_search_returns_both_results(self):
        """Should return results from both vector and BM25."""
        vector_results = [
            RetrievedNode(node_id="v1", content="vector content", score=0.9, metadata={}),
        ]
        bm25_results = [
            RetrievedNode(node_id="b1", content="bm25 content", score=0.8, metadata={}),
        ]

        self.retriever.vector_retriever.retrieve = AsyncMock(return_value=vector_results)
        self.retriever.bm25_retriever.retrieve = AsyncMock(return_value=bm25_results)

        v, b = await self.retriever._parallel_search("test query", None)

        assert len(v) == 1
        assert v[0].node_id == "v1"
        assert len(b) == 1
        assert b[0].node_id == "b1"

    @pytest.mark.asyncio
    async def test_parallel_search_handles_vector_failure(self):
        """Should return empty list for vector when it fails."""
        bm25_results = [
            RetrievedNode(node_id="b1", content="bm25 content", score=0.8, metadata={}),
        ]

        self.retriever.vector_retriever.retrieve = AsyncMock(side_effect=Exception("Vector failed"))
        self.retriever.bm25_retriever.retrieve = AsyncMock(return_value=bm25_results)

        v, b = await self.retriever._parallel_search("test query", None)

        assert v == []
        assert len(b) == 1

    @pytest.mark.asyncio
    async def test_parallel_search_handles_bm25_failure(self):
        """Should return empty list for BM25 when it fails."""
        vector_results = [
            RetrievedNode(node_id="v1", content="vector content", score=0.9, metadata={}),
        ]

        self.retriever.vector_retriever.retrieve = AsyncMock(return_value=vector_results)
        self.retriever.bm25_retriever.retrieve = AsyncMock(side_effect=Exception("BM25 failed"))

        v, b = await self.retriever._parallel_search("test query", None)

        assert len(v) == 1
        assert b == []

    @pytest.mark.asyncio
    async def test_parallel_search_handles_both_failures(self):
        """Should return empty lists when both fail."""
        self.retriever.vector_retriever.retrieve = AsyncMock(side_effect=Exception("Vector failed"))
        self.retriever.bm25_retriever.retrieve = AsyncMock(side_effect=Exception("BM25 failed"))

        v, b = await self.retriever._parallel_search("test query", None)

        assert v == []
        assert b == []


class TestHybridRetrieverSearch:
    """Test HybridRetriever search method."""

    def setup_method(self):
        with patch('rag.retrieval.hybrid_retriever.get_settings') as mock_settings:
            mock_settings.return_value.rag.retrieval.bm25_persist_path = "bm25_test.json"
            mock_settings.return_value.rag.retrieval.weights = {"vector": 0.6, "bm25": 0.4}
            mock_settings.return_value.rag.retrieval.rrf_k = 60
            mock_settings.return_value.rag.retrieval.final_top_k = 10
            mock_settings.return_value.rag.retrieval.vector_top_k = 20
            mock_settings.return_value.rag.retrieval.bm25_top_k = 20
            self.retriever = HybridRetriever()

    @pytest.mark.asyncio
    async def test_search_returns_fused_results(self):
        """Should return fused results from both retrievers."""
        vector_results = [
            RetrievedNode(node_id="v1", content="vector content", score=0.9, metadata={}),
        ]
        bm25_results = [
            RetrievedNode(node_id="b1", content="bm25 content", score=0.8, metadata={}),
        ]

        self.retriever.vector_retriever.retrieve = AsyncMock(return_value=vector_results)
        self.retriever.bm25_retriever.retrieve = AsyncMock(return_value=bm25_results)

        results = await self.retriever.search("test query")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        """Should limit results to top_k."""
        vector_results = [
            RetrievedNode(node_id=f"v{i}", content=f"content{i}", score=0.9 - i*0.01, metadata={})
            for i in range(30)
        ]
        bm25_results = [
            RetrievedNode(node_id=f"b{i}", content=f"bm25{i}", score=0.8 - i*0.01, metadata={})
            for i in range(30)
        ]

        self.retriever.vector_retriever.retrieve = AsyncMock(return_value=vector_results)
        self.retriever.bm25_retriever.retrieve = AsyncMock(return_value=bm25_results)

        results = await self.retriever.search("test query", top_k=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_search_applies_content_type_boosting(self):
        """Should boost content type based on query."""
        vector_results = [
            RetrievedNode(node_id="text1", content="普通文本", score=0.8, metadata={"content_type": "text"}),
            RetrievedNode(node_id="table1", content="| 表1 |", score=0.7, metadata={"content_type": "table"}),
        ]
        bm25_results = []

        self.retriever.vector_retriever.retrieve = AsyncMock(return_value=vector_results)
        self.retriever.bm25_retriever.retrieve = AsyncMock(return_value=bm25_results)

        results = await self.retriever.search("表1的数据是什么")

        assert results[0].node_id == "table1"

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Should pass filters to both retrievers."""
        vector_results = [RetrievedNode(node_id="1", content="content", score=0.9, metadata={})]
        bm25_results = [RetrievedNode(node_id="1", content="content", score=0.8, metadata={})]

        self.retriever.vector_retriever.retrieve = AsyncMock(return_value=vector_results)
        self.retriever.bm25_retriever.retrieve = AsyncMock(return_value=bm25_results)

        filters = {"doc_id": "doc-1"}
        await self.retriever.search("test query", filters=filters)

        self.retriever.vector_retriever.retrieve.assert_called_once()
        self.retriever.bm25_retriever.retrieve.assert_called_once()
        assert self.retriever.vector_retriever.retrieve.call_args.kwargs.get("filters") == filters


class TestHybridRetrieverAddDocuments:
    """Test HybridRetriever add_documents."""

    def setup_method(self):
        with patch('rag.retrieval.hybrid_retriever.get_settings') as mock_settings:
            mock_settings.return_value.rag.retrieval.bm25_persist_path = "bm25_test.json"
            mock_settings.return_value.rag.retrieval.weights = {"vector": 0.6, "bm25": 0.4}
            mock_settings.return_value.rag.retrieval.rrf_k = 60
            mock_settings.return_value.rag.retrieval.final_top_k = 10
            mock_settings.return_value.rag.retrieval.vector_top_k = 20
            mock_settings.return_value.rag.retrieval.bm25_top_k = 20
            self.retriever = HybridRetriever()

    @pytest.mark.asyncio
    async def test_add_documents_to_both_retrievers(self):
        """Should add documents to both vector and BM25."""
        nodes = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="content2", score=0.8, metadata={}),
        ]

        self.retriever.vector_retriever.add = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.add = AsyncMock(return_value=None)

        await self.retriever.add_documents(nodes)

        self.retriever.vector_retriever.add.assert_called_once_with(nodes)
        self.retriever.bm25_retriever.add.assert_called_once_with(nodes)

    @pytest.mark.asyncio
    async def test_add_documents_handles_partial_failure(self):
        """Should continue even if one retriever fails."""
        nodes = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
        ]

        self.retriever.vector_retriever.add = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.add = AsyncMock(side_effect=Exception("BM25 failed"))

        await self.retriever.add_documents(nodes)


class TestHybridRetrieverDeleteDocuments:
    """Test HybridRetriever delete_documents."""

    def setup_method(self):
        with patch('rag.retrieval.hybrid_retriever.get_settings') as mock_settings:
            mock_settings.return_value.rag.retrieval.bm25_persist_path = "bm25_test.json"
            mock_settings.return_value.rag.retrieval.weights = {"vector": 0.6, "bm25": 0.4}
            mock_settings.return_value.rag.retrieval.rrf_k = 60
            mock_settings.return_value.rag.retrieval.final_top_k = 10
            mock_settings.return_value.rag.retrieval.vector_top_k = 20
            mock_settings.return_value.rag.retrieval.bm25_top_k = 20
            self.retriever = HybridRetriever()

    @pytest.mark.asyncio
    async def test_delete_documents_from_both_retrievers(self):
        """Should delete from both vector and BM25."""
        ids = ["node1", "node2"]

        self.retriever.vector_retriever.delete = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.delete = AsyncMock(return_value=None)

        await self.retriever.delete_documents(ids)

        self.retriever.vector_retriever.delete.assert_called_once_with(ids)
        self.retriever.bm25_retriever.delete.assert_called_once_with(ids)

    @pytest.mark.asyncio
    async def test_delete_documents_handles_partial_failure(self):
        """Should continue even if one retriever fails."""
        ids = ["node1"]

        self.retriever.vector_retriever.delete = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.delete = AsyncMock(side_effect=Exception("BM25 failed"))

        await self.retriever.delete_documents(ids)


class TestHybridRetrieverDeleteDocumentsAtomic:
    """Test HybridRetriever delete_documents_atomic."""

    def setup_method(self):
        with patch('rag.retrieval.hybrid_retriever.get_settings') as mock_settings:
            mock_settings.return_value.rag.retrieval.bm25_persist_path = "bm25_test.json"
            mock_settings.return_value.rag.retrieval.weights = {"vector": 0.6, "bm25": 0.4}
            mock_settings.return_value.rag.retrieval.rrf_k = 60
            mock_settings.return_value.rag.retrieval.final_top_k = 10
            mock_settings.return_value.rag.retrieval.vector_top_k = 20
            mock_settings.return_value.rag.retrieval.bm25_top_k = 20
            self.retriever = HybridRetriever()

    @pytest.mark.asyncio
    async def test_delete_documents_atomic_success(self):
        """Should delete from both and return success."""
        self.retriever.vector_retriever.delete_by_doc_id = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.documents = []
        self.retriever.bm25_retriever.corpus = []
        self.retriever.bm25_retriever._save = MagicMock()

        result = await self.retriever.delete_documents_atomic("doc-1")

        assert result["success"] is True
        self.retriever.vector_retriever.delete_by_doc_id.assert_called_once_with("doc-1")

    @pytest.mark.asyncio
    async def test_delete_documents_atomic_vector_failure(self):
        """Should return failure when vector delete fails."""
        self.retriever.vector_retriever.delete_by_doc_id = AsyncMock(
            side_effect=Exception("Vector failed")
        )
        self.retriever.bm25_retriever.documents = []
        self.retriever.bm25_retriever.corpus = []
        self.retriever.bm25_retriever._save = MagicMock()

        result = await self.retriever.delete_documents_atomic("doc-1")

        assert result["success"] is False
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_delete_documents_atomic_bm25_success(self):
        """Should properly filter and update BM25 index."""
        self.retriever.vector_retriever.delete_by_doc_id = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.documents = [
            {"content": "doc1 content", "metadata": {"doc_id": "doc-1"}},
            {"content": "doc2 content", "metadata": {"doc_id": "doc-2"}},
        ]
        self.retriever.bm25_retriever.corpus = ["doc1 content", "doc2 content"]
        self.retriever.bm25_retriever._save = MagicMock()
        self.retriever.bm25_retriever.bm25 = MagicMock()

        result = await self.retriever.delete_documents_atomic("doc-1")

        assert result["success"] is True
        assert result["bm25_deleted"] == 1

    @pytest.mark.asyncio
    async def test_delete_documents_atomic_empty_bm25(self):
        """Should handle empty BM25 corpus."""
        self.retriever.vector_retriever.delete_by_doc_id = AsyncMock(return_value=None)
        self.retriever.bm25_retriever.documents = []
        self.retriever.bm25_retriever.corpus = []
        self.retriever.bm25_retriever._save = MagicMock()
        self.retriever.bm25_retriever.bm25 = MagicMock()

        result = await self.retriever.delete_documents_atomic("doc-1")

        assert result["success"] is True
        assert result["bm25_deleted"] == 0


class TestBM25RetrieverFilters:
    """Test BM25Retriever filter support."""

    def setup_method(self):
        from rag.retrieval.bm25_retriever import BM25Retriever
        self.retriever = BM25Retriever()

    @pytest.mark.asyncio
    async def test_retrieve_with_content_type_filter(self):
        """Should filter by content_type."""
        from app.models.schemas import RetrievedNode

        nodes = [
            RetrievedNode(
                node_id="1",
                content="糖尿病的诊断标准",
                score=0.0,
                metadata={"content_type": "text"},
            ),
            RetrievedNode(
                node_id="2",
                content="| 指标 | 数值 |",
                score=0.0,
                metadata={"content_type": "table"},
            ),
        ]

        await self.retriever.add(nodes)

        results = await self.retriever.retrieve(
            "糖尿病",
            top_k=5,
            filters={"content_type": "table"}
        )

        assert len(results) == 1
        assert results[0].node_id == "2"

    @pytest.mark.asyncio
    async def test_retrieve_with_doc_id_filter(self):
        """Should filter by doc_id."""
        from app.models.schemas import RetrievedNode

        nodes = [
            RetrievedNode(
                node_id="1",
                content="文档1的内容",
                score=0.0,
                metadata={"doc_id": "doc-1"},
            ),
            RetrievedNode(
                node_id="2",
                content="文档2的内容",
                score=0.0,
                metadata={"doc_id": "doc-2"},
            ),
        ]

        await self.retriever.add(nodes)

        results = await self.retriever.retrieve(
            "内容",
            top_k=5,
            filters={"doc_id": "doc-1"}
        )

        assert all(r.metadata.get("doc_id") == "doc-1" for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_heading_id_filter(self):
        """Should filter by heading_id."""
        from app.models.schemas import RetrievedNode

        nodes = [
            RetrievedNode(
                node_id="1",
                content="第一节的内容",
                score=0.0,
                metadata={"heading_id": "heading-1"},
            ),
            RetrievedNode(
                node_id="2",
                content="第二节的内容",
                score=0.0,
                metadata={"heading_id": "heading-2"},
            ),
        ]

        await self.retriever.add(nodes)

        results = await self.retriever.retrieve(
            "内容",
            top_k=5,
            filters={"heading_id": "heading-1"}
        )

        assert all(r.metadata.get("heading_id") == "heading-1" for r in results)
