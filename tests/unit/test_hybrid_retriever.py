import pytest

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
