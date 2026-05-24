import os
import tempfile
import pytest

from app.models.schemas import RetrievedNode
from rag.retrieval.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    def setup_method(self):
        self.retriever = BM25Retriever()

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        nodes = [
            RetrievedNode(
                node_id="1",
                content="糖尿病的诊断标准",
                score=0.0,
                metadata={},
            ),
            RetrievedNode(
                node_id="2",
                content="高血压的治疗方法",
                score=0.0,
                metadata={},
            ),
        ]

        await self.retriever.add(nodes)

        results = await self.retriever.retrieve("糖尿病", top_k=2)

        assert len(results) > 0
        assert results[0].node_id == "1"

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        results = await self.retriever.retrieve("测试", top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_delete(self):
        nodes = [
            RetrievedNode(
                node_id="1",
                content="糖尿病的诊断标准",
                score=0.0,
                metadata={},
            ),
        ]

        await self.retriever.add(nodes)
        await self.retriever.delete(["1"])

        results = await self.retriever.retrieve("糖尿病", top_k=5)
        assert len(results) == 0


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_hybrid_retriever_initialization(self):
        from rag.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever()
        assert retriever is not None
        assert retriever.vector_weight == 0.6
        assert retriever.bm25_weight == 0.4


class TestBM25Persistence:
    @pytest.mark.asyncio
    async def test_bm25_persistence_saves_tokenized_corpus(self):
        """Test that BM25 persistence includes tokenized corpus for faster reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = os.path.join(tmpdir, "bm25.json")

            # Create retriever with persistence
            retriever = BM25Retriever(persist_path=persist_path)

            # Add documents
            nodes = [
                RetrievedNode(
                    node_id="1",
                    content="糖尿病的诊断标准包括空腹血糖",
                    score=0.0,
                    metadata={},
                ),
                RetrievedNode(
                    node_id="2",
                    content="高血压患者应该低盐饮食",
                    score=0.0,
                    metadata={},
                ),
            ]
            await retriever.add(nodes)

            # Verify file was created with tokenized_corpus
            import json
            with open(persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert "documents" in data
            assert "tokenized_corpus" in data
            assert len(data["tokenized_corpus"]) == 2
            assert len(data["tokenized_corpus"][0]) > 0  # Tokenized content is not empty

    @pytest.mark.asyncio
    async def test_bm25_reload_uses_tokenized_corpus(self):
        """Test that BM25 reloads without re-tokenizing when tokenized_corpus exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = os.path.join(tmpdir, "bm25.json")

            # Create first retriever and add documents
            retriever1 = BM25Retriever(persist_path=persist_path)
            nodes = [
                RetrievedNode(
                    node_id="1",
                    content="糖尿病的诊断标准",
                    score=0.0,
                    metadata={},
                ),
            ]
            await retriever1.add(nodes)

            # Create second retriever from same persist file
            retriever2 = BM25Retriever(persist_path=persist_path)

            # Should be able to retrieve without re-adding
            results = await retriever2.retrieve("糖尿病", top_k=1)
            assert len(results) > 0
            assert results[0].node_id == "1"
