"""Tests for retrieval evaluation module."""

import pytest

from rag.evaluation.retrieval_eval import RetrievalEvaluator, RetrievalMetrics


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = RetrievalEvaluator(k_values=[5, 10, 20])

    def test_precision_at_k_with_relevant_docs(self):
        """Test Precision@K when there are relevant docs in retrieved."""
        # 3 relevant out of 5 retrieved
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            ground_truth_ids=["doc1", "doc2", "doc3"],
        )

        assert metrics.precision_at_k[5] == pytest.approx(3 / 5, rel=0.01)
        assert metrics.precision_at_k.get(10, 0.0) == 0.0  # Not enough docs retrieved

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            ground_truth_ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
        )

        # 5 relevant out of 6 total (doc1-5 are all in ground truth)
        assert metrics.recall_at_k[5] == pytest.approx(5 / 6, rel=0.01)

    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            ground_truth_ids=["doc1", "doc2", "doc3"],
            scores=[1.0, 0.9, 0.8, 0.7, 0.6],
        )

        # With scores, NDCG should be high if relevant docs are at top
        assert 0.0 <= metrics.ndcg_at_k[5] <= 1.0

    def test_mrr_first_relevant_at_top(self):
        """Test MRR when first relevant doc is at position 1."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc4", "doc2", "doc5", "doc3"],
            ground_truth_ids=["doc1"],  # doc1 is at position 1
        )

        assert metrics.mrr == pytest.approx(1.0, rel=0.01)

    def test_mrr_first_relevant_at_position_3(self):
        """Test MRR when first relevant doc is at position 3."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc4", "doc5", "doc1", "doc2", "doc3"],
            ground_truth_ids=["doc1"],  # doc1 is at position 3
        )

        assert metrics.mrr == pytest.approx(1 / 3, rel=0.01)

    def test_mrr_no_relevant_docs(self):
        """Test MRR when no relevant docs retrieved."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc4", "doc5", "doc6"],
            ground_truth_ids=["doc1", "doc2", "doc3"],
        )

        assert metrics.mrr == 0.0

    def test_hit_rate_with_hit(self):
        """Test Hit Rate when at least 1 relevant doc found."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc4", "doc5"],
            ground_truth_ids=["doc1"],
        )

        assert metrics.hit_rate == 1.0

    def test_hit_rate_without_hit(self):
        """Test Hit Rate when no relevant docs found."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc4", "doc5", "doc6"],
            ground_truth_ids=["doc1", "doc2"],
        )

        assert metrics.hit_rate == 0.0

    def test_empty_retrieved_ids(self):
        """Test with empty retrieved list."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=[],
            ground_truth_ids=["doc1", "doc2"],
        )

        assert metrics.mrr == 0.0
        assert metrics.hit_rate == 0.0

    def test_empty_ground_truth_ids(self):
        """Test with empty ground truth list."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3"],
            ground_truth_ids=[],
        )

        assert metrics.mrr == 0.0

    def test_evaluate_without_ground_truth(self):
        """Test evaluation without ground truth (hit rate only)."""
        result = self.evaluator.evaluate_without_ground_truth(
            retrieved_ids=["doc1", "doc2", "doc3"],
            min_relevant=1,
        )

        assert result["hit_rate"] == 1.0
        assert result["retrieved_count"] == 3

    def test_multiple_k_values(self):
        """Test that all configured K values are computed."""
        evaluator = RetrievalEvaluator(k_values=[3, 5, 10])

        metrics = evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10", "doc11"],
            ground_truth_ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"],
        )

        assert 3 in metrics.precision_at_k
        assert 5 in metrics.precision_at_k
        assert 10 in metrics.precision_at_k
        assert 20 not in metrics.precision_at_k  # Not configured

    def test_retrieval_metrics_dataclass(self):
        """Test RetrievalMetrics dataclass."""
        metrics = RetrievalMetrics(
            precision_at_k={5: 0.8},
            recall_at_k={5: 0.6},
            ndcg_at_k={5: 0.75},
            mrr=0.5,
            hit_rate=1.0,
        )

        assert metrics.precision_at_k[5] == 0.8
        assert metrics.mrr == 0.5
        assert metrics.hit_rate == 1.0


class TestRetrievalEvaluatorEdgeCases:
    """Edge case tests for RetrievalEvaluator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = RetrievalEvaluator(k_values=[5])

    def test_duplicate_docs_in_retrieved(self):
        """Test handling of duplicate docs in retrieved list."""
        # Duplicates shouldn't affect metrics much
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc1", "doc2", "doc2", "doc3"],
            ground_truth_ids=["doc1", "doc2"],
        )

        # Precision should still be calculated correctly
        assert 0.0 <= metrics.precision_at_k[5] <= 1.0

    def test_all_docs_relevant(self):
        """Test when all retrieved docs are relevant."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            ground_truth_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        )

        # 5/5 = 1.0 precision
        assert metrics.precision_at_k[5] == pytest.approx(1.0, rel=0.01)

    def test_subset_of_ground_truth(self):
        """Test when retrieved is subset of ground truth."""
        metrics = self.evaluator.evaluate(
            retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            ground_truth_ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8"],
        )

        # 5/5 = 1.0 precision, but 5/8 recall
        assert metrics.precision_at_k[5] == pytest.approx(1.0, rel=0.01)
        assert metrics.recall_at_k[5] == pytest.approx(5 / 8, rel=0.01)