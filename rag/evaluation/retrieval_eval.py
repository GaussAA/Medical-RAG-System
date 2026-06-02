"""Retrieval Evaluation Module.

Provides retrieval-specific metrics:
- Precision@K
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Hit Rate
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics."""

    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate: float = 0.0


class RetrievalEvaluator:
    """Evaluates retrieval quality given ground truth."""

    def __init__(self, k_values: list[int] | None = None):
        """
        Initialize retrieval evaluator.

        Args:
            k_values: List of K values for @K metrics. Defaults to [5, 10, 20].
        """
        self.k_values = k_values or [5, 10, 20]

    def evaluate(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: list[str],
        scores: list[float] | None = None,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval results against ground truth.

        Args:
            retrieved_ids: List of retrieved document IDs in ranking order.
            ground_truth_ids: List of relevant (ground truth) document IDs.
            scores: Optional retrieval scores for NDCG calculation.

        Returns:
            RetrievalMetrics containing all computed metrics.
        """
        metrics = RetrievalMetrics()

        if not retrieved_ids or not ground_truth_ids:
            return metrics

        # Build relevance array (1 if in ground truth, 0 otherwise)
        relevance = [1 if doc_id in set(ground_truth_ids) else 0 for doc_id in retrieved_ids]

        # Calculate metrics for each K
        for k in self.k_values:
            if k > len(retrieved_ids):
                continue

            # Precision@K
            metrics.precision_at_k[k] = self._precision_at_k(relevance[:k], k)

            # Recall@K
            metrics.recall_at_k[k] = self._recall_at_k(relevance[:k], len(ground_truth_ids))

            # NDCG@K
            if scores:
                metrics.ndcg_at_k[k] = self._ndcg_at_k(retrieved_ids[:k], ground_truth_ids, scores[:k])
            else:
                metrics.ndcg_at_k[k] = self._ndcg_at_k(retrieved_ids[:k], ground_truth_ids, [1.0] * k)

        # MRR
        metrics.mrr = self._mrr(relevance)

        # Hit Rate (at least 1 relevant in top-K)
        metrics.hit_rate = 1.0 if sum(relevance[: max(self.k_values)]) > 0 else 0.0

        return metrics

    def _precision_at_k(self, relevance: list[int], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        return sum(relevance) / k

    def _recall_at_k(self, relevance: list[int], total_relevant: int) -> float:
        """Calculate Recall@K."""
        if total_relevant == 0:
            return 0.0
        return sum(relevance) / total_relevant

    def _ndcg_at_k(
        self,
        doc_ids: list[str],
        ground_truth_ids: list[str],
        scores: list[float],
    ) -> float:
        """
        Calculate NDCG@K using graded relevance.

        Args:
            doc_ids: Retrieved document IDs.
            ground_truth_ids: Ground truth relevant IDs.
            scores: Retrieval scores.

        Returns:
            NDCG@K score (0-1).
        """
        ground_truth_set = set(ground_truth_ids)

        # Compute DCG
        dcg = 0.0
        for i, doc_id in enumerate(doc_ids):
            if doc_id in ground_truth_set:
                # Use score as relevance for DCG calculation (or 1.0 if scores unavailable)
                rel = scores[i] if i < len(scores) else 1.0
                dcg += rel / self._log2(i + 2)  # i+2 because position starts at 1, +1 for 1-indexing

        # Compute IDCG (ideal case: all relevant docs at top positions)
        ideal_relevance = [1.0] * min(len(ground_truth_ids), len(doc_ids))
        idcg = sum(rel / self._log2(i + 2) for i, rel in enumerate(ideal_relevance))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _mrr(self, relevance: list[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        Returns 1/rank where r is the position of first relevant doc (0 if none).
        """
        for i, rel in enumerate(relevance):
            if rel == 1:
                return 1.0 / (i + 1)
        return 0.0

    def _log2(self, n: int) -> float:
        """Compute log2 safely."""
        import math

        return math.log2(n) if n > 0 else 0.0

    def evaluate_without_ground_truth(
        self,
        retrieved_ids: list[str],
        min_relevant: int = 1,
    ) -> dict[str, Any]:
        """
        Evaluate without ground truth - computes hit rate only.

        Args:
            retrieved_ids: Retrieved document IDs.
            min_relevant: Minimum relevant docs to count as hit.

        Returns:
            Dict with hit_rate and retrieved_count.
        """
        return {
            "hit_rate": 1.0 if len(retrieved_ids) >= min_relevant else 0.0,
            "retrieved_count": len(retrieved_ids),
        }
