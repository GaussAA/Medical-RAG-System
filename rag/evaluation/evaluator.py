"""Main RAG Evaluator Module.

Provides unified evaluation interface combining retrieval, generation, and medical safety metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

from app.models.schemas import QueryResponse

from rag.evaluation.retrieval_eval import RetrievalEvaluator, RetrievalMetrics
from rag.evaluation.generation_eval import GenerationEvaluator
from rag.evaluation.medical_safety_eval import MedicalSafetyEvaluator


@dataclass
class EvalGroundTruth:
    """Ground truth for evaluation."""

    query_id: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    difficulty: str = "medium"
    safety_sensitive: bool = False


@dataclass
class RAGEvaluationResult:
    """
    Complete RAG evaluation result.

    Aggregates retrieval, generation, and medical safety metrics.
    """

    query_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Retrieval metrics
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    retrieval_hit_rate: float = 0.0

    # Generation metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    citation_accuracy: float = 0.0
    hallucination_ratio: float = 0.0

    # Medical safety metrics
    entity_accuracy: float | None = None
    warning_coverage: dict[str, bool] | None = None
    contradiction_detected: bool = False
    safety_score: float = 0.0

    # Overall score
    overall_score: float = 0.0

    # Metadata
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "retrieval": {
                "precision_at_k": self.precision_at_k,
                "recall_at_k": self.recall_at_k,
                "ndcg_at_k": self.ndcg_at_k,
                "mrr": self.mrr,
                "hit_rate": self.retrieval_hit_rate,
            },
            "generation": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "citation_accuracy": self.citation_accuracy,
                "hallucination_ratio": self.hallucination_ratio,
            },
            "medical_safety": {
                "entity_accuracy": self.entity_accuracy,
                "warning_coverage": self.warning_coverage,
                "contradiction_detected": self.contradiction_detected,
                "safety_score": self.safety_score,
            },
            "overall_score": self.overall_score,
            "details": self.details,
        }


class RAGEvaluator:
    """
    Main RAG Evaluator class.

    Provides unified interface for evaluating RAG responses across
    retrieval, generation, and medical safety dimensions.
    """

    def __init__(self, llm_generator: Any | None = None):
        """
        Initialize RAG evaluator.

        Args:
            llm_generator: Optional LLM generator for LLM-based evaluation.
        """
        self.retrieval_evaluator = RetrievalEvaluator(k_values=[5, 10, 20])
        self.generation_evaluator = GenerationEvaluator(llm_generator=llm_generator)
        self.medical_safety_evaluator = MedicalSafetyEvaluator()

    async def evaluate(
        self,
        query: str,
        response: QueryResponse,
        ground_truth: EvalGroundTruth | None = None,
        retrieved_doc_ids: list[str] | None = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate a RAG response.

        Args:
            query: The original user query.
            response: The QueryResponse from RAG engine.
            ground_truth: Optional ground truth for evaluation.
            retrieved_doc_ids: List of retrieved document IDs in ranking order.

        Returns:
            RAGEvaluationResult with all computed metrics.
        """
        result = RAGEvaluationResult(query_id=getattr(ground_truth, "query_id", "unknown"))

        # Extract contexts from response metadata or citations
        contexts = self._extract_contexts(response)

        # Auto-extract retrieved doc IDs from response metadata if not provided
        if not retrieved_doc_ids and response.metadata:
            retrieved_doc_ids = response.metadata.get("retrieved_doc_ids") or response.metadata.get("retrieved_node_ids") or []

        # Retrieval evaluation
        retrieval_metrics = self._evaluate_retrieval(
            retrieved_doc_ids=retrieved_doc_ids or [],
            ground_truth=ground_truth,
        )
        result.precision_at_k = retrieval_metrics.precision_at_k
        result.recall_at_k = retrieval_metrics.recall_at_k
        result.ndcg_at_k = retrieval_metrics.ndcg_at_k
        result.mrr = retrieval_metrics.mrr
        result.retrieval_hit_rate = retrieval_metrics.hit_rate

        # Generation evaluation
        generation_metrics = await self.generation_evaluator.evaluate(
            query=query,
            answer=response.answer,
            contexts=contexts,
            citations=response.citations,
        )
        result.faithfulness = generation_metrics.faithfulness
        result.answer_relevancy = generation_metrics.answer_relevancy
        result.citation_accuracy = generation_metrics.citation_accuracy
        result.hallucination_ratio = generation_metrics.hallucination_ratio

        # Medical safety evaluation
        safety_metrics = await self.medical_safety_evaluator.evaluate(
            query=query,
            answer=response.answer,
            contexts=contexts,
            warnings=response.warnings,
        )
        result.entity_accuracy = safety_metrics.entity_accuracy
        result.warning_coverage = safety_metrics.warning_coverage
        result.contradiction_detected = safety_metrics.contradiction_detected
        result.safety_score = safety_metrics.safety_score

        # Calculate overall score
        result.overall_score = self._calculate_overall_score(result)

        # Add metadata
        result.details = {
            "confidence": response.confidence,
            "retrieved_chunks": response.metadata.get("retrieved_chunks", 0),
            "processing_time": response.processing_time,
            "retrieved_uuids": len(retrieved_doc_ids or []),
            "gt_uuids": len(ground_truth.relevant_doc_ids) if ground_truth else 0,
        }

        return result

    def _extract_contexts(self, response: QueryResponse) -> list[str]:
        """Extract context strings from response metadata or citations."""
        # Priority 1: retrieved_contents from metadata (set by RAGEngine for evaluation)
        if response.metadata and response.metadata.get("retrieved_contents"):
            return list(response.metadata["retrieved_contents"])

        # Priority 2: citation chunk_content
        contexts = []
        for citation in response.citations:
            chunk = getattr(citation, "chunk_content", None)
            if chunk:
                contexts.append(chunk)

        return contexts

    def _evaluate_retrieval(
        self,
        retrieved_doc_ids: list[str],
        ground_truth: EvalGroundTruth | None,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.

        Args:
            retrieved_doc_ids: Retrieved document IDs.
            ground_truth: Ground truth (if available).
            response: Query response for metadata.

        Returns:
            RetrievalMetrics with computed metrics.
        """
        # If we have ground truth with actual relevant IDs, use full evaluation
        if ground_truth and ground_truth.relevant_doc_ids and retrieved_doc_ids:
            return self.retrieval_evaluator.evaluate(
                retrieved_ids=retrieved_doc_ids,
                ground_truth_ids=ground_truth.relevant_doc_ids,
            )

        # Otherwise, use hit rate evaluation
        metrics = self.retrieval_evaluator.evaluate_without_ground_truth(
            retrieved_doc_ids,
            min_relevant=1,
        )

        # Build RetrievalMetrics object
        result = RetrievalMetrics()
        result.hit_rate = metrics["hit_rate"]
        return result

    def _calculate_overall_score(self, result: RAGEvaluationResult) -> float:
        """
        Calculate overall quality score.

        Weights:
        - Retrieval: 30%
        - Generation: 40%
        - Medical Safety: 30%
        """
        # Retrieval score (average of key metrics)
        retrieval_score = 0.0
        if result.mrr > 0:
            retrieval_score = (result.mrr + result.retrieval_hit_rate) / 2

        # Generation score
        generation_score = (
            result.faithfulness * 0.4
            + result.answer_relevancy * 0.3
            + result.citation_accuracy * 0.3
        )

        # Medical safety score
        safety_score = result.safety_score

        # Weighted overall
        overall = retrieval_score * 0.3 + generation_score * 0.4 + safety_score * 0.3

        return round(overall, 4)

    async def evaluate_batch(
        self,
        queries: list[str],
        responses: list[QueryResponse],
        ground_truths: list[EvalGroundTruth] | None = None,
        retrieved_doc_ids_list: list[list[str]] | None = None,
    ) -> list[RAGEvaluationResult]:
        """
        Evaluate a batch of RAG responses.

        Args:
            queries: List of queries.
            responses: List of responses.
            ground_truths: Optional list of ground truths.
            retrieved_doc_ids_list: Optional list of retrieved doc IDs.

        Returns:
            List of RAGEvaluationResults.
        """
        results = []
        for i, response in enumerate(responses):
            query = queries[i] if i < len(queries) else ""
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            retrieved_ids = (
                retrieved_doc_ids_list[i]
                if retrieved_doc_ids_list and i < len(retrieved_doc_ids_list)
                else None
            )

            result = await self.evaluate(
                query=query,
                response=response,
                ground_truth=ground_truth,
                retrieved_doc_ids=retrieved_ids,
            )
            results.append(result)

        return results