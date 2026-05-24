"""Evaluation Reporter Module.

Provides reporting capabilities for RAG evaluation results.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rag.evaluation.evaluator import RAGEvaluationResult
from rag.evaluation.benchmark_runner import BenchmarkResult


@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""

    total_queries: int
    successful_queries: int
    failed_queries: int

    # Retrieval summary
    avg_precision_at_5: float
    avg_recall_at_5: float
    avg_ndcg_at_5: float
    avg_mrr: float
    avg_hit_rate: float

    # Generation summary
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_citation_accuracy: float
    avg_hallucination_ratio: float

    # Safety summary
    avg_safety_score: float

    # Overall
    avg_overall_score: float
    min_overall_score: float
    max_overall_score: float


class EvaluationReporter:
    """
    Reporter for generating evaluation reports.

    Supports JSON and summary text formats.
    """

    def __init__(self):
        """Initialize evaluation reporter."""
        pass

    def generate_json(
        self,
        results: list[RAGEvaluationResult] | BenchmarkResult,
    ) -> dict[str, Any]:
        """
        Generate JSON report from evaluation results.

        Args:
            results: Either a list of RAGEvaluationResult or a BenchmarkResult.

        Returns:
            Dictionary representation of the report.
        """
        if isinstance(results, BenchmarkResult):
            return results.to_dict()

        # Convert list of results
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(results),
            "results": [r.to_dict() for r in results],
            "summary": self._generate_summary_dict(results),
        }

    def generate_summary(
        self,
        results: list[RAGEvaluationResult] | BenchmarkResult,
    ) -> str:
        """
        Generate human-readable summary text.

        Args:
            results: Either a list of RAGEvaluationResult or a BenchmarkResult.

        Returns:
            Summary text string.
        """
        if isinstance(results, BenchmarkResult):
            return self._format_benchmark_summary(results)

        summary = self._compute_summary(results)
        return self._format_summary_text(summary)

    def generate_summary_dict(
        self,
        results: list[RAGEvaluationResult] | BenchmarkResult,
    ) -> EvaluationSummary:
        """
        Generate summary dataclass from results.

        Args:
            results: Either a list of RAGEvaluationResult or a BenchmarkResult.

        Returns:
            EvaluationSummary dataclass.
        """
        if isinstance(results, BenchmarkResult):
            return self._benchmark_to_summary(results)

        return self._compute_summary(results)

    def _compute_summary(
        self,
        results: list[RAGEvaluationResult],
    ) -> EvaluationSummary:
        """Compute summary statistics from results."""
        if not results:
            return EvaluationSummary(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                avg_precision_at_5=0.0,
                avg_recall_at_5=0.0,
                avg_ndcg_at_5=0.0,
                avg_mrr=0.0,
                avg_hit_rate=0.0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_citation_accuracy=0.0,
                avg_hallucination_ratio=0.0,
                avg_safety_score=0.0,
                avg_overall_score=0.0,
                min_overall_score=0.0,
                max_overall_score=0.0,
            )

        n = len(results)
        overall_scores = [r.overall_score for r in results]

        # Retrieval metrics
        p5_values = [r.precision_at_k.get(5, 0.0) for r in results if r.precision_at_k]
        r5_values = [r.recall_at_k.get(5, 0.0) for r in results if r.recall_at_k]
        n5_values = [r.ndcg_at_k.get(5, 0.0) for r in results if r.ndcg_at_k]

        return EvaluationSummary(
            total_queries=n,
            successful_queries=n,
            failed_queries=0,
            avg_precision_at_5=sum(p5_values) / len(p5_values) if p5_values else 0.0,
            avg_recall_at_5=sum(r5_values) / len(r5_values) if r5_values else 0.0,
            avg_ndcg_at_5=sum(n5_values) / len(n5_values) if n5_values else 0.0,
            avg_mrr=sum(r.mrr for r in results) / n,
            avg_hit_rate=sum(r.retrieval_hit_rate for r in results) / n,
            avg_faithfulness=sum(r.faithfulness for r in results) / n,
            avg_answer_relevancy=sum(r.answer_relevancy for r in results) / n,
            avg_citation_accuracy=sum(r.citation_accuracy for r in results) / n,
            avg_hallucination_ratio=sum(r.hallucination_ratio for r in results) / n,
            avg_safety_score=sum(r.safety_score for r in results if r.safety_score > 0) / n if any(r.safety_score > 0 for r in results) else 0.0,
            avg_overall_score=sum(overall_scores) / n,
            min_overall_score=min(overall_scores),
            max_overall_score=max(overall_scores),
        )

    def _generate_summary_dict(
        self,
        results: list[RAGEvaluationResult],
    ) -> dict[str, Any]:
        """Generate summary dictionary."""
        summary = self._compute_summary(results)
        return {
            "total_queries": summary.total_queries,
            "retrieval": {
                "precision@5": round(summary.avg_precision_at_5, 4),
                "recall@5": round(summary.avg_recall_at_5, 4),
                "ndcg@5": round(summary.avg_ndcg_at_5, 4),
                "mrr": round(summary.avg_mrr, 4),
                "hit_rate": round(summary.avg_hit_rate, 4),
            },
            "generation": {
                "faithfulness": round(summary.avg_faithfulness, 4),
                "answer_relevancy": round(summary.avg_answer_relevancy, 4),
                "citation_accuracy": round(summary.avg_citation_accuracy, 4),
                "hallucination_ratio": round(summary.avg_hallucination_ratio, 4),
            },
            "safety": {
                "avg_safety_score": round(summary.avg_safety_score, 4),
            },
            "overall": {
                "avg_score": round(summary.avg_overall_score, 4),
                "min_score": round(summary.min_overall_score, 4),
                "max_score": round(summary.max_overall_score, 4),
            },
        }

    def _format_summary_text(self, summary: EvaluationSummary) -> str:
        """Format summary as human-readable text."""
        lines = [
            "=" * 60,
            "RAG Evaluation Summary",
            "=" * 60,
            f"Total Queries: {summary.total_queries}",
            f"Successful: {summary.successful_queries}",
            f"Failed: {summary.failed_queries}",
            "",
            "Retrieval Metrics:",
            f"  Precision@5:  {summary.avg_precision_at_5:.4f}",
            f"  Recall@5:     {summary.avg_recall_at_5:.4f}",
            f"  NDCG@5:       {summary.avg_ndcg_at_5:.4f}",
            f"  MRR:          {summary.avg_mrr:.4f}",
            f"  Hit Rate:     {summary.avg_hit_rate:.4f}",
            "",
            "Generation Metrics:",
            f"  Faithfulness:      {summary.avg_faithfulness:.4f}",
            f"  Answer Relevancy:  {summary.avg_answer_relevancy:.4f}",
            f"  Citation Accuracy: {summary.avg_citation_accuracy:.4f}",
            f"  Hallucination Ratio: {summary.avg_hallucination_ratio:.4f}",
            "",
            "Safety:",
            f"  Safety Score: {summary.avg_safety_score:.4f}",
            "",
            "Overall Score:",
            f"  Average: {summary.avg_overall_score:.4f}",
            f"  Min: {summary.min_overall_score:.4f}",
            f"  Max: {summary.max_overall_score:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _format_benchmark_summary(self, result: BenchmarkResult) -> str:
        """Format BenchmarkResult as text summary."""
        lines = [
            "=" * 60,
            f"Benchmark: {result.benchmark_id}",
            f"Dataset: {result.dataset_name}",
            f"Timestamp: {result.timestamp}",
            "=" * 60,
            f"Total Queries: {result.total_queries}",
            f"Successful: {result.successful_evaluations}",
            f"Failed: {result.failed_evaluations}",
            "",
            "Aggregated Metrics:",
            f"  Precision@5:  {result.avg_precision_at_5:.4f}",
            f"  Recall@5:     {result.avg_recall_at_5:.4f}",
            f"  NDCG@5:       {result.avg_ndcg_at_5:.4f}",
            f"  MRR:          {result.avg_mrr:.4f}",
            f"  Hit Rate:     {result.avg_hit_rate:.4f}",
            "",
            f"  Faithfulness:      {result.avg_faithfulness:.4f}",
            f"  Answer Relevancy:  {result.avg_answer_relevancy:.4f}",
            f"  Citation Accuracy: {result.avg_citation_accuracy:.4f}",
            f"  Hallucination Ratio: {result.avg_hallucination_ratio:.4f}",
            "",
            f"  Safety Score: {result.avg_safety_score:.4f}",
            "",
            "Overall Score:",
            f"  Average: {result.avg_overall_score:.4f}",
            f"  Min: {result.min_overall_score:.4f}",
            f"  Max: {result.max_overall_score:.4f}",
            f"  Std: {result.std_overall_score:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _benchmark_to_summary(self, result: BenchmarkResult) -> EvaluationSummary:
        """Convert BenchmarkResult to EvaluationSummary."""
        return EvaluationSummary(
            total_queries=result.total_queries,
            successful_queries=result.successful_evaluations,
            failed_queries=result.failed_evaluations,
            avg_precision_at_5=result.avg_precision_at_5,
            avg_recall_at_5=result.avg_recall_at_5,
            avg_ndcg_at_5=result.avg_ndcg_at_5,
            avg_mrr=result.avg_mrr,
            avg_hit_rate=result.avg_hit_rate,
            avg_faithfulness=result.avg_faithfulness,
            avg_answer_relevancy=result.avg_answer_relevancy,
            avg_citation_accuracy=result.avg_citation_accuracy,
            avg_hallucination_ratio=result.avg_hallucination_ratio,
            avg_safety_score=result.avg_safety_score,
            avg_overall_score=result.avg_overall_score,
            min_overall_score=result.min_overall_score,
            max_overall_score=result.max_overall_score,
        )