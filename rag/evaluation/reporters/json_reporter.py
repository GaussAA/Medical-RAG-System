"""JSON Reporter for evaluation results."""

import json
from datetime import datetime
from typing import Any

from rag.evaluation.evaluator import RAGEvaluationResult


class JSONReporter:
    """Generate JSON format evaluation reports."""

    def generate(self, results: list[RAGEvaluationResult]) -> dict[str, Any]:
        """Generate JSON structure."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "results": [r.to_dict() for r in results],
            "summary": self._compute_summary(results),
        }

    def _compute_summary(self, results: list[RAGEvaluationResult]) -> dict[str, float]:
        if not results:
            return {}

        n = len(results)
        return {
            "avg_overall_score": sum(r.overall_score for r in results) / n,
            "avg_mrr": sum(r.mrr for r in results) / n,
            "avg_faithfulness": sum(r.faithfulness for r in results) / n,
            "avg_safety_score": sum(r.safety_score for r in results) / n,
        }

    def supports_format(self, fmt: str) -> bool:
        return fmt.lower() == "json"