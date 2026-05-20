"""CSV Exporter for evaluation results."""

import csv
from pathlib import Path
from io import StringIO

from rag.evaluation.evaluator import RAGEvaluationResult


class CSVReporter:
    """Export evaluation results in CSV format."""

    def generate(self, results: list[RAGEvaluationResult]) -> str:
        """Generate CSV string."""
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow([
            "query_id",
            "overall_score",
            "mrr",
            "retrieval_hit_rate",
            "faithfulness",
            "answer_relevancy",
            "citation_accuracy",
            "hallucination_ratio",
            "safety_score",
        ])

        # Write data
        for r in results:
            writer.writerow([
                r.query_id,
                f"{r.overall_score:.4f}",
                f"{r.mrr:.4f}",
                f"{r.retrieval_hit_rate:.4f}",
                f"{r.faithfulness:.4f}",
                f"{r.answer_relevancy:.4f}",
                f"{r.citation_accuracy:.4f}",
                f"{r.hallucination_ratio:.4f}",
                f"{r.safety_score:.4f}",
            ])

        return output.getvalue()

    def export(self, results: list[RAGEvaluationResult], output_path: Path) -> Path:
        """Export to file."""
        content = self.generate(results)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def supports_format(self, fmt: str) -> bool:
        return fmt.lower() in ("csv",)