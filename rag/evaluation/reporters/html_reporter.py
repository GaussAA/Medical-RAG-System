"""HTML Report Generator with Plotly charts."""

from datetime import datetime

import plotly.express as px  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]

from rag.evaluation.evaluator import RAGEvaluationResult


class HTMLReporter:
    """Generate HTML evaluation reports with interactive charts."""

    def generate(self, results: list[RAGEvaluationResult]) -> str:
        """Generate complete HTML report."""
        if not results:
            return self._empty_report()

        summary = self._compute_summary(results)
        radar_html = self._generate_radar_chart(summary)
        histogram_html = self._generate_histogram(results)
        table_html = self._generate_results_table(results)

        return self._build_html(
            radar=radar_html,
            histogram=histogram_html,
            table=table_html,
            summary=summary,
        )

    def _generate_radar_chart(self, summary: dict) -> str:
        """Generate radar chart."""
        categories = ["Retrieval", "Generation", "Safety"]
        values = [
            summary.get("retrieval_score", 0),
            summary.get("generation_score", 0),
            summary.get("safety_score", 0),
        ]

        fig = go.Figure(
            data=go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name="Score",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400,
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _generate_histogram(self, results: list[RAGEvaluationResult]) -> str:
        """Generate score distribution histogram."""
        scores = [r.overall_score for r in results]

        fig = px.histogram(
            x=scores,
            nbins=10,
            labels={"x": "Overall Score", "y": "Count"},
            title="Score Distribution",
        )

        fig.update_layout(height=300, showlegend=False)
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _generate_results_table(self, results: list[RAGEvaluationResult]) -> str:
        """Generate detailed results table."""
        rows = []
        for r in results:
            status = "PASS" if r.overall_score >= 0.7 else "FAIL"
            rows.append(f"""
                <tr>
                    <td>{r.query_id[:30]}</td>
                    <td>{r.overall_score:.3f}</td>
                    <td>{r.mrr:.3f}</td>
                    <td>{r.faithfulness:.3f}</td>
                    <td>{r.safety_score:.3f}</td>
                    <td>{status}</td>
                </tr>
            """)

        return f"""
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Query</th>
                        <th>Overall</th>
                        <th>MRR</th>
                        <th>Faithfulness</th>
                        <th>Safety</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        """

    def _compute_summary(self, results: list[RAGEvaluationResult]) -> dict:
        """Compute summary metrics."""
        n = len(results)
        if n == 0:
            return {}

        return {
            "total": n,
            "avg_overall": sum(r.overall_score for r in results) / n,
            "retrieval_score": sum(r.mrr for r in results) / n,
            "generation_score": sum(r.faithfulness for r in results) / n,
            "safety_score": sum(r.safety_score for r in results) / n if any(r.safety_score > 0 for r in results) else 0,
        }

    def _build_html(self, radar: str, histogram: str, table: str, summary: dict) -> str:
        """Build complete HTML page."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Medical RAG Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin: 30px 0; }}
        .chart {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Medical RAG Evaluation Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total Queries: {summary.get("total", 0)}</p>
        <p>Average Overall Score: {summary.get("avg_overall", 0):.4f}</p>
    </div>

    <div class="section">
        <h2>Overall Score Distribution</h2>
        <div class="chart">{radar}</div>
    </div>

    <div class="section">
        <h2>Score Distribution</h2>
        <div class="chart">{histogram}</div>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        {table}
    </div>
</body>
</html>
        """

    def _empty_report(self) -> str:
        return "<html><body><h1>No evaluation results</h1></body></html>"

    def supports_format(self, fmt: str) -> bool:
        return fmt.lower() in ("html", "htm")
