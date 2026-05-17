#!/usr/bin/env python
"""CLI for RAG evaluation benchmarks."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag.evaluation.evaluator import RAGEvaluator, EvalGroundTruth


app = typer.Typer(help="Medical RAG Evaluation CLI")
console = Console()


@app.command()
def evaluate(
    query: str,
    expected: str,
    retrieved_doc_ids: Annotated[str | None, typer.Option(help="Comma-separated doc IDs")] = None,
) -> None:
    """Run single query evaluation."""
    async def _run() -> None:
        evaluator = RAGEvaluator()

        # Build ground truth from expected answer
        ground_truth = EvalGroundTruth(
            query_id="cli-eval",
            reference_answer=expected,
        )

        # Mock response for evaluation - in CLI mode we evaluate retrieval against expected
        # Note: Full evaluation requires a real QueryResponse from RAG engine
        console.print("[yellow]Note: Single eval requires RAG engine integration[/yellow]")
        console.print(f"[bold]Query:[/bold] {query}")
        console.print(f"[bold]Expected:[/bold] {expected}")

        # For CLI, we primarily validate the retrieval component
        if retrieved_doc_ids:
            doc_ids = [d.strip() for d in retrieved_doc_ids.split(",")]
            from rag.evaluation.retrieval_eval import RetrievalEvaluator
            ret_eval = RetrievalEvaluator(k_values=[5, 10, 20])
            metrics = ret_eval.evaluate_without_ground_truth(doc_ids, min_relevant=1)
            console.print(f"[bold]Hit Rate:[/bold] {metrics['hit_rate']:.4f}")

    asyncio.run(_run())


@app.command()
def benchmark(
    dataset: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(help="Output JSON file for results")] = None,
) -> None:
    """Run benchmark from dataset JSONL file.

    Dataset format (JSONL):
    {
        "query": "What is the treatment for hypertension?",
        "expected_answer": "...",
        "relevant_doc_ids": ["doc1", "doc2"],
        "retrieved_doc_ids": ["doc1", "doc3"]
    }
    """
    async def _run() -> None:
        evaluator = RAGEvaluator()

        with open(dataset) as f:
            data = [json.loads(line) for line in f]

        results = []
        for item in data:
            query = item["query"]
            expected = item.get("expected_answer", "")
            relevant_ids = item.get("relevant_doc_ids", [])
            retrieved_ids = item.get("retrieved_doc_ids", [])

            ground_truth = EvalGroundTruth(
                query_id=item.get("query_id", f"bench-{len(results)}"),
                reference_answer=expected,
                relevant_doc_ids=relevant_ids,
            )

            # Run retrieval evaluation
            from rag.evaluation.retrieval_eval import RetrievalEvaluator
            ret_eval = RetrievalEvaluator(k_values=[5, 10, 20])

            if relevant_ids and retrieved_ids:
                metrics = ret_eval.evaluate(
                    retrieved_ids=retrieved_ids,
                    ground_truth_ids=relevant_ids,
                )
                retrieval_score = (metrics.mrr + metrics.hit_rate) / 2
            else:
                metrics = ret_eval.evaluate_without_ground_truth(retrieved_ids or [], min_relevant=1)
                retrieval_score = metrics["hit_rate"]

            results.append({
                "query": query,
                "query_id": ground_truth.query_id,
                "retrieval_score": retrieval_score,
                "hit_rate": metrics.hit_rate if hasattr(metrics, "hit_rate") else metrics.get("hit_rate", 0),
                "mrr": metrics.mrr if hasattr(metrics, "mrr") else 0,
            })

        _print_results(results)

        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {output}[/green]")

    asyncio.run(_run())


def _print_results(results: list[dict]) -> None:
    """Print benchmark results as a table."""
    table = Table(title="Benchmark Results")
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Retrieval Score", justify="right", style="green")
    table.add_column("Hit Rate", justify="right", style="yellow")
    table.add_column("MRR", justify="right", style="magenta")

    for r in results:
        table.add_row(
            r["query"][:60] + "..." if len(r["query"]) > 60 else r["query"],
            f"{r['retrieval_score']:.4f}",
            f"{r['hit_rate']:.4f}",
            f"{r['mrr']:.4f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()