#!/usr/bin/env python
"""CLI for RAG evaluation benchmarks."""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag.evaluation import RetrievalEvaluator
from rag.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig


app = typer.Typer(help="Medical RAG Evaluation CLI")
console = Console()


@app.command()
def evaluate(
    query: str,
    expected: str,
    retrieved_doc_ids: Annotated[str | None, typer.Option(help="Comma-separated doc IDs")] = None,
) -> None:
    """Run single query evaluation.

    This performs retrieval-only evaluation since generation and medical safety
    evaluation require a full QueryResponse from the RAG engine.
    """
    async def _run() -> None:
        console.print(f"[bold]Query:[/bold] {query}")
        console.print(f"[bold]Expected:[/bold] {expected}")

        if not retrieved_doc_ids:
            console.print("[yellow]No retrieved_doc_ids provided. Only validation possible.[/yellow]")
            return

        doc_ids = [d.strip() for d in retrieved_doc_ids.split(",")]
        ret_eval = RetrievalEvaluator(k_values=[5, 10, 20])

        if doc_ids:
            metrics = ret_eval.evaluate_without_ground_truth(doc_ids, min_relevant=1)
            console.print(f"[bold]Retrieved Documents:[/bold] {len(doc_ids)}")
            console.print(f"[bold]Hit Rate:[/bold] {metrics['hit_rate']:.4f}")
            console.print(f"[bold]Retrieval Count:[/bold] {metrics['retrieved_count']}")
        else:
            console.print("[yellow]No retrieved document IDs provided[/yellow]")

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

    Note: This performs retrieval-only evaluation. Full RAG evaluation with
    generation and medical safety metrics requires integration with the RAG engine.
    Use BenchmarkRunner for comprehensive evaluation.
    """
    async def _run() -> None:
        # Initialize BenchmarkRunner for proper benchmark execution
        runner = BenchmarkRunner(config=BenchmarkConfig(
            dataset_path=str(dataset),
            output_dir=str(output.parent) if output else "data/evaluation/reports",
        ))

        # Load JSONL data
        with open(dataset) as f:
            lines = f.read().splitlines()

        queries_data = []
        for line in lines:
            if line.strip():
                item = json.loads(line)
                queries_data.append({
                    "query_id": item.get("query_id", f"bench-{len(queries_data)}"),
                    "query_text": item["query"],
                    "relevant_doc_ids": item.get("relevant_doc_ids", []),
                    "retrieved_doc_ids": item.get("retrieved_doc_ids", []),
                    "expected_answer": item.get("expected_answer", ""),
                })

        ret_eval = RetrievalEvaluator(k_values=[5, 10, 20])
        results = []

        for item in queries_data:
            query = item["query_text"]
            relevant_ids = item.get("relevant_doc_ids", [])
            retrieved_ids = item.get("retrieved_doc_ids", [])

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
                "query_id": item["query_id"],
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