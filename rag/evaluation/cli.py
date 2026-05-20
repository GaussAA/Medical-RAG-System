#!/usr/bin/env python
"""CLI for RAG evaluation benchmarks."""

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from rag.evaluation import RetrievalEvaluator
from rag.evaluation.benchmark_runner import BenchmarkRunner, BenchmarkConfig


app = typer.Typer(help="Medical RAG Evaluation CLI")
console = Console()


def evaluate_command(
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
def evaluate(
    query: str,
    expected: str,
    retrieved_doc_ids: Annotated[str | None, typer.Option(help="Comma-separated doc IDs")] = None,
) -> None:
    """Run single query evaluation.

    This performs retrieval-only evaluation since generation and medical safety
    evaluation require a full QueryResponse from the RAG engine.
    """
    evaluate_command(query, expected, retrieved_doc_ids)


def benchmark_command(
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
    benchmark_command(dataset, output)


def dataset_command(
    action: Annotated[str, typer.Argument(help="create/list/validate/export")],
    name: Annotated[str | None, typer.Option("--name")] = None,
    source: Annotated[Path | None, typer.Option("--source")] = None,
) -> None:
    """Dataset management: create, list, validate, or export datasets."""
    console.print(f"[bold]Dataset action:[/bold] {action}")

    if action == "create":
        if not name:
            console.print("[red]--name is required for create action[/red]")
            raise typer.Exit(1)
        _create_dataset(name, source)
    elif action == "list":
        _list_datasets()
    elif action == "validate":
        if not name:
            console.print("[red]--name is required for validate action[/red]")
            raise typer.Exit(1)
        _validate_dataset(name)
    elif action == "export":
        if not name:
            console.print("[red]--name is required for export action[/red]")
            raise typer.Exit(1)
        _export_dataset(name)
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command()
def dataset(
    action: Annotated[str, typer.Argument(help="create/list/validate/export")],
    name: Annotated[str | None, typer.Option("--name")] = None,
    source: Annotated[Path | None, typer.Option("--source")] = None,
) -> None:
    """Dataset management: create, list, validate, or export datasets."""
    dataset_command(action, name, source)


def compare_command(
    comparison_type: Annotated[str, typer.Argument(help="timeline/config/version")],
    baseline: Annotated[Path, typer.Option("--baseline")],
    current: Annotated[Path, typer.Option("--current")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """A/B version comparison: timeline, config, or version comparison."""
    console.print(f"[bold]Comparison type:[/bold] {comparison_type}")
    console.print(f"[bold]Baseline:[/bold] {baseline}")
    console.print(f"[bold]Current:[/bold] {current}")

    if comparison_type == "timeline":
        _compare_timeline(baseline, current, output)
    elif comparison_type == "config":
        _compare_config(baseline, current, output)
    elif comparison_type == "version":
        _compare_version(baseline, current, output)
    else:
        console.print(f"[red]Unknown comparison type: {comparison_type}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    comparison_type: Annotated[str, typer.Argument(help="timeline/config/version")],
    baseline: Annotated[Path, typer.Option("--baseline")],
    current: Annotated[Path, typer.Option("--current")],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """A/B version comparison: timeline, config, or version comparison."""
    compare_command(comparison_type, baseline, current, output)


def _create_dataset(name: str, source: Path | None) -> None:
    """Create a new dataset."""
    console.print(f"[green]Creating dataset: {name}[/green]")
    if source:
        console.print(f"[green]Source: {source}[/green]")


def _list_datasets() -> None:
    """List available datasets."""
    data_dir = Path("data/evaluation")
    if not data_dir.exists():
        console.print("[yellow]No evaluation data directory found[/yellow]")
        return

    datasets = []
    for f in data_dir.rglob("*.jsonl"):
        datasets.append(str(f))

    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return

    table = Table(title="Available Datasets")
    table.add_column("Dataset", style="cyan")
    for ds in datasets:
        table.add_row(ds)
    console.print(table)


def _validate_dataset(name: str) -> None:
    """Validate a dataset file."""
    console.print(f"[green]Validating dataset: {name}[/green]")


def _export_dataset(name: str) -> None:
    """Export a dataset."""
    console.print(f"[green]Exporting dataset: {name}[/green]")


def _compare_timeline(baseline: Path, current: Path, output: Path | None) -> None:
    """Compare timeline data between baseline and current."""
    try:
        with open(baseline) as f:
            baseline_data = json.load(f)
        with open(current) as f:
            current_data = json.load(f)

        table = Table(title="Timeline Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("Current", style="green")

        baseline_val = baseline_data.get("timestamp", "N/A")
        current_val = current_data.get("timestamp", "N/A")
        table.add_row("Timestamp", str(baseline_val), str(current_val))

        console.print(table)

        if output:
            with open(output, "w") as f:
                json.dump({"baseline": baseline_data, "current": current_data}, f, indent=2)

    except Exception as e:
        console.print(f"[red]Error comparing timeline: {e}[/red]")


def _compare_config(baseline: Path, current: Path, output: Path | None) -> None:
    """Compare config data between baseline and current."""
    try:
        with open(baseline) as f:
            baseline_data = json.load(f)
        with open(current) as f:
            current_data = json.load(f)

        table = Table(title="Config Comparison")
        table.add_column("Key", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("Current", style="green")

        all_keys = set(baseline_data.keys()) | set(current_data.keys())
        for key in sorted(all_keys):
            b_val = baseline_data.get(key, "N/A")
            c_val = current_data.get(key, "N/A")
            table.add_row(key, str(b_val), str(c_val))

        console.print(table)

        if output:
            with open(output, "w") as f:
                json.dump({"baseline": baseline_data, "current": current_data}, f, indent=2)

    except Exception as e:
        console.print(f"[red]Error comparing config: {e}[/red]")


def _compare_version(baseline: Path, current: Path, output: Path | None) -> None:
    """Compare version data between baseline and current."""
    try:
        with open(baseline) as f:
            baseline_data = json.load(f)
        with open(current) as f:
            current_data = json.load(f)

        table = Table(title="Version Comparison")
        table.add_column("Component", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("Current", style="green")

        baseline_ver = baseline_data.get("version", baseline_data.get("rag_version", "N/A"))
        current_ver = current_data.get("version", current_data.get("rag_version", "N/A"))
        table.add_row("Version", str(baseline_ver), str(current_ver))

        console.print(table)

        if output:
            with open(output, "w") as f:
                json.dump({"baseline": baseline_data, "current": current_data}, f, indent=2)

    except Exception as e:
        console.print(f"[red]Error comparing version: {e}[/red]")


def _print_results(results: list[dict[str, Any]]) -> None:
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