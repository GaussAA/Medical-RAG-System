#!/usr/bin/env python
"""Medical RAG Evaluation CLI"""

import typer
from pathlib import Path

app = typer.Typer(
    name="rag-eval",
    help="Medical RAG System Evaluation CLI",
)


@app.command()
def evaluate(
    query: str = typer.Option(None, "--query", "-q", help="Query text"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url"),
    response_file: Path = typer.Option(None, "--response-file", help="Offline response file"),
    fallback_file: Path = typer.Option(None, "--fallback-file", help="Fallback response file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output file"),
):
    """单次查询评估"""
    from rag.evaluation.cli import evaluate_command
    typer.run(evaluate_command)


@app.command()
def benchmark(
    dataset: Path = typer.Argument(..., exists=True),
    mode: str = typer.Option("online", "--mode", help="online/offline/hybrid"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url"),
    output_dir: Path = typer.Option(Path("data/evaluation/results"), "--output-dir"),
    sample: int = typer.Option(None, "--sample"),
):
    """批量基准测试"""
    from rag.evaluation.cli import benchmark_command
    typer.run(benchmark_command)


@app.command()
def dataset(
    action: str = typer.Argument(..., help="create/list/validate/export"),
    name: str = typer.Option(None, "--name"),
    source: Path = typer.Option(None, "--source"),
):
    """数据集管理"""
    from rag.evaluation.cli import dataset_command
    typer.run(dataset_command)


@app.command()
def compare(
    comparison_type: str = typer.Argument(..., help="timeline/config/version"),
    baseline: Path = typer.Option(..., "--baseline"),
    current: Path = typer.Option(..., "--current"),
    output: Path = typer.Option(None, "--output"),
):
    """A/B 版本对比"""
    from rag.evaluation.cli import compare_command
    typer.run(compare_command)


if __name__ == "__main__":
    app()