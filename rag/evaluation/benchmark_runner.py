"""Benchmark Runner Module.

Provides batch evaluation capabilities for RAG systems using benchmark datasets.
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from app.models.schemas import QueryResponse
from rag.evaluation.evaluator import RAGEvaluator, RAGEvaluationResult, EvalGroundTruth


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    dataset_path: str
    sample_size: int | None = None
    output_dir: str = "data/evaluation/reports"
    include_failed_cases: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    benchmark_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    dataset_name: str = ""
    total_queries: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0

    # Aggregated metrics
    avg_precision_at_5: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_ndcg_at_5: float = 0.0
    avg_mrr: float = 0.0
    avg_hit_rate: float = 0.0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_citation_accuracy: float = 0.0
    avg_hallucination_ratio: float = 0.0
    avg_safety_score: float = 0.0
    avg_overall_score: float = 0.0

    # Per-query results
    results: list[RAGEvaluationResult] = field(default_factory=list)

    # Statistics
    min_overall_score: float = 0.0
    max_overall_score: float = 0.0
    std_overall_score: float = 0.0

    # Failed cases
    failed_cases: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "dataset_name": self.dataset_name,
            "total_queries": self.total_queries,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "aggregated_metrics": {
                "precision_at_5": self.avg_precision_at_5,
                "recall_at_5": self.avg_recall_at_5,
                "ndcg_at_5": self.avg_ndcg_at_5,
                "mrr": self.avg_mrr,
                "hit_rate": self.avg_hit_rate,
                "faithfulness": self.avg_faithfulness,
                "answer_relevancy": self.avg_answer_relevancy,
                "citation_accuracy": self.avg_citation_accuracy,
                "hallucination_ratio": self.avg_hallucination_ratio,
                "safety_score": self.avg_safety_score,
                "overall_score": self.avg_overall_score,
            },
            "statistics": {
                "min_overall_score": self.min_overall_score,
                "max_overall_score": self.max_overall_score,
                "std_overall_score": self.std_overall_score,
            },
            "results": [r.to_dict() for r in self.results],
            "failed_cases": self.failed_cases,
        }


class BenchmarkRunner:
    """
    Benchmark runner for RAG evaluation.

    Loads evaluation datasets and runs batch evaluations.
    """

    def __init__(self, config: BenchmarkConfig | None = None):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig(dataset_path="")

    def load_dataset(self, dataset_path: str | None = None) -> list[EvalGroundTruth]:
        """
        Load evaluation dataset from JSON file.

        Args:
            dataset_path: Path to dataset JSON file.

        Returns:
            List of EvalGroundTruth objects.
        """
        path = dataset_path or self.config.dataset_path
        if not path:
            raise ValueError("No dataset path provided")

        dataset_file = Path(path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ground_truths = []
        for item in data:
            gt = EvalGroundTruth(
                query_id=item.get("query_id", ""),
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                expected_keywords=item.get("expected_keywords", []),
                reference_answer=item.get("reference_answer"),
                difficulty=item.get("difficulty", "medium"),
                safety_sensitive=item.get("safety_sensitive", False),
            )
            ground_truths.append(gt)

        logger.info(f"Loaded {len(ground_truths)} evaluation queries from {path}")
        return ground_truths

    async def run(
        self,
        evaluator: RAGEvaluator,
        dataset_path: str | None = None,
        queries_data: list[dict[str, Any]] | None = None,
        sample_size: int | None = None,
        rag_engine: Any | None = None,
    ) -> BenchmarkResult:
        """
        Run benchmark evaluation with RAG engine.

        Args:
            evaluator: RAGEvaluator instance.
            dataset_path: Path to dataset JSON file.
            queries_data: Alternative to dataset_path - list of query dicts.
            sample_size: Optional limit on number of queries to evaluate.
            rag_engine: Optional RAGEngine instance. If provided, queries will
                       be executed against it for end-to-end evaluation.

        Returns:
            BenchmarkResult with aggregated metrics.
        """
        # Load dataset
        if queries_data:
            ground_truths = [
                EvalGroundTruth(
                    query_id=q.get("query_id", f"q_{i}"),
                    relevant_doc_ids=q.get("relevant_doc_ids", []),
                    expected_keywords=q.get("expected_keywords", []),
                    reference_answer=q.get("reference_answer"),
                    difficulty=q.get("difficulty", "medium"),
                    safety_sensitive=q.get("safety_sensitive", False),
                )
                for i, q in enumerate(queries_data)
            ]
            query_texts = [q.get("query") or q.get("query_text") or "" for q in queries_data]
        else:
            ground_truths = self.load_dataset(dataset_path)
            query_texts = [gt.query_id for gt in ground_truths]

        # Apply sampling
        if sample_size:
            ground_truths = ground_truths[:sample_size]
            query_texts = query_texts[:sample_size]
        elif self.config.sample_size:
            ground_truths = ground_truths[:self.config.sample_size]
            query_texts = query_texts[:self.config.sample_size]

        dataset_name = Path(dataset_path or self.config.dataset_path).stem
        benchmark_id = f"benchmark_{dataset_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            dataset_name=dataset_name,
            total_queries=len(ground_truths),
        )

        if rag_engine:
            # Execute each query against the RAG engine and evaluate
            from app.models.schemas import QueryRequest

            eval_results = []
            for i, gt in enumerate(ground_truths):
                try:
                    query_text = query_texts[i] if i < len(query_texts) else ""
                    if not query_text:
                        logger.warning(f"Empty query text for {gt.query_id}")
                        result.failed_evaluations += 1
                        continue

                    query_req = QueryRequest(question=query_text)
                    response = await rag_engine.query(query_req)

                    eval_result = await evaluator.evaluate(
                        query=query_text,
                        response=response,
                        ground_truth=gt,
                    )
                    eval_results.append(eval_result)
                    result.successful_evaluations += 1

                except Exception as e:
                    logger.error(f"Failed to evaluate query {gt.query_id}: {e}")
                    result.failed_evaluations += 1
                    if self.config.include_failed_cases:
                        result.failed_cases.append({
                            "query_id": gt.query_id,
                            "error": str(e),
                        })

            result.results = eval_results
        else:
            # Fallback: without rag_engine, only collect metadata
            for gt in ground_truths:
                logger.debug(f"Evaluating query (metadata only): {gt.query_id}")

        # Calculate aggregated statistics
        self._calculate_statistics(result)

        logger.info(
            f"Benchmark completed: {result.successful_evaluations}/{result.total_queries} successful"
        )

        return result

    async def run_with_responses(
        self,
        evaluator: RAGEvaluator,
        queries: list[str],
        responses: list[Any],
        ground_truths: list[EvalGroundTruth] | None = None,
        retrieved_doc_ids_list: list[list[str]] | None = None,
    ) -> BenchmarkResult:
        """
        Run benchmark with pre-computed RAG responses.

        This is useful when you already have RAG responses and want to evaluate them.

        Args:
            evaluator: RAGEvaluator instance.
            queries: List of query strings.
            responses: List of QueryResponse objects.
            ground_truths: Optional list of ground truths.
            retrieved_doc_ids_list: Optional list of retrieved doc ID lists.

        Returns:
            BenchmarkResult with aggregated metrics.
        """
        benchmark_id = f"benchmark_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            total_queries=len(queries),
        )

        eval_results = []

        for i, response in enumerate(responses):
            try:
                query = queries[i] if i < len(queries) else ""
                gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
                retrieved_ids = (
                    retrieved_doc_ids_list[i]
                    if retrieved_doc_ids_list and i < len(retrieved_doc_ids_list)
                    else None
                )

                eval_result = await evaluator.evaluate(
                    query=query,
                    response=response,
                    ground_truth=gt,
                    retrieved_doc_ids=retrieved_ids,
                )
                eval_results.append(eval_result)
                result.successful_evaluations += 1

            except Exception as e:
                logger.error(f"Failed to evaluate query {i}: {e}")
                result.failed_evaluations += 1
                if self.config.include_failed_cases:
                    result.failed_cases.append({
                        "query_id": getattr(ground_truths[i] if ground_truths else None, "query_id", f"q_{i}"),
                        "error": str(e),
                    })

        result.results = eval_results

        # Calculate aggregated statistics
        self._calculate_statistics(result)

        return result

    async def run_hybrid(
        self,
        evaluator: RAGEvaluator,
        queries_data: list[dict[str, Any]],
        api_url: str | None = None,
        fallback_file: str | None = None,
        parallel: int = 1,
    ) -> BenchmarkResult:
        """
        Hybrid mode:优先实时调用，失败时回退到离线。

        Args:
            evaluator: RAGEvaluator 实例
            queries_data: 查询数据列表
            api_url: RAG API 地址
            fallback_file: 离线响应文件路径
            parallel: 并发数量（当前未实现）

        Returns:
            BenchmarkResult with aggregated metrics.
        """
        benchmark_id = f"benchmark_hybrid_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            total_queries=len(queries_data),
        )

        for query_item in queries_data:
            query_id = query_item.get("query_id", "")
            query_text = query_item.get("query_text", "")
            response: QueryResponse | None = None

            try:
                # Step 1: 尝试实时 API 调用
                if api_url:
                    response = await self._call_rag_api(api_url, query_text)

                # Step 2: 回退到离线响应
                if response is None and fallback_file:
                    response = await self._load_fallback_response(fallback_file, query_id)

                # Step 3: 执行评估
                if response is not None:
                    ground_truth = EvalGroundTruth(
                        query_id=query_id,
                        relevant_doc_ids=query_item.get("relevant_doc_ids", []),
                        expected_keywords=query_item.get("expected_keywords", []),
                        reference_answer=query_item.get("reference_answer"),
                        difficulty=query_item.get("difficulty", "medium"),
                        safety_sensitive=query_item.get("safety_sensitive", False),
                    )

                    retrieved_ids = None
                    if hasattr(response, "metadata") and response.metadata:
                        retrieved_ids = response.metadata.get("retrieved_doc_ids")

                    eval_result = await evaluator.evaluate(
                        query=query_text,
                        response=response,
                        ground_truth=ground_truth,
                        retrieved_doc_ids=retrieved_ids,
                    )
                    result.results.append(eval_result)
                    result.successful_evaluations += 1
                else:
                    # 无法获取响应
                    result.failed_evaluations += 1
                    if self.config.include_failed_cases:
                        result.failed_cases.append({
                            "query_id": query_id,
                            "error": "No response available from API or fallback",
                        })

            except Exception as e:
                logger.error(f"Failed to evaluate query {query_id}: {e}")
                result.failed_evaluations += 1
                if self.config.include_failed_cases:
                    result.failed_cases.append({
                        "query_id": query_id,
                        "error": str(e),
                    })

        # 计算聚合统计
        self._calculate_statistics(result)

        logger.info(
            f"Hybrid benchmark completed: {result.successful_evaluations}/{result.total_queries} successful"
        )

        return result

    async def _call_rag_api(self, api_url: str, query: str) -> QueryResponse | None:
        """
        调用 RAG API。

        Args:
            api_url: API 基础地址
            query: 查询文本

        Returns:
            QueryResponse if successful, None otherwise.
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{api_url}/api/v1/query",
                    json={"question": query},
                )
                if response.status_code == 200:
                    data = response.json()
                    return QueryResponse(**data)
        except Exception as e:
            logger.warning(f"RAG API call failed: {e}")
        return None

    async def _load_fallback_response(
        self, fallback_file: str, query_id: str
    ) -> QueryResponse | None:
        """
        从离线文件加载响应。

        Args:
            fallback_file: 离线响应文件路径
            query_id: 查询 ID

        Returns:
            QueryResponse if found, None otherwise.
        """
        try:
            fallback_path = Path(fallback_file)
            if not fallback_path.exists():
                logger.warning(f"Fallback file not found: {fallback_file}")
                return None

            with open(fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 支持两种格式：字典或字典列表
            if isinstance(data, dict):
                # 格式1: {query_id: response_dict}
                if query_id in data:
                    return QueryResponse(**data[query_id])
            elif isinstance(data, list):
                # 格式2: [{query_id, ...}, ...]
                for item in data:
                    if item.get("query_id") == query_id:
                        return QueryResponse(
                            answer=item.get("answer", ""),
                            confidence=item.get("confidence", 0.0),
                            citations=item.get("citations", []),
                            warnings=item.get("warnings", []),
                            session_id=item.get("session_id", ""),
                            processing_time=item.get("processing_time", 0.0),
                            metadata=item.get("metadata", {}),
                        )

            logger.warning(f"Query {query_id} not found in fallback file")
            return None

        except Exception as e:
            logger.error(f"Failed to load fallback response: {e}")
            return None

    def _calculate_statistics(self, result: BenchmarkResult) -> None:
        """Calculate aggregated statistics from results."""
        if not result.results:
            return

        # Extract metric values
        overall_scores = [r.overall_score for r in result.results]

        # Calculate aggregates
        result.avg_overall_score = sum(overall_scores) / len(overall_scores)
        result.min_overall_score = min(overall_scores)
        result.max_overall_score = max(overall_scores)

        # Standard deviation
        mean = result.avg_overall_score
        variance = sum((s - mean) ** 2 for s in overall_scores) / len(overall_scores)
        result.std_overall_score = variance**0.5

        # Retrieval metrics
        if result.results[0].precision_at_k:
            p5_values = [r.precision_at_k.get(5, 0.0) for r in result.results]
            result.avg_precision_at_5 = sum(p5_values) / len(p5_values)

        if result.results[0].recall_at_k:
            r5_values = [r.recall_at_k.get(5, 0.0) for r in result.results]
            result.avg_recall_at_5 = sum(r5_values) / len(r5_values)

        if result.results[0].ndcg_at_k:
            n5_values = [r.ndcg_at_k.get(5, 0.0) for r in result.results]
            result.avg_ndcg_at_5 = sum(n5_values) / len(n5_values)

        mrr_values = [r.mrr for r in result.results]
        result.avg_mrr = sum(mrr_values) / len(mrr_values)

        hr_values = [r.retrieval_hit_rate for r in result.results]
        result.avg_hit_rate = sum(hr_values) / len(hr_values)

        # Generation metrics
        faith_values = [r.faithfulness for r in result.results]
        result.avg_faithfulness = sum(faith_values) / len(faith_values)

        rel_values = [r.answer_relevancy for r in result.results]
        result.avg_answer_relevancy = sum(rel_values) / len(rel_values)

        cite_values = [r.citation_accuracy for r in result.results]
        result.avg_citation_accuracy = sum(cite_values) / len(cite_values)

        hall_values = [r.hallucination_ratio for r in result.results]
        result.avg_hallucination_ratio = sum(hall_values) / len(hall_values)

        # Safety metrics
        safety_values = [r.safety_score for r in result.results if r.safety_score > 0]
        if safety_values:
            result.avg_safety_score = sum(safety_values) / len(safety_values)

    def save_results(self, result: BenchmarkResult, output_path: str | None = None) -> str:
        """
        Save benchmark results to JSON file.

        Args:
            result: BenchmarkResult to save.
            output_path: Optional output path.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"{result.benchmark_id}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Benchmark results saved to {output_path}")
        return output_path