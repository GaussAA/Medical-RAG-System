"""Evaluation API routes."""
import asyncio
from loguru import logger
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.api.deps import RAGEngineDep
from app.models.schemas import QueryRequest
from rag.evaluation.evaluator import RAGEvaluator, EvalGroundTruth, RAGEvaluationResult

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])

MAX_CONCURRENT_BENCHMARK = 5  # Limit concurrent queries


class EvaluateRequest(BaseModel):
    query: str
    expected_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None


class BenchmarkRequest(BaseModel):
    dataset: list[dict]


# In-memory storage for evaluation history (in production, use database)
_evaluation_history: list[RAGEvaluationResult] = []


@router.post("/evaluate")
async def evaluate(request: Request, eval_request: EvaluateRequest, rag_engine: RAGEngineDep) -> dict:
    """Run single query evaluation."""
    # Build query request
    query_req = QueryRequest(
        question=eval_request.query,
        session_id=eval_request.session_id,
    )

    # Get RAG response
    session_manager = request.app.state.session_manager
    rag_response = await rag_engine.query(query_req, session_manager)

    # Evaluate
    evaluator = RAGEvaluator()
    ground_truth = EvalGroundTruth(
        query_id=eval_request.query[:50],
        relevant_doc_ids=eval_request.relevant_doc_ids,
        reference_answer=eval_request.expected_answer,
    )

    result = await evaluator.evaluate(
        query=eval_request.query,
        response=rag_response,
        ground_truth=ground_truth,
    )

    _evaluation_history.append(result)
    return result.to_dict()


async def _process_single_benchmark_item(
    item: dict,
    rag_engine: RAGEngineDep,
    session_manager,
) -> RAGEvaluationResult | None:
    """Process a single benchmark item with timeout."""
    query = item.get("query", "")
    if not query:
        return None

    ground_truth = EvalGroundTruth(
        query_id=item.get("query_id", query[:50]),
        relevant_doc_ids=item.get("relevant_doc_ids", []),
        reference_answer=item.get("expected_answer"),
    )

    try:
        # Get RAG response with timeout
        query_req = QueryRequest(question=query)
        rag_response = await asyncio.wait_for(
            rag_engine.query(query_req, session_manager),
            timeout=60.0  # 60s per query timeout
        )

        # Evaluate
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            query=query,
            response=rag_response,
            ground_truth=ground_truth,
        )
        return result
    except asyncio.TimeoutError:
        # Return a placeholder on timeout
        return RAGEvaluationResult(
            query_id=query[:50],
            overall_score=0.0,
            mrr=0.0,
            faithfulness=0.0,
        )
    except Exception as e:
        logger.error(f"Benchmark evaluation failed for query '{query[:50]}': {e}")
        return None


@router.post("/benchmark")
async def benchmark(request: Request, benchmark_request: BenchmarkRequest, rag_engine: RAGEngineDep) -> dict:
    """Run benchmark from dataset with concurrent processing."""
    session_manager = request.app.state.session_manager
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BENCHMARK)

    async def bounded_process(item: dict) -> RAGEvaluationResult | None:
        async with semaphore:
            return await _process_single_benchmark_item(item, rag_engine, session_manager)

    # Process all items concurrently (bounded by semaphore)
    tasks = [bounded_process(item) for item in benchmark_request.dataset]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    return {
        "results": [r.to_dict() for r in valid_results],
        "total": len(valid_results),
    }


@router.get("/history")
async def get_evaluation_history(limit: int = 20) -> dict:
    """Get evaluation history."""
    recent = _evaluation_history[-limit:] if _evaluation_history else []

    return {
        "history": [r.to_dict() for r in recent],
        "total": len(_evaluation_history),
    }