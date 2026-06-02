"""Evaluation API routes."""
import asyncio
import json
from pathlib import Path

from loguru import logger
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.api.deps import RAGEngineDep
from app.models.schemas import QueryRequest
from rag.evaluation.evaluator import RAGEvaluator, EvalGroundTruth, RAGEvaluationResult
from rag.generation.llm_generator import LLMGenerator
from app.services.document_store import DocumentStore

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])

MAX_CONCURRENT_BENCHMARK = 5  # Limit concurrent queries

# JSONL persistence for evaluation history
EVALUATION_HISTORY_PATH = Path("data/evaluation/results/history.jsonl")
MAX_HISTORY_ENTRIES = 1000
_history_save_lock = asyncio.Lock()


class EvaluateRequest(BaseModel):
    query: str
    expected_answer: str | None = None
    relevant_doc_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None


class BenchmarkRequest(BaseModel):
    dataset: list[dict]


# Cache for document-title-to-chunk-IDs resolution (lazy loaded)
_doc_title_to_chunk_ids: dict[str, list[str]] | None = None


async def _resolve_relevant_doc_ids(doc_titles: list[str]) -> list[str]:
    """
    Resolve document title-based ground truth IDs to chunk-level IDs.

    The dataset stores 'relevant_doc_ids' as document titles (e.g. "儿童肺炎支原体肺炎诊疗指南（2025年版）"),
    but the RAG system retrieves chunk-level IDs. This function resolves
    document titles to their chunk IDs so retrieval metrics can be computed correctly.
    """
    global _doc_title_to_chunk_ids
    if _doc_title_to_chunk_ids is None:
        _doc_title_to_chunk_ids = {}
        async with DocumentStore() as store:
            docs, _ = await store.list_documents(page_size=100)
            for doc in docs:
                doc_uuid = str(doc.id)
                # Index by title (with and without file extension)
                _doc_title_to_chunk_ids[doc.title] = doc_uuid
                _doc_title_to_chunk_ids[doc.file_name] = doc_uuid
                for ext in (".md", ".markdown"):
                    if doc.title.endswith(ext):
                        _doc_title_to_chunk_ids[doc.title[: -len(ext)]] = doc_uuid
                    if doc.file_name.endswith(ext):
                        _doc_title_to_chunk_ids[doc.file_name[: -len(ext)]] = doc_uuid

    resolved_doc_ids = []
    for tid in doc_titles:
        if tid in _doc_title_to_chunk_ids:
            resolved_doc_ids.append(_doc_title_to_chunk_ids[tid])
        else:
            matched = False
            for key, val in _doc_title_to_chunk_ids.items():
                if tid in key or key in tid:
                    resolved_doc_ids.append(val)
                    matched = True
                    break
            if not matched:
                logger.warning(f"No document found for title: {tid}")

    result = list(dict.fromkeys(resolved_doc_ids))
    logger.info(f"Resolved {len(doc_titles)} doc title(s) to {len(result)} doc UUID(s)")
    if not result and doc_titles:
        logger.warning(f"Resolution produced 0 doc IDs for {doc_titles}")
    return result


# In-memory storage for evaluation history, backed by JSONL on disk
_evaluation_history: list[RAGEvaluationResult] = []


async def _save_result_to_disk(result_dict: dict) -> None:
    """Append a single evaluation result to the JSONL history file.

    Uses an asyncio lock to prevent interleaved writes from concurrent requests.
    The file is kept at MAX_HISTORY_ENTRIES lines by trimming from the top.
    """
    async with _history_save_lock:
        try:
            EVALUATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Append the new result
            line = json.dumps(result_dict, ensure_ascii=False)
            with open(EVALUATION_HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            # Trim to MAX_HISTORY_ENTRIES if needed
            if EVALUATION_HISTORY_PATH.stat().st_size > 10 * 1024 * 1024:  # 10MB threshold
                lines = EVALUATION_HISTORY_PATH.read_text(encoding="utf-8").splitlines()
                if len(lines) > MAX_HISTORY_ENTRIES:
                    trimmed = lines[-MAX_HISTORY_ENTRIES:]
                    EVALUATION_HISTORY_PATH.write_text(
                        "\n".join(trimmed) + "\n", encoding="utf-8"
                    )
                    logger.info(f"Trimmed evaluation history to {MAX_HISTORY_ENTRIES} entries")

        except Exception as e:
            logger.warning(f"Failed to persist evaluation result to disk: {e}")


def _load_history_from_disk() -> list[RAGEvaluationResult]:
    """Load evaluation history from the JSONL file into memory."""
    results = []
    try:
        if EVALUATION_HISTORY_PATH.exists():
            with open(EVALUATION_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        result = RAGEvaluationResult(
                            query_id=data.get("query_id", "unknown"),
                            overall_score=data.get("overall_score", 0.0),
                        )
                        # Parse sub-metrics if available
                        retrieval = data.get("retrieval", {})
                        result.mrr = retrieval.get("mrr", 0.0)
                        result.retrieval_hit_rate = retrieval.get("hit_rate", 0.0)
                        result.precision_at_k = retrieval.get("precision_at_k", {})
                        result.recall_at_k = retrieval.get("recall_at_k", {})
                        result.ndcg_at_k = retrieval.get("ndcg_at_k", {})

                        generation = data.get("generation", {})
                        result.faithfulness = generation.get("faithfulness", 0.0)
                        result.answer_relevancy = generation.get("answer_relevancy", 0.0)
                        result.citation_accuracy = generation.get("citation_accuracy", 0.0)
                        result.hallucination_ratio = generation.get("hallucination_ratio", 0.0)

                        safety = data.get("medical_safety", {})
                        result.safety_score = safety.get("safety_score", 0.0)
                        result.entity_accuracy = safety.get("entity_accuracy")
                        result.contradiction_detected = safety.get("contradiction_detected", False)

                        result.details = data.get("details", {})
                        results.append(result)

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Skipping malformed history line: {e}")
                        continue

            logger.info(f"Loaded {len(results)} evaluation results from {EVALUATION_HISTORY_PATH}")

    except Exception as e:
        logger.warning(f"Failed to load evaluation history from disk: {e}")

    return results


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

    # Resolve document-title-based ground truth IDs to chunk-level IDs
    ground_truth_ids = await _resolve_relevant_doc_ids(eval_request.relevant_doc_ids)

    # Evaluate
    evaluator = RAGEvaluator(llm_generator=LLMGenerator())
    ground_truth = EvalGroundTruth(
        query_id=eval_request.query[:50],
        relevant_doc_ids=ground_truth_ids,
        reference_answer=eval_request.expected_answer,
    )

    result = await evaluator.evaluate(
        query=eval_request.query,
        response=rag_response,
        ground_truth=ground_truth,
    )

    result_dict = result.to_dict()
    _evaluation_history.append(result)

    # Persist to disk (best-effort, won't fail the request)
    await _save_result_to_disk(result_dict)

    return result_dict


async def _process_single_benchmark_item(
    item: dict,
    rag_engine: RAGEngineDep,
    session_manager,
) -> RAGEvaluationResult | None:
    """Process a single benchmark item with timeout."""
    query = item.get("query", "")
    if not query:
        return None

    try:
        # Resolve document-title-based ground truth IDs to chunk-level IDs
        doc_titles = item.get("relevant_doc_ids", [])
        ground_truth_ids = await _resolve_relevant_doc_ids(doc_titles)

        ground_truth = EvalGroundTruth(
            query_id=item.get("query_id", query[:50]),
            relevant_doc_ids=ground_truth_ids,
            reference_answer=item.get("expected_answer"),
        )

        # Get RAG response with timeout
        query_req = QueryRequest(question=query)
        rag_response = await asyncio.wait_for(
            rag_engine.query(query_req, session_manager),
            timeout=60.0  # 60s per query timeout
        )

        # Evaluate
        evaluator = RAGEvaluator(llm_generator=LLMGenerator())
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

    # Persist all benchmark results to disk (best-effort)
    result_dicts = [r.to_dict() for r in valid_results]
    for rd in result_dicts:
        await _save_result_to_disk(rd)
    _evaluation_history.extend(valid_results)

    return {
        "results": result_dicts,
        "total": len(valid_results),
    }


@router.get("/history")
async def get_evaluation_history(limit: int = 20) -> dict:
    """Get evaluation history."""
    # Load from disk on first access if memory is empty (cold start)
    if not _evaluation_history and EVALUATION_HISTORY_PATH.exists():
        _evaluation_history.extend(_load_history_from_disk())

    recent = _evaluation_history[-limit:] if _evaluation_history else []

    return {
        "history": [r.to_dict() for r in recent],
        "total": len(_evaluation_history),
    }