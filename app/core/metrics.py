from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RETRIEVAL_COUNT = Counter(
    "rag_retrieval_total",
    "Total number of retrievals",
    ["retriever_type"],
)

ERROR_COUNT = Counter(
    "rag_errors_total",
    "Total number of errors",
    ["error_type"],
)

LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "Total number of LLM tokens used",
    ["token_type"],
)

ACTIVE_QUERIES = Gauge(
    "rag_active_queries",
    "Number of currently active queries",
)

DOCUMENT_PROCESSING = Histogram(
    "rag_document_processing_seconds",
    "Document processing time in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

SESSION_COUNT = Gauge(
    "rag_sessions_total",
    "Total number of active sessions",
)

MESSAGE_COUNT = Counter(
    "rag_messages_total",
    "Total number of messages processed",
    ["role"],
)

# GPU metrics
GPU_MEMORY_TOTAL = Gauge(
    "rag_gpu_memory_total_mb",
    "Total GPU memory in MB",
)

GPU_MEMORY_USED = Gauge(
    "rag_gpu_memory_used_mb",
    "Used GPU memory in MB",
)

GPU_MEMORY_FREE = Gauge(
    "rag_gpu_memory_free_mb",
    "Free GPU memory in MB",
)

GPU_MEMORY_RESERVED = Gauge(
    "rag_gpu_memory_reserved_mb",
    "Reserved GPU memory in MB (PyTorch cache)",
)

GPU_MODELS_LOADED = Gauge(
    "rag_gpu_models_loaded",
    "Number of models loaded on GPU",
)

# Stage latency metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval stage latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

RERANK_LATENCY = Histogram(
    "rag_rerank_duration_seconds",
    "Reranking stage latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "LLM generation stage latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0],
)


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    _update_gpu_metrics()
    return generate_latest()


def get_content_type() -> str:
    """Get the content type for Prometheus metrics response."""
    return CONTENT_TYPE_LATEST


def _update_gpu_metrics() -> None:
    """Update GPU memory metrics from GPUMemoryManager."""
    try:
        from app.core.gpu_memory_manager import GPUMemoryManager

        manager = GPUMemoryManager.get_instance()
        info = manager.get_memory_info()
        models = manager.get_loaded_models()

        if info["available"]:
            GPU_MEMORY_TOTAL.set(info["total_mb"])
            GPU_MEMORY_USED.set(info["used_mb"])
            GPU_MEMORY_FREE.set(info["free_mb"])
            GPU_MEMORY_RESERVED.set(info.get("reserved_mb", 0))
        else:
            GPU_MEMORY_TOTAL.set(0)
            GPU_MEMORY_USED.set(0)
            GPU_MEMORY_FREE.set(0)
            GPU_MEMORY_RESERVED.set(0)

        GPU_MODELS_LOADED.set(len(models))
    except Exception as e:
        logger.debug(f"GPU metrics unavailable: {e}")
