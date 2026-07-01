"""Prometheus metrics - migrated from app/core/metrics.py.

Uses a custom CollectorRegistry to avoid conflicts with the legacy
app/core/metrics.py during the migration transition.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

# === GPU metrics ===

GPU_MEMORY_TOTAL: Gauge = Gauge("gpu_memory_total_mb", "Total GPU memory in MB", registry=REGISTRY)
GPU_MEMORY_USED: Gauge = Gauge("gpu_memory_used_mb", "Used GPU memory in MB", registry=REGISTRY)
GPU_MEMORY_FREE: Gauge = Gauge("gpu_memory_free_mb", "Free GPU memory in MB", registry=REGISTRY)
GPU_MEMORY_RESERVED: Gauge = Gauge("gpu_memory_reserved_mb", "Reserved GPU memory in MB", registry=REGISTRY)
GPU_MODELS_LOADED: Gauge = Gauge("gpu_models_loaded", "Number of models loaded on GPU", registry=REGISTRY)

# === Query metrics ===

ACTIVE_QUERIES: Gauge = Gauge("rag_active_queries", "Number of queries currently being processed", registry=REGISTRY)
QUERY_LATENCY: Histogram = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)

# === Retrieval metrics ===

RETRIEVAL_COUNT: Counter = Counter(
    "rag_retrieval_total",
    "Total number of retrieval operations",
    ["retriever_type"],
    registry=REGISTRY,
)
RETRIEVAL_LATENCY: Histogram = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY,
)

# === Rerank metrics ===

RERANK_LATENCY: Histogram = Histogram(
    "rag_rerank_latency_seconds",
    "Reranking latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY,
)

# === Generation metrics ===

GENERATION_LATENCY: Histogram = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY,
)
LLM_TOKENS: Counter = Counter(
    "rag_llm_tokens_total",
    "Total number of LLM tokens used",
    ["token_type"],
    registry=REGISTRY,
)

# === Error metrics ===

ERROR_COUNT: Counter = Counter(
    "rag_errors_total",
    "Total number of errors by type",
    ["error_type"],
    registry=REGISTRY,
)


def get_metrics() -> bytes:
    """Generate Prometheus metrics output using the custom registry."""
    _update_gpu_metrics()
    return generate_latest(REGISTRY)


def get_content_type() -> str:
    """Get the content type for Prometheus metrics response."""
    return CONTENT_TYPE_LATEST


def _update_gpu_metrics() -> None:
    """Update GPU memory metrics from torch."""
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            allocated = torch.cuda.memory_allocated(0) / (1024**2)
            reserved = torch.cuda.memory_reserved(0) / (1024**2)
            GPU_MEMORY_TOTAL.set(total)
            GPU_MEMORY_USED.set(allocated)
            GPU_MEMORY_FREE.set(total - reserved)
            GPU_MEMORY_RESERVED.set(reserved)
        else:
            GPU_MEMORY_TOTAL.set(0)
            GPU_MEMORY_USED.set(0)
            GPU_MEMORY_FREE.set(0)
            GPU_MEMORY_RESERVED.set(0)

        GPU_MODELS_LOADED.set(2)
    except Exception:
        pass
