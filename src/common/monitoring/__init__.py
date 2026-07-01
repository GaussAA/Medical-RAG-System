"""Prometheus metrics and monitoring endpoints."""

from src.common.monitoring.api import router as metrics_router
from src.common.monitoring.metrics import (
    ACTIVE_QUERIES,
    ERROR_COUNT,
    GENERATION_LATENCY,
    GPU_MEMORY_FREE,
    GPU_MEMORY_RESERVED,
    GPU_MEMORY_TOTAL,
    GPU_MEMORY_USED,
    GPU_MODELS_LOADED,
    LLM_TOKENS,
    QUERY_LATENCY,
    RERANK_LATENCY,
    RETRIEVAL_COUNT,
    RETRIEVAL_LATENCY,
    REGISTRY,
    get_content_type,
    get_metrics,
)

__all__ = [
    "REGISTRY",
    "GPU_MEMORY_TOTAL",
    "GPU_MEMORY_USED",
    "GPU_MEMORY_FREE",
    "GPU_MEMORY_RESERVED",
    "GPU_MODELS_LOADED",
    "ACTIVE_QUERIES",
    "QUERY_LATENCY",
    "RETRIEVAL_COUNT",
    "RETRIEVAL_LATENCY",
    "RERANK_LATENCY",
    "GENERATION_LATENCY",
    "LLM_TOKENS",
    "ERROR_COUNT",
    "get_metrics",
    "get_content_type",
    "metrics_router",
]
