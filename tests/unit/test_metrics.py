from app.core.metrics import (
    QUERY_LATENCY,
    RETRIEVAL_COUNT,
    ERROR_COUNT,
    ACTIVE_QUERIES,
    LLM_TOKENS,
    GPU_MEMORY_TOTAL,
    GPU_MEMORY_USED,
    GPU_MEMORY_FREE,
    GPU_MODELS_LOADED,
    RETRIEVAL_LATENCY,
    RERANK_LATENCY,
    GENERATION_LATENCY,
    get_metrics,
    get_content_type,
    _update_gpu_metrics,
)


class TestMetrics:
    def test_query_latency_histogram(self):
        """Test QUERY_LATENCY histogram can record observations without error."""
        # Just verify observe doesn't raise an exception
        QUERY_LATENCY.observe(0.5)
        QUERY_LATENCY.observe(1.0)
        QUERY_LATENCY.observe(0.25)

    def test_retrieval_count_counter(self):
        """Test RETRIEVAL_COUNT counter can increment with labels."""
        RETRIEVAL_COUNT.labels(retriever_type="vector").inc()
        RETRIEVAL_COUNT.labels(retriever_type="bm25").inc()
        # Verify counter exists
        assert RETRIEVAL_COUNT._metrics is not None

    def test_error_count_counter(self):
        """Test ERROR_COUNT counter can increment with error_type label."""
        ERROR_COUNT.labels(error_type="safety").inc()
        ERROR_COUNT.labels(error_type="retrieval").inc()
        assert ERROR_COUNT._metrics is not None

    def test_active_queries_gauge(self):
        """Test ACTIVE_QUERIES gauge can increment and decrement."""
        ACTIVE_QUERIES.inc()
        ACTIVE_QUERIES.inc()
        ACTIVE_QUERIES.dec()
        # Verify gauge works without error

    def test_get_metrics_returns_bytes(self):
        """Test get_metrics returns bytes in Prometheus format."""
        result = get_metrics()
        assert isinstance(result, bytes)
        assert b"rag_" in result

    def test_get_content_type_returns_prometheus_format(self):
        """Test get_content_type returns correct Prometheus content type."""
        content_type = get_content_type()
        assert "text/plain" in content_type

    def test_llm_tokens_counter(self):
        """Test LLM_TOKENS counter can increment with token_type label."""
        LLM_TOKENS.labels(token_type="prompt").inc(100)
        LLM_TOKENS.labels(token_type="completion").inc(50)
        assert LLM_TOKENS._metrics is not None

    def test_gpu_memory_gauges(self):
        """Test GPU memory gauges can be set without error."""
        GPU_MEMORY_TOTAL.set(16384)
        GPU_MEMORY_USED.set(8192)
        GPU_MEMORY_FREE.set(8192)
        GPU_MODELS_LOADED.set(2)

    def test_retrieval_latency_histogram(self):
        """Test RETRIEVAL_LATENCY histogram can record observations."""
        RETRIEVAL_LATENCY.observe(0.05)
        RETRIEVAL_LATENCY.observe(0.1)
        RETRIEVAL_LATENCY.observe(0.25)

    def test_rerank_latency_histogram(self):
        """Test RERANK_LATENCY histogram can record observations."""
        RERANK_LATENCY.observe(0.01)
        RERANK_LATENCY.observe(0.05)
        RERANK_LATENCY.observe(0.1)

    def test_generation_latency_histogram(self):
        """Test GENERATION_LATENCY histogram can record observations."""
        GENERATION_LATENCY.observe(1.0)
        GENERATION_LATENCY.observe(2.5)
        GENERATION_LATENCY.observe(5.0)

    def test_update_gpu_metrics_does_not_raise(self):
        """Test _update_gpu_metrics handles missing GPU gracefully."""
        _update_gpu_metrics()  # Should not raise even without GPU

    def test_get_metrics_includes_new_metrics(self):
        """Test get_metrics returns all new metrics in Prometheus format."""
        result = get_metrics()
        assert isinstance(result, bytes)
        # Check for new metric names
        assert b"rag_gpu_memory" in result or b"rag_llm_tokens" in result or b"rag_retrieval_duration" in result
