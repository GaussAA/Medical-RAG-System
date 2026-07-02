# =============================================================================
# Medical RAG System — Multi-stage Dockerfile
# =============================================================================

# ---- Builder stage ----
FROM python:3.12-slim AS builder

ENV UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency manifests
COPY pyproject.toml uv.lock ./

# Install runtime deps (no dev deps)
RUN uv sync --no-dev --frozen

# ---- Runtime stage ----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY pyproject.toml .

# Create data directories (mounted volume will override)
RUN mkdir -p data/raw_documents data/processed data/cache data/database

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Start uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
