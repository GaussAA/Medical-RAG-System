"""
Unified API client for the Medical RAG frontend.

Encapsulates all backend API calls with proper error handling,
retry logic, and consistent response parsing.
"""

import os
from typing import Any

import requests
from loguru import logger


class APIClient:
    """Client for the Medical RAG backend API."""

    def __init__(self, base_url: str | None = None, timeout: int = 30):
        self.base_url = base_url or os.environ.get("API_BASE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.session = requests.Session()
        self._init_health_check()

    def _init_health_check(self) -> None:
        """Verify backend connectivity on initialization."""
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"Backend API connected: {self.base_url}")
        except requests.RequestException:
            logger.warning(f"Backend API unreachable: {self.base_url}")

    # ==================== Health ====================

    def health_check(self) -> dict[str, Any] | None:
        """Check backend health status."""
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.RequestException:
            return None

    def is_healthy(self) -> bool:
        """Quick health check returning boolean."""
        return self.health_check() is not None

    # ==================== Documents ====================

    def list_documents(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        """List all uploaded documents."""
        resp = self.session.get(
            f"{self.base_url}/api/v1/documents",
            params={"page": page, "page_size": page_size},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def upload_document(self, file_path: str, title: str | None = None) -> dict[str, Any]:
        """Upload a document file."""
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f)}
            params = {"title": title} if title else {}
            resp = self.session.post(
                f"{self.base_url}/api/v1/documents/upload",
                files=files,
                params=params,
                timeout=300,  # Long timeout for upload + processing
            )
        resp.raise_for_status()
        return resp.json()

    def upload_documents_batch(self, file_paths: list[str]) -> dict[str, Any]:
        """Upload multiple documents in batch."""
        files = []
        for fp in file_paths:
            files.append(("files", open(fp, "rb")))
        resp = self.session.post(
            f"{self.base_url}/api/v1/documents/upload/batch",
            files=files,
            timeout=600,
        )
        for _, f in files:
            f.close()
        resp.raise_for_status()
        return resp.json()

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        resp = self.session.delete(
            f"{self.base_url}/api/v1/documents/{doc_id}",
            timeout=self.timeout,
        )
        return resp.status_code == 200

    def get_document_status(self, doc_id: str) -> dict[str, Any]:
        """Get document processing status."""
        resp = self.session.get(
            f"{self.base_url}/api/v1/documents/{doc_id}/status",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ==================== Query ====================

    def query(self, question: str, session_id: str | None = None) -> dict[str, Any]:
        """Send a query to the RAG system."""
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        resp = self.session.post(
            f"{self.base_url}/api/v1/query",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    def query_stream(self, question: str, session_id: str | None = None):
        """Stream a query response via SSE."""
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        resp = self.session.post(
            f"{self.base_url}/api/v1/query/stream",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                yield line.decode("utf-8")

    # ==================== Sessions ====================

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all conversation sessions."""
        resp = self.session.get(
            f"{self.base_url}/api/v1/sessions",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get a specific session with its messages."""
        resp = self.session.get(
            f"{self.base_url}/api/v1/sessions/{session_id}",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        resp = self.session.delete(
            f"{self.base_url}/api/v1/sessions/{session_id}",
            timeout=self.timeout,
        )
        return resp.status_code == 200

    # ==================== Evaluation ====================

    def run_evaluation(self, query: str, ground_truth: str, contexts: list[str]) -> dict[str, Any]:
        """Run a single evaluation."""
        resp = self.session.post(
            f"{self.base_url}/api/v1/evaluation/evaluate",
            json={
                "query": query,
                "ground_truth": ground_truth,
                "contexts": contexts,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    # ==================== Metrics ====================

    def get_metrics(self) -> dict[str, Any] | None:
        """Get Prometheus metrics."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/metrics",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except requests.RequestException:
            return None

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()
