# tests/integration/test_full_pipeline.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import create_app
from app.models.schemas import QueryResponse, Citation, RiskWarning


@pytest.fixture
def mock_app_state():
    """Create a mock app state with required services."""
    mock_session_manager = MagicMock()
    mock_session_manager.create_session_db = AsyncMock(return_value=MagicMock(
        session_id="test-session-id",
        session_title=None,
        messages=[],
        is_active=True,
    ))
    mock_session_manager.add_message = AsyncMock(return_value=MagicMock(
        message_id="test-message-id",
        role="user",
        content="test",
    ))
    mock_session_manager.get_session = MagicMock(return_value=MagicMock(
        session_id="test-session-id",
        messages=[],
        is_active=True,
    ))

    mock_document_service = MagicMock()
    mock_document_service.process_document = AsyncMock(return_value=True)
    mock_document_service.init_document = AsyncMock(return_value=None)

    return mock_session_manager, mock_document_service


@pytest.fixture
async def client(mock_app_state):
    """Create async test client with mocked app state."""
    mock_session_manager, mock_document_service = mock_app_state

    app = create_app()
    app.state.session_manager = mock_session_manager
    app.state.document_service = mock_document_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_full_query_pipeline(client):
    """Test complete flow: upload -> index -> query -> generate."""
    # Mock document processing
    with patch("app.services.document.DocumentService.process_document",
               new_callable=AsyncMock, return_value=True):
        doc_response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.md", b"# Test\n\nMedical content here.", "text/markdown")},
        )
    assert doc_response.status_code == 200

    # Mock query response
    with patch("app.core.rag_engine.RAGEngine.query",
               new_callable=AsyncMock) as mock_query:
        mock_query.return_value = QueryResponse(
            answer="Test answer",
            confidence=0.9,
            citations=[],
            warnings=[],
            session_id="test-session",
            processing_time=0.1,
        )
        query_response = await client.post(
            "/api/v1/query",
            json={"question": "What is the medical content?"},
        )
    assert query_response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health endpoint returns OK."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_query_with_citations(client):
    """Test query endpoint returns citations properly."""
    mock_citations = [
        Citation(
            source_id="chunk-1",
            document_id="doc-1",
            file_name="test.md",
            chunk_content="Medical content here",
            relevance_score=0.95,
            position="direct",
            verified=True,
        )
    ]

    with patch("app.core.rag_engine.RAGEngine.query",
               new_callable=AsyncMock) as mock_query:
        mock_query.return_value = QueryResponse(
            answer="Based on the document, medical content is important.",
            confidence=0.85,
            citations=mock_citations,
            warnings=[],
            session_id="test-session",
            processing_time=0.15,
        )
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is the medical content about?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "citations" in data
    assert len(data["citations"]) == 1
    assert data["citations"][0]["file_name"] == "test.md"


@pytest.mark.asyncio
async def test_query_with_warnings(client):
    """Test query endpoint returns risk warnings properly."""
    mock_warnings = [
        RiskWarning(
            type="medication",
            message="Content mentions medication information",
            priority="medium",
        )
    ]

    with patch("app.core.rag_engine.RAGEngine.query",
               new_callable=AsyncMock) as mock_query:
        mock_query.return_value = QueryResponse(
            answer="Some medication-related answer.",
            confidence=0.75,
            citations=[],
            warnings=mock_warnings,
            session_id="test-session",
            processing_time=0.12,
        )
        response = await client.post(
            "/api/v1/query",
            json={"question": "Tell me about aspirin"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    assert len(data["warnings"]) == 1
    assert data["warnings"][0]["type"] == "medication"


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_file_type(client):
    """Test that upload rejects unsupported file types."""
    response = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.pdf", b"PDF content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_query_requires_question(client):
    """Test that query endpoint validates question field."""
    response = await client.post(
        "/api/v1/query",
        json={},
    )
    assert response.status_code == 422  # Validation error

    response = await client.post(
        "/api/v1/query",
        json={"question": ""},
    )
    assert response.status_code == 400  # Empty string not allowed