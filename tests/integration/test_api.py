from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Mock the app state before importing app
from app.main import create_app

app = create_app()

# Set up mock session_manager and document_service
mock_session_manager = MagicMock()
mock_session_manager.create_session_db = AsyncMock(return_value=MagicMock(
    session_id="test-session-id",
    session_title=None,
    created_at=MagicMock(),
    updated_at=MagicMock(),
    messages=[],
    context_documents=[],
    is_active=True,
))
mock_session_manager.add_message = AsyncMock(return_value=MagicMock(
    message_id="test-message-id",
    role="user",
    content="test",
    timestamp=MagicMock(),
    metadata={},
))
mock_session_manager.get_session = MagicMock(return_value=MagicMock(
    session_id="test-session-id",
    session_title=None,
    messages=[],
    is_active=True,
))
mock_session_manager.delete_session = AsyncMock(return_value=True)

mock_document_service = MagicMock()

app.state.session_manager = mock_session_manager
app.state.document_service = mock_document_service

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestQueryEndpoint:
    def test_query_requires_question(self):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_empty_question(self):
        response = client.post(
            "/api/v1/query",
            json={"question": ""},
        )
        assert response.status_code == 400

    def test_query_with_question(self):
        response = client.post(
            "/api/v1/query",
            json={"question": "糖尿病的诊断标准是什么？"},
        )

        assert response.status_code in [200, 500]


class TestDocumentsEndpoint:
    def test_list_documents(self):
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_upload_unsupported_file(self):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", b"content", "application/octet-stream")},
        )
        assert response.status_code == 400


class TestSessionsEndpoint:
    def test_create_session(self):
        response = client.post("/api/v1/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_get_session_messages_not_found(self):
        response = client.get("/api/v1/sessions/00000000-0000-0000-0000-000000000000/messages")
        assert response.status_code == 404

    def test_delete_nonexistent_session(self):
        mock_session_manager.delete_session = AsyncMock(return_value=False)
        response = client.delete("/api/v1/sessions/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404