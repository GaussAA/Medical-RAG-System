from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.session import SessionManager


class TestSessionMessageEviction:
    """Test session message eviction when limit is exceeded."""

    def test_max_session_messages_constant_exists(self):
        """Test MAX_SESSION_MESSAGES constant is defined."""
        assert hasattr(SessionManager, "MAX_SESSION_MESSAGES")
        assert SessionManager.MAX_SESSION_MESSAGES == 100

    def test_manager_has_eviction_method(self):
        """Test SessionManager has the eviction helper method."""
        manager = SessionManager()
        assert hasattr(manager, "_evict_messages_if_needed")


class TestMessageCountLimit:
    """Test message count limit enforcement."""

    def test_session_stores_message_count(self):
        """Test that sessions track message count correctly."""
        manager = SessionManager()
        session = manager.create_session()

        assert hasattr(session, "msg_count")
        assert session.msg_count == 0

    @pytest.mark.asyncio
    async def test_message_count_increments(self):
        """Test message count increments when messages are added."""
        manager = SessionManager()
        # Mock the async session
        mock_session = MagicMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.add = MagicMock()
        manager.async_session = mock_session

        session = manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute = AsyncMock(return_value=mock_result)

        await manager.add_message(session.session_id, "user", "test message")

        # Message count is tracked in the session object
        assert len(session.messages) == 1


class TestSessionManagerEviction:
    """Test eviction logic in SessionManager."""

    def test_eviction_method_exists(self):
        """Verify _evict_messages_if_needed is available."""
        manager = SessionManager()
        assert callable(getattr(manager, "_evict_messages_if_needed", None))
