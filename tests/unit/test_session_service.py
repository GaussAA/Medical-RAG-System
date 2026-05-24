import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.session import SessionManager


class TestSessionManager:
    def setup_method(self):
        self.manager = SessionManager(max_history=5, max_context_length=1000)
        # Mock the _ensure_session to avoid database calls
        self.mock_session = MagicMock()
        self.mock_session.execute = AsyncMock()
        self.mock_session.commit = AsyncMock()
        self.mock_session.add = MagicMock()
        self.mock_session.close = MagicMock()
        self.manager.async_session = self.mock_session
        # Force _ensure_session to return our mock session
        self.manager._ensure_session = AsyncMock(return_value=self.mock_session)

    def test_create_session(self):
        session = self.manager.create_session()

        assert session.session_id is not None
        assert session.session_title is None
        assert len(session.messages) == 0
        assert session.is_active is True

    def test_get_session(self):
        created = self.manager.create_session()
        retrieved = self.manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_session_not_found(self):
        retrieved = self.manager.get_session("nonexistent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_add_message_user(self):
        session = self.manager.create_session()

        # Mock the database update
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        message = await self.manager.add_message(
            session_id=session.session_id,
            role="user",
            content="糖尿病有什么症状？",
        )

        assert message is not None
        assert message.role == "user"
        assert message.content == "糖尿病有什么症状？"

    @pytest.mark.asyncio
    async def test_add_message_sets_session_title(self):
        session = self.manager.create_session()
        assert session.session_title is None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(
            session_id=session.session_id,
            role="user",
            content="关于糖尿病的诊断标准是什么？",
        )

        updated = self.manager.get_session(session.session_id)
        assert updated.session_title == "关于糖尿病的诊断标准是什么？"

    @pytest.mark.asyncio
    async def test_add_message_truncates_long_title(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(
            session_id=session.session_id,
            role="user",
            content="A" * 100,
        )

        updated = self.manager.get_session(session.session_id)
        assert len(updated.session_title) == 53
        assert updated.session_title.endswith("...")

    @pytest.mark.asyncio
    async def test_add_message_assistant(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(
            session_id=session.session_id,
            role="user",
            content="糖尿病的症状是什么？",
        )
        assistant_msg = await self.manager.add_message(
            session_id=session.session_id,
            role="assistant",
            content="糖尿病的典型症状是多饮、多尿、多食和体重下降。",
        )

        assert assistant_msg.role == "assistant"

    @pytest.mark.asyncio
    async def test_add_message_session_not_found(self):
        message = await self.manager.add_message(
            session_id="nonexistent",
            role="user",
            content="test",
        )
        assert message is None

    @pytest.mark.asyncio
    async def test_get_messages(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(session.session_id, "user", "第一个问题")
        await self.manager.add_message(session.session_id, "assistant", "第一个回答")
        await self.manager.add_message(session.session_id, "user", "第二个问题")

        messages = self.manager.get_messages(session.session_id)

        assert len(messages) == 3
        assert messages[0].content == "第一个问题"
        assert messages[1].content == "第一个回答"
        assert messages[2].content == "第二个问题"

    def test_get_messages_empty(self):
        messages = self.manager.get_messages("nonexistent")
        assert messages == []

    @pytest.mark.asyncio
    async def test_delete_session(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        result = await self.manager.delete_session(session.session_id)

        assert result is True
        assert self.manager.get_session(session.session_id) is None

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        result = await self.manager.delete_session("nonexistent")
        assert result is False

    def test_build_context(self):
        session = self.manager.create_session()
        session.messages = [
            MagicMock(content="糖尿病的症状"),
            MagicMock(content="多饮多尿"),
        ]

        retrieved_docs = [
            {"source": "指南", "content": "糖尿病典型症状是多饮、多尿、多食、体重下降"}
        ]

        context = self.manager.build_context(
            session_id=session.session_id,
            current_query="糖尿病有什么症状？",
            retrieved_docs=retrieved_docs,
        )

        assert "糖尿病的症状" in context or "糖尿病有什么症状" in context
        assert "参考文档" in context
        assert "指南" in context

    def test_build_context_empty_session(self):
        context = self.manager.build_context(
            session_id="nonexistent",
            current_query="test",
            retrieved_docs=[],
        )
        assert context == ""

    def test_build_context_respects_max_length(self):
        session = self.manager.create_session()
        session.messages = [MagicMock(content="A" * 500)]

        retrieved_docs = [{"source": "doc", "content": "B" * 2000}]

        context = self.manager.build_context(
            session_id=session.session_id,
            current_query="test",
            retrieved_docs=retrieved_docs,
        )

        assert self.manager._count_tokens(context) <= self.manager.max_context_length


class TestSessionTitleGeneration:
    def setup_method(self):
        self.manager = SessionManager()
        self.mock_session = MagicMock()
        self.mock_session.execute = AsyncMock()
        self.mock_session.commit = AsyncMock()
        self.mock_session.add = MagicMock()
        self.manager.async_session = self.mock_session

    @pytest.mark.asyncio
    async def test_first_user_message_becomes_title(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(
            session_id=session.session_id,
            role="user",
            content="我想了解糖尿病的诊断标准",
        )

        updated = self.manager.get_session(session.session_id)
        assert updated.session_title == "我想了解糖尿病的诊断标准"

    @pytest.mark.asyncio
    async def test_first_assistant_message_does_not_set_title(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(session.session_id, "assistant", "有什么可以帮您？")

        updated = self.manager.get_session(session.session_id)
        assert updated.session_title is None

    @pytest.mark.asyncio
    async def test_title_not_overwritten(self):
        session = self.manager.create_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        await self.manager.add_message(session.session_id, "user", "第一个问题")
        await self.manager.add_message(session.session_id, "user", "第二个问题")

        updated = self.manager.get_session(session.session_id)
        assert updated.session_title == "第一个问题"