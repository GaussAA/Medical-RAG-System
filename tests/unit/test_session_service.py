import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.session import SessionManager
from app.models.schemas import Message


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


class TestCountTokens:
    """Tests for _count_tokens pure logic method."""

    def setup_method(self):
        self.manager = SessionManager()

    def test_english_text(self):
        """English text: each ~4 chars = 1 token."""
        text = "hello world, this is a test"
        result = self.manager._count_tokens(text)
        assert result == len(text) // 4

    def test_chinese_text(self):
        """Chinese text: len in characters, // 4."""
        text = "糖尿病典型症状是多饮多尿"
        expected = len(text) // 4
        assert self.manager._count_tokens(text) == expected

    def test_mixed_text(self):
        """Mixed Chinese and English text."""
        text = "Diabetes症状包括polyuria"
        expected = len(text) // 4
        assert self.manager._count_tokens(text) == expected

    def test_empty_string(self):
        """Empty string = 0 tokens."""
        assert self.manager._count_tokens("") == 0

    def test_short_string(self):
        """String shorter than 4 chars rounds down to 0."""
        assert self.manager._count_tokens("abc") == 0


class TestFormatDocuments:
    """Tests for _format_documents pure logic method."""

    def setup_method(self):
        self.manager = SessionManager()

    def test_empty_docs(self):
        """Empty list returns placeholder."""
        result = self.manager._format_documents([])
        assert result == "无可用参考文档"

    def test_single_doc(self):
        """Single document formatted with index and source."""
        docs = [{"source": "糖尿病指南.pdf", "content": "糖尿病典型症状是多饮、多尿、多食、体重下降"}]
        result = self.manager._format_documents(docs)
        assert "[1] 糖尿病指南.pdf:" in result
        assert "糖尿病典型症状" in result

    def test_multiple_docs(self):
        """Multiple documents each get their own index."""
        docs = [
            {"source": "指南A", "content": "内容A"},
            {"source": "指南B", "content": "内容B"},
        ]
        result = self.manager._format_documents(docs)
        assert "[1] 指南A:" in result
        assert "[2] 指南B:" in result
        assert "内容A" in result
        assert "内容B" in result

    def test_doc_truncates_long_content(self):
        """Content over 200 chars gets truncated with '...'."""
        long_content = "X" * 300
        docs = [{"source": "test", "content": long_content}]
        result = self.manager._format_documents(docs)
        # Should only show first 200 chars + "..."
        assert long_content[:200] in result
        assert len(long_content) not in [len(r) for r in result.split("\n")]
        assert "..." in result
        # Full 300-char content should NOT appear
        assert long_content not in result

    def test_doc_missing_source_uses_default(self):
        """Doc without 'source' key uses '未知来源'."""
        docs = [{"content": "some content"}]
        result = self.manager._format_documents(docs)
        assert "未知来源" in result

    def test_doc_missing_content_shows_empty(self):
        """Doc without 'content' key uses empty string."""
        docs = [{"source": "guide.pdf"}]
        result = self.manager._format_documents(docs)
        assert "[1] guide.pdf:\n..." in result


class TestFormatHistory:
    """Tests for _format_history and format_history_message integration."""

    def setup_method(self):
        self.manager = SessionManager()

    def test_empty_messages(self):
        """Empty message list returns empty string."""
        result = self.manager._format_history([])
        assert result == ""

    def test_single_user_message(self):
        """User message formatted with role label and sanitized content."""
        msg = Message(role="user", content="糖尿病的症状是什么？")
        result = self.manager._format_history([msg])
        assert "**用户**:" in result
        assert "糖尿病的症状是什么？" in result

    def test_single_assistant_message(self):
        """Assistant message formatted with role label."""
        msg = Message(role="assistant", content="糖尿病典型症状是多饮多尿")
        result = self.manager._format_history([msg])
        assert "**助手**:" in result
        assert "糖尿病典型症状是多饮多尿" in result

    def test_multiple_messages(self):
        """Multiple messages joined by double newline."""
        msgs = [
            Message(role="user", content="问题1"),
            Message(role="assistant", content="回答1"),
            Message(role="user", content="问题2"),
        ]
        result = self.manager._format_history(msgs)
        assert result.count("\n\n") == 2  # 3 messages → 2 separators
        assert "**用户**: 问题1" in result
        assert "**助手**: 回答1" in result

    def test_markdown_sanitization(self):
        """format_history_message sanitizes markdown special characters."""
        msg = Message(role="user", content="**bold** and `code` and [link] and #header")
        result = self.manager._format_history([msg])
        # Markdown chars should be stripped
        assert "**bold**" not in result
        assert "`code`" not in result
        assert "[link]" not in result
        assert "#header" not in result
        # Content words should remain
        assert "bold" in result
        assert "code" in result
        assert "link" in result
        assert "header" in result


class TestFilterRelevantHistory:
    """Tests for _filter_relevant_history pure logic method."""

    def setup_method(self):
        self.manager = SessionManager(max_history=10)

    def test_empty_messages(self):
        """Empty messages returns empty list."""
        result = self.manager._filter_relevant_history("query", [])
        assert result == []

    def test_relevant_message_by_term_overlap(self):
        """Message sharing terms with query is included."""
        msgs = [
            Message(role="user", content="diabetes symptoms treatment"),
        ]
        result = self.manager._filter_relevant_history("diabetes symptoms", msgs)
        assert len(result) == 1
        assert result[0].content == "diabetes symptoms treatment"

    def test_irrelevant_message_filtered_out(self):
        """Message with no term overlap is excluded."""
        msgs = [
            Message(role="user", content="hello world"),
        ]
        result = self.manager._filter_relevant_history("diabetes symptoms", msgs)
        # No overlap → score 0 → below threshold → falls back to last 3
        assert len(result) == 1  # only 1 message total, so last 3 = 1
        assert result[0].content == "hello world"

    def test_fallback_to_last_three_when_none_relevant(self):
        """When no messages exceed threshold, fallback to last 3."""
        msgs = [
            Message(role="user", content="a b c"),
            Message(role="assistant", content="d e f"),
            Message(role="user", content="g h i"),
            Message(role="assistant", content="j k l"),
            Message(role="user", content="m n o"),
        ]
        result = self.manager._filter_relevant_history("x y z", msgs)
        # No overlap → fallback to last 3
        assert len(result) == 3
        assert result[0].content == "g h i"
        assert result[2].content == "m n o"

    def test_respects_max_history_window(self):
        """Only last max_history messages are scored."""
        manager = SessionManager(max_history=3)
        msgs = [
            Message(role="user", content="diabetes symptoms"),     # old, outside window
            Message(role="assistant", content="common signs"),     # old, outside window
            Message(role="user", content="x y z"),                 # inside window
            Message(role="assistant", content="no match here"),    # inside window
            Message(role="user", content="diabetes again"),        # inside window, relevant
        ]
        result = manager._filter_relevant_history("diabetes symptoms", msgs)
        # Only last 3 scored. "diabetes again" has overlap with "diabetes" (1/2=0.5 >= 0.3)
        # "x y z" has no overlap (0/2). "no match here" has no overlap.
        # So only 1 relevant → but wait: if at least one is relevant, only those are returned
        assert len(result) == 1
        assert result[0].content == "diabetes again"

    def test_partial_overlap_scoring(self):
        """Message with partial term overlap gets scored proportionally."""
        msgs = [
            Message(role="user", content="diabetes only"),
            Message(role="user", content="diabetes symptoms treatment"),
        ]
        # Query: "diabetes symptoms causes" → 3 terms
        # msg1: {"diabetes", "only"} ∩ {"diabetes", "symptoms", "causes"} = {"diabetes"} → 1/3 = 0.33 >= 0.3
        # msg2: {"diabetes", "symptoms", "treatment"} ∩ {"diabetes", "symptoms", "causes"} = {"diabetes", "symptoms"} → 2/3 = 0.67 >= 0.3
        result = self.manager._filter_relevant_history("diabetes symptoms causes", msgs)
        assert len(result) == 2

    def test_case_insensitive_matching(self):
        """Term matching is case-insensitive."""
        msgs = [
            Message(role="user", content="DIABETES Symptoms"),
        ]
        result = self.manager._filter_relevant_history("diabetes symptoms", msgs)
        assert len(result) == 1

    def test_exact_threshold_boundary(self):
        """Message at exact threshold of 0.3 is included."""
        # Query has 10 unique terms, overlap of 3 → 3/10 = 0.3 → included
        msgs = [
            Message(role="user", content="a b c"),
        ]
        result = self.manager._filter_relevant_history("a b c d e f g h i j", msgs)
        assert len(result) == 1


class TestTruncateContext:
    """Tests for _truncate_context pure logic method."""

    def setup_method(self):
        self.manager = SessionManager(max_context_length=10)
        # max_chars = max_context_length * 4 = 40

    def test_context_under_limit(self):
        """Context shorter than max_chars is returned unchanged."""
        short = "short context"
        result = self.manager._truncate_context(short)
        assert result == short

    def test_context_at_exact_limit(self):
        """Context exactly at max_chars is returned unchanged."""
        exact = "A" * 40  # max_context_length(10) * 4 = 40
        result = self.manager._truncate_context(exact)
        assert result == exact
        assert "\n\n...(上下文已截断)" not in result

    def test_context_over_limit(self):
        """Context over max_chars is truncated with suffix."""
        long_text = "A" * 100
        result = self.manager._truncate_context(long_text)
        assert len(long_text) > 40  # should be truncated
        assert result.endswith("\n\n...(上下文已截断)")
        assert len(result) == 40 + len("\n\n...(上下文已截断)")

    def test_context_just_one_over(self):
        """Context 1 char over limit gets truncated."""
        over_by_one = "A" * 41
        result = self.manager._truncate_context(over_by_one)
        assert result.endswith("\n\n...(上下文已截断)")


class TestBuildContextDetailed:
    """Detailed tests for build_context using real Message objects."""

    def setup_method(self):
        self.manager = SessionManager(max_history=5, max_context_length=100)

    def test_returns_empty_for_nonexistent_session(self):
        """Build context returns '' when session does not exist."""
        result = self.manager.build_context("nonexistent", "query", [])
        assert result == ""

    def test_includes_history_and_docs_sections(self):
        """Context includes history and reference documents sections."""
        session = self.manager.create_session()
        session.messages = [
            Message(role="user", content="糖尿病有什么症状？"),
            Message(role="assistant", content="典型症状包括多饮、多尿、多食、体重下降。"),
        ]
        docs = [{"source": "指南.pdf", "content": "糖尿病诊断标准：空腹血糖>=7.0mmol/L"}]

        context = self.manager.build_context(session.session_id, "糖尿病诊断", docs)

        assert "**用户**:" in context  # history formatted
        assert "## 参考文档" in context
        assert "指南.pdf" in context

    def test_handles_missing_session(self):
        """When session exists in sessions dict but has no id in messages."""
        result = self.manager.build_context("nonexistent", "query", [])
        assert result == ""

    def test_with_empty_messages_and_empty_docs(self):
        """Context with empty messages and empty docs."""
        session = self.manager.create_session()
        session.messages = []

        context = self.manager.build_context(session.session_id, "query", [])
        # No history section (empty), but docs placeholder
        assert "无可用参考文档" in context

    def test_with_empty_messages_and_docs(self):
        """Context with empty messages but some docs."""
        session = self.manager.create_session()
        session.messages = []

        docs = [{"source": "guide.pdf", "content": "important content here"}]
        context = self.manager.build_context(session.session_id, "query", docs)
        assert "guide.pdf" in context
        assert "important content here" in context

    def test_relevant_history_included(self):
        """History with relevant messages is included in context."""
        session = self.manager.create_session()
        session.messages = [
            Message(role="user", content="diabetes symptoms causes treatment"),
            Message(role="assistant", content="here are the diabetes symptoms"),
            Message(role="user", content="unrelated topic here"),
        ]

        context = self.manager.build_context(
            session.session_id,
            "diabetes symptoms",
            [],
        )
        # "diabetes symptoms causes treatment" has high overlap
        assert "diabetes symptoms causes treatment" in context
        # "here are the diabetes symptoms" also has overlap ("diabetes", "symptoms")
        assert "here are the diabetes symptoms" in context

    def test_history_format_messages_joined(self):
        """Multiple history messages appear as formatted lines."""
        session = self.manager.create_session()
        session.messages = [
            Message(role="user", content="question one"),
            Message(role="assistant", content="answer one"),
        ]

        context = self.manager.build_context(session.session_id, "question one", [])
        assert "**用户**: question one" in context
        assert "**助手**: answer one" in context

    def test_context_truncation_triggers(self):
        """When context exceeds max_context_length, truncation is applied."""
        manager = SessionManager(max_history=5, max_context_length=5)
        # max_chars = 5 * 4 = 20, which is tiny
        session = manager.create_session()
        session.messages = [
            Message(role="user", content="hello world this is a long message"),
        ]
        docs = [{"source": "doc", "content": "very long document content that exceeds limits"}]

        context = manager.build_context(session.session_id, "hello world", docs)
        # Should be truncated
        assert "...(上下文已截断)" in context

    def test_no_truncation_when_under_limit(self):
        """Context under token limit is NOT truncated."""
        manager = SessionManager(max_history=5, max_context_length=5000)
        session = manager.create_session()
        session.messages = [
            Message(role="user", content="short"),
        ]
        context = manager.build_context(session.session_id, "query", [])
        assert "...(上下文已截断)" not in context


class TestSessionManagerEdgeCases:
    """Tests for edge cases requiring mocked DB interactions."""

    def setup_method(self):
        self.manager = SessionManager(max_history=5, max_context_length=1000)
        self.mock_session = MagicMock()
        self.mock_session.execute = AsyncMock()
        self.mock_session.commit = AsyncMock()
        self.mock_session.add = MagicMock()
        self.mock_session.close = AsyncMock()
        self.manager.async_session = self.mock_session
        self.manager._ensure_session = AsyncMock(return_value=self.mock_session)

    @pytest.mark.asyncio
    async def test_close_when_session_created_here(self):
        """close() closes and resets when _session_created_here is True."""
        manager = SessionManager()
        mock_sess = AsyncMock()
        manager.async_session = mock_sess
        manager._session_created_here = True
        await manager.close()
        mock_sess.close.assert_awaited_once()
        assert manager.async_session is None
        assert manager._session_created_here is False

    @pytest.mark.asyncio
    async def test_close_when_session_not_created_here(self):
        """close() does NOT close when _session_created_here is False."""
        manager = SessionManager()
        mock_sess = AsyncMock()
        manager.async_session = mock_sess
        manager._session_created_here = False
        await manager.close()
        mock_sess.close.assert_not_awaited()
        assert manager.async_session is mock_sess  # unchanged

    @pytest.mark.asyncio
    async def test_close_when_no_session(self):
        """close() handles None async_session gracefully."""
        manager = SessionManager()
        manager.async_session = None
        manager._session_created_here = True
        # Should not raise
        await manager.close()

    @pytest.mark.asyncio
    async def test_add_message_with_db_confirmed(self):
        """add_message writes to DB when session has db_confirmed=True."""
        session = self.manager.create_session()
        session.db_confirmed = True

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        message = await self.manager.add_message(
            session_id=session.session_id,
            role="user",
            content="test message",
        )

        assert message is not None
        # DB session.add() should have been called (Message insert)
        self.mock_session.add.assert_called()
        # DB session.commit() should have been called (Message + Conversation update)
        assert self.mock_session.commit.await_count >= 1

    @pytest.mark.asyncio
    async def test_add_message_from_db_session_loads(self):
        """add_message loads session from DB when not in memory."""
        # Use a UUID that doesn't exist in sessions dict
        import uuid as uuid_mod
        test_id = str(uuid_mod.uuid4())

        # Mock the DB returning a Conversation row
        mock_conv = MagicMock()
        mock_conv.id = uuid_mod.UUID(test_id)
        mock_conv.is_active = True

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=mock_conv)
        self.mock_session.execute = AsyncMock(return_value=mock_result)

        message = await self.manager.add_message(
            session_id=test_id,
            role="user",
            content="loaded from db",
        )

        assert message is not None
        # Session should now be in memory
        assert test_id in self.manager.sessions

    @pytest.mark.asyncio
    async def test_evict_messages_when_over_limit(self):
        """Messages are evicted when count exceeds MAX_SESSION_MESSAGES."""
        session = self.manager.create_session()
        # Create 102 messages (51 user-assistant pairs)
        msgs = []
        for i in range(51):
            msgs.append(Message(role="user", content=f"question {i}"))
            msgs.append(Message(role="assistant", content=f"answer {i}"))
        session.messages = msgs
        session.msg_count = len(msgs)

        assert len(session.messages) > self.manager.MAX_SESSION_MESSAGES

        await self.manager._evict_messages_if_needed(
            session.session_id, self.mock_session
        )

        # After eviction, message count should be <= MAX
        assert len(session.messages) <= self.manager.MAX_SESSION_MESSAGES
        assert session.msg_count == len(session.messages)

    @pytest.mark.asyncio
    async def test_evict_messages_under_limit_no_op(self):
        """No eviction when message count is under limit."""
        session = self.manager.create_session()
        session.messages = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
        ]
        session.msg_count = 2

        original_len = len(session.messages)
        await self.manager._evict_messages_if_needed(
            session.session_id, self.mock_session
        )

        assert len(session.messages) == original_len

    @pytest.mark.asyncio
    async def test_evict_messages_nonexistent_session(self):
        """_evict_messages_if_needed handles nonexistent session gracefully."""
        # Should not raise
        await self.manager._evict_messages_if_needed("nonexistent", self.mock_session)

    @pytest.mark.asyncio
    async def test_ensure_session_creates_factory_session(self):
        """_ensure_session creates session via factory when async_session is None."""
        manager = SessionManager()
        manager.async_session = None
        manager._session_created_here = False

        with patch("app.services.session.get_session_factory") as mock_factory:
            mock_session = MagicMock()
            # get_session_factory() returns a callable (the factory),
            # _ensure_session calls factory() which returns the session
            mock_factory.return_value = lambda: mock_session
            result = await manager._ensure_session()
            assert result is mock_session
            assert manager._session_created_here is True

    def test_get_session_returns_none_for_missing(self):
        """get_session returns None for nonexistent id."""
        assert self.manager.get_session("no-such-id") is None

    @pytest.mark.asyncio
    async def test_delete_session_invalid_uuid(self):
        """delete_session returns False for invalid UUID."""
        result = await self.manager.delete_session("not-a-valid-uuid!!!")
        assert result is False