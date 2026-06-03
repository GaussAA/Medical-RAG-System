import uuid
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.models.database import Conversation, Message
from app.models.schemas import ConversationSession
from app.models.schemas import Message as MessageSchema
from rag.generation.prompt import format_history_message


class SessionManager:
    MAX_SESSION_MESSAGES = 100

    def __init__(
        self,
        max_history: int = 10,
        max_context_length: int = 4000,
        async_session: AsyncSession | None = None,
    ):
        self.max_history = max_history
        self.max_context_length = max_context_length
        self.sessions: dict[str, ConversationSession] = {}
        self.async_session = async_session
        self._session_created_here = False  # Track if we created this session

    async def _ensure_session(self) -> AsyncSession:
        """Ensure a valid database session."""
        if self.async_session is None:
            factory = get_session_factory()
            self.async_session = factory()
            self._session_created_here = True
        return self.async_session

    async def close(self) -> None:
        """关闭数据库会话，释放连接池连接"""
        if self.async_session and self._session_created_here:
            await self.async_session.close()
            self.async_session = None
            self._session_created_here = False

    def create_session(self) -> ConversationSession:
        """创建新会话（同步版本，仅更新内存）"""
        session = ConversationSession(
            session_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            messages=[],
            context_documents=[],
            is_active=True,
        )
        self.sessions[session.session_id] = session
        return session

    async def create_session_db(self) -> ConversationSession:
        """创建新会话（异步版本，写入数据库）"""
        session: AsyncSession = await self._ensure_session()

        conv = Conversation(
            id=uuid.uuid4(),
            is_active=True,
        )
        session.add(conv)
        await session.commit()

        conversation_session = ConversationSession(
            session_id=str(conv.id),
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            messages=[],
            context_documents=[],
            is_active=True,
        )
        conversation_session.db_confirmed = True
        self.sessions[conversation_session.session_id] = conversation_session
        return conversation_session

    def get_session(self, session_id: str) -> ConversationSession | None:
        return self.sessions.get(session_id)

    async def get_or_load_session(self, session_id: str) -> ConversationSession | None:
        """Get session, lazily loading messages from DB if needed.

        Unlike get_session(), this ensures messages are loaded
        for returning sessions after server restart.
        """
        session = self.sessions.get(session_id)
        if session and not session.messages:
            await self._load_messages_if_needed(session_id)
        return self.sessions.get(session_id)  # re-fetch after lazy load

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> MessageSchema | None:
        """添加消息到会话（异步版本，写入数据库）"""
        session: AsyncSession = await self._ensure_session()

        # Validate UUID format before trying to use it
        try:
            session_uuid = uuid.UUID(session_id)
        except (ValueError, AttributeError):
            return None

        # 检查内存中是否有session，如果没有则从数据库加载
        session_obj = self.sessions.get(session_id)
        db_conv_exists = getattr(session_obj, "db_confirmed", False)

        if not session_obj:
            # 尝试从数据库加载该session
            result = await session.execute(select(Conversation).where(Conversation.id == session_uuid))
            conv = result.scalar_one_or_none()
            if conv:
                session_obj = ConversationSession(
                    session_id=str(conv.id),
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    messages=[],
                    context_documents=[],
                    is_active=conv.is_active,
                )
                self.sessions[session_id] = session_obj
                db_conv_exists = True

        if not session_obj:
            return None

        # Set session title from first user message
        if role == "user" and session_obj.session_title is None:
            session_obj.session_title = content[:50] + ("..." if len(content) > 50 else "")

        msg_schema = MessageSchema(
            message_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(UTC),
            metadata=metadata or {},
        )

        if db_conv_exists:
            msg = Message(
                id=uuid.UUID(msg_schema.message_id),
                session_id=uuid.UUID(session_id),
                role=role,
                content=content,
                confidence=metadata.get("confidence") if metadata else None,
                citations=metadata.get("citations") if metadata else None,
                warnings=metadata.get("warnings") if metadata else None,
                extra_data=metadata,
            )
            session.add(msg)

            await session.execute(
                update(Conversation)
                .where(Conversation.id == uuid.UUID(session_id))
                .values(
                    message_count=Conversation.message_count + 1,
                    updated_at=datetime.now(UTC),
                    session_title=session_obj.session_title,
                )
            )
            await session.commit()
            await self._evict_messages_if_needed(session_id, session)
        else:
            # Just update in-memory, don't persist to DB
            pass

        session_obj.messages.append(msg_schema)
        session_obj.updated_at = datetime.now(UTC)
        # Keep msg_count in sync with actual message list
        session_obj.msg_count = len(session_obj.messages)

        return msg_schema

    async def _evict_messages_if_needed(self, session_id: str, session: AsyncSession) -> None:
        """Evict oldest user-assistant pairs if message count exceeds limit."""
        session_obj = self.sessions.get(session_id)
        if not session_obj:
            return

        message_count = len(session_obj.messages)
        if message_count <= self.MAX_SESSION_MESSAGES:
            return

        # Calculate how many pairs to remove
        pairs_to_remove = message_count - self.MAX_SESSION_MESSAGES
        evicted_ids: list[str] = []

        # Evict from the oldest messages (beginning of list)
        # Each pair is user + assistant (2 messages)
        removed_count = 0
        i = 0
        while i < len(session_obj.messages) - 1 and removed_count < pairs_to_remove:
            msg = session_obj.messages[i]
            next_msg = session_obj.messages[i + 1]
            if msg.role == "user" and next_msg.role == "assistant":
                evicted_ids.extend([msg.message_id, next_msg.message_id])
                removed_count += 1
                i += 2  # Skip both messages of the pair
            else:
                i += 1  # Skip this message (not a valid pair)

        # Batch update evicted messages in database
        if evicted_ids:
            try:
                uuids = [uuid.UUID(mid) for mid in evicted_ids]
                await session.execute(update(Message).where(Message.id.in_(uuids)).values(extra_data={"evicted": True}))
                await session.commit()
            except Exception as e:
                logger.warning(f"Failed to mark messages as evicted: {e}")

        # Remove evicted messages from session object
        if evicted_ids:
            # Remove from the beginning (oldest pairs)
            session_obj.messages = session_obj.messages[len(evicted_ids) :]
            # Sync msg_count with actual message list length after eviction
            session_obj.msg_count = len(session_obj.messages)

    def get_messages(self, session_id: str) -> list[MessageSchema]:
        session = self.sessions.get(session_id)
        if not session:
            return []
        return session.messages

    async def delete_session(self, session_id: str) -> bool:
        """删除会话（从数据库硬删除）"""
        session_uuid = None
        try:
            session_uuid = uuid.UUID(session_id)
        except (ValueError, AttributeError):
            return False

        # Check in-memory sessions first
        in_memory = session_id in self.sessions

        session: AsyncSession = await self._ensure_session()

        # Delete from DB (hard delete - actually remove the record)
        db_deleted = False
        if session_uuid:
            from sqlalchemy import delete

            try:
                result = await session.execute(delete(Conversation).where(Conversation.id == session_uuid))
                await session.commit()
                # rowcount may not be reliable for all DB drivers in async context
                rowcount = getattr(result, "rowcount", 0)
                db_deleted = rowcount > 0 if isinstance(rowcount, int) else False
            except Exception as e:
                logger.error(f"Failed to delete session {session_id} from database: {e}")
                db_deleted = False

        # Delete from in-memory if present
        if in_memory:
            del self.sessions[session_id]

        return db_deleted or in_memory

    async def _load_sessions_from_db(self):
        """启动时从数据库加载活跃会话列表到内存（不加载消息，懒加载）"""
        if self.sessions:
            return

        session: AsyncSession = await self._ensure_session()
        try:
            result = await session.execute(select(Conversation).where(Conversation.is_active))
            for conv in result.scalars():
                # 只加载会话元数据，不加载消息（懒加载）
                self.sessions[str(conv.id)] = ConversationSession(
                    session_id=str(conv.id),
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    messages=[],  # 懒加载，初始为空
                    context_documents=[],
                    is_active=conv.is_active,
                )
            logger.info(f"Loaded {len(self.sessions)} sessions from database (lazy messages)")
        except Exception as e:
            logger.error(f"Failed to load sessions from database: {e}")

    async def _load_messages_if_needed(self, session_id: str) -> None:
        """按需加载会话消息（懒加载）"""
        session_obj = self.sessions.get(session_id)
        if not session_obj:
            return

        # 如果消息已加载则跳过
        if session_obj.messages:
            return

        try:
            session: AsyncSession = await self._ensure_session()
            session_uuid = uuid.UUID(session_id)

            messages_result = await session.execute(
                select(Message).where(Message.session_id == session_uuid).order_by(Message.created_at)
            )
            session_obj.messages = [
                MessageSchema(
                    message_id=str(m.id),
                    role=m.role,
                    content=m.content,
                    timestamp=m.created_at,
                    metadata=m.extra_data or {},
                )
                for m in messages_result.scalars()
            ]
            logger.debug(f"Loaded {len(session_obj.messages)} messages for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to load messages for session {session_id}: {e}")

    def build_context(
        self,
        session_id: str,
        current_query: str,
        retrieved_docs: list[dict],
    ) -> str:
        session = self.sessions.get(session_id)
        if not session:
            return ""

        relevant_history = self._filter_relevant_history(current_query, session.messages)

        history_text = self._format_history(relevant_history)

        docs_text = self._format_documents(retrieved_docs)

        context = f"{history_text}\n\n## 参考文档\n{docs_text}"

        if self._count_tokens(context) > self.max_context_length:
            context = self._truncate_context(context)

        return context

    def _filter_relevant_history(self, current_query: str, messages: list[MessageSchema]) -> list[MessageSchema]:
        if not messages:
            return []

        query_terms = set(current_query.lower().split())

        scored_messages = []
        for msg in messages[-self.max_history :]:
            content_terms = set(msg.content.lower().split())
            overlap = len(query_terms & content_terms)
            score = overlap / len(query_terms) if query_terms else 0
            scored_messages.append((msg, score))

        threshold = 0.3
        relevant = [msg for msg, score in scored_messages if score >= threshold]

        return relevant if relevant else messages[-3:]

    def _format_history(self, messages: list[MessageSchema]) -> str:
        if not messages:
            return ""
        lines = [format_history_message(msg.role, msg.content) for msg in messages]
        return "\n\n".join(lines)

    def _format_documents(self, docs: list[dict]) -> str:
        if not docs:
            return "无可用参考文档"

        lines = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "未知来源")
            content = doc.get("content", "")
            lines.append(f"[{i}] {source}:\n{content[:200]}...")

        return "\n\n".join(lines)

    def _count_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate_context(self, context: str) -> str:
        max_chars = self.max_context_length * 4
        if len(context) <= max_chars:
            return context
        return context[:max_chars] + "\n\n...(上下文已截断)"
