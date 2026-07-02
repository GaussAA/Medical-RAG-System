"""Database engine and session management.

⚠️ ORM models are NOT re-exported from here to avoid circular imports.
Import them directly from each slice's models.py:
  from src.documents.models import Document, Chunk, Heading
  from src.conversation.models import Conversation, Message
"""

from src.common.database.engine import (
    _ensure_engine_initialized,
    close_engine,
    get_engine,
    get_session,
    get_session_factory,
    get_standalone_session,
)
from src.common.database.models import Base

__all__ = [
    "_ensure_engine_initialized",
    "get_engine",
    "get_session_factory",
    "get_session",
    "get_standalone_session",
    "close_engine",
    "Base",
]
