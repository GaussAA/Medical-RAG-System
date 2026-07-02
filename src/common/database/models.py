"""
SQLAlchemy ORM models — Base class only (shared metadata).

Concrete models are defined in each slice's own models.py:
- src/documents/models.py    → Document, Chunk, Heading (documents schema)
- src/conversation/models.py → Conversation, Message (conversation schema)
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        },
    )
