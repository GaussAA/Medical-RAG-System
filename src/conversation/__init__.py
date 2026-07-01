"""Conversation slice — public API.

Other modules MUST import from here instead of conversation.internal modules.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.common.models import Message


class SessionManagerPort(Protocol):
    """What conversation provides to query and other modules."""

    async def get_or_load_session(self, session_id: str) -> Any: ...

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message | None: ...

    async def create_session_db(self) -> Any: ...

    async def close(self) -> None: ...


class ConsistencyCheckerPort(Protocol):
    """What conversation provides for cross-store consistency checks."""

    async def check_all_consistency(self, repair: bool = False) -> Any: ...

    async def cleanup_orphans(self) -> Any: ...
