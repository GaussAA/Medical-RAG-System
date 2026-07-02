"""Generation slice — LLM-based answer generation with citations.

Public API — other modules MUST import from here.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMGeneratorPort(Protocol):
    """Interface: what generation provides to other modules."""

    async def generate(self, query: str, contexts: list[Any], **kwargs: Any) -> dict[str, Any]: ...


# Concrete implementations
from src.generation.generator import LLMGenerator  # noqa: E402, F401
from src.generation.prompt import (  # noqa: E402, F401
    FALLBACK_RESPONSES,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_system_prompt,
    build_user_prompt,
    format_contexts,
    format_history_message,
)
from src.generation.prompt_builder import PromptBuilder  # noqa: E402, F401
from src.generation.warnings import WarningsGenerator  # noqa: E402, F401

__all__ = [
    "LLMGeneratorPort",
    "LLMGenerator",
    "PromptBuilder",
    "WarningsGenerator",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "FALLBACK_RESPONSES",
    "build_system_prompt",
    "build_user_prompt",
    "format_contexts",
    "format_history_message",
]
