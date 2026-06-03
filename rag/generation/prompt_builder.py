# rag/generation/prompt_builder.py
from typing import Any

from app.models.schemas import RetrievedNode
from rag.generation.prompt import (
    build_system_prompt,
    build_user_prompt,
    format_contexts,
)


class PromptBuilder:
    """Builds prompts for LLM generation."""

    @staticmethod
    def build(
        query: str,
        contexts: list[RetrievedNode],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        """Build system and user prompts from query and contexts.

        Note: conversation_history is acknowledged here for API compatibility
        but actual history injection is handled by LLMGenerator as
        separate messages in the chat completion call.
        """
        _ = conversation_history  # History handled by LLMGenerator as separate messages
        context_texts = [
            {
                "content": ctx.content,
                "source": ctx.metadata.get("source_file", "未知来源"),
                "page": ctx.metadata.get("page_number"),
            }
            for ctx in contexts
        ]
        formatted_contexts = format_contexts(context_texts)
        return build_system_prompt(), build_user_prompt(query, formatted_contexts)
