# rag/generation/prompt_builder.py
from typing import Any

from app.models.schemas import RetrievedNode
from rag.generation.prompt import build_system_prompt, build_user_prompt, format_contexts, format_history_message


class PromptBuilder:
    """Builds prompts for LLM generation."""

    @staticmethod
    def build(
        query: str,
        contexts: list[RetrievedNode],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        """Build system and user prompts from query and contexts."""
        context_texts = [
            {
                "content": ctx.content,
                "source": ctx.metadata.get("source_file", "未知来源"),
                "page": ctx.metadata.get("page_number"),
            }
            for ctx in contexts
        ]
        formatted_contexts = format_contexts(context_texts)
        history_text = PromptBuilder._format_history(conversation_history) if conversation_history else ""
        return build_system_prompt(), build_user_prompt(query, formatted_contexts, history_text)

    @staticmethod
    def _format_history(history: list[dict[str, Any]]) -> str:
        """Format conversation history into a string."""
        if not history:
            return ""
        lines = [
            format_history_message(msg.get("role", ""), msg.get("content", ""))
            for msg in history
        ]
        return "\n\n".join(lines)