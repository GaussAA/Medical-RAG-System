# rag/generation/citation_extractor.py
from typing import Any

from app.models.schemas import Citation, CitationPosition


class CitationExtractor:
    """Extracts citations from retrieved contexts."""

    def extract(self, contexts: list[dict[str, Any]]) -> list[Citation]:
        """Extract citations from context list."""
        citations = []
        for i, ctx in enumerate(contexts, 1):
            citation = Citation(
                source_id=str(i),
                document_id=ctx.get("node_id"),
                file_name=ctx.get("source", "未知来源"),
                page_number=ctx.get("page"),
                chunk_content=ctx.get("content", "")[:200],
                relevance_score=ctx.get("score", ctx.get("relevance_score", 0.0)),
                position=CitationPosition.DIRECT,
                verified=True,
                quote_in_answer=None,
                verification_message=None,
            )
            citations.append(citation)
        return citations