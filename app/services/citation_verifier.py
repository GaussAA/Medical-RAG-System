import re
from typing import Any

from app.models.schemas import Citation, CitationPosition, RetrievedNode


class CitationVerifier:
    """Service to extract and verify citations from LLM answer text."""

    # Pattern: [来源X](文件名称#页码) - old format (English brackets)
    CITATION_PATTERN_OLD = re.compile(r"\[来源(\d+)\]\(([^)]+)\)")
    # Pattern: 「来源X」（文件名#页码）- new format (Chinese quotes)
    CITATION_PATTERN_NEW = re.compile(r"「来源(\d+)」（([^）]+)）")

    def extract_and_verify(
        self,
        answer: str,
        contexts: list[RetrievedNode],
    ) -> list[Citation]:
        """
        Extract citations from LLM answer text and verify against retrieval contexts.

        Args:
            answer: LLM生成的回答文本
            contexts: 检索到的节点列表

        Returns:
            list[Citation]: 提取并验证后的引用列表
        """
        if not contexts:
            return []

        # Build context index by position (1-based)
        context_by_index: dict[int, RetrievedNode] = {i: ctx for i, ctx in enumerate(contexts, 1)}

        citations: list[Citation] = []
        matched_indices: set[int] = set()

        # Find all citation patterns in the answer (both old and new formats)
        for pattern in [self.CITATION_PATTERN_OLD, self.CITATION_PATTERN_NEW]:
            for match in pattern.finditer(answer):
                source_index = int(match.group(1))
                source_desc = match.group(2)  # e.g., "文件名称#页码"

                # Parse source description
                file_name, page_number = self._parse_source_desc(source_desc)

                if source_index in context_by_index:
                    ctx = context_by_index[source_index]
                    matched_indices.add(source_index)

                    # Check if source description matches context metadata
                    ctx_file = ctx.metadata.get("source_file", "")
                    verified = self._verify_citation(
                        file_name,
                        page_number,
                        ctx_file,
                        ctx.metadata.get("page_number"),
                    )

                    citation = Citation(
                        source_id=str(source_index),
                        document_id=ctx.metadata.get("doc_id"),
                        file_name=file_name or ctx.metadata.get("source_file", "未知来源"),
                        page_number=page_number or ctx.metadata.get("page_number"),
                        chunk_content=ctx.content,
                        relevance_score=ctx.score,
                        position=CitationPosition.DIRECT if verified else CitationPosition.UNVERIFIED,
                        verified=verified,
                        quote_in_answer=match.group(0),
                        verification_message=None if verified else f"引用来源 '{file_name}' 与检索文档不匹配",
                    )
                else:
                    # Citation index out of range - likely hallucination
                    citation = Citation(
                        source_id=str(source_index),
                        document_id=None,
                        file_name=file_name or "未知来源",
                        page_number=page_number,
                        chunk_content="",
                        relevance_score=0.0,
                        position=CitationPosition.UNVERIFIED,
                        verified=False,
                        quote_in_answer=match.group(0),
                        verification_message=f"引用索引 {source_index} 超出范围，可能为幻觉",
                    )

                # Avoid duplicate citations if both patterns match same text
                if not any(c.quote_in_answer == match.group(0) for c in citations):
                    citations.append(citation)

        # Add unmatched contexts as unverified citations
        for i, ctx in enumerate(contexts, 1):
            if i not in matched_indices:
                # Context was retrieved but not cited in the answer
                citation = Citation(
                    source_id=str(i),
                    document_id=ctx.metadata.get("doc_id"),
                    file_name=ctx.metadata.get("source_file", "未知来源"),
                    page_number=ctx.metadata.get("page_number"),
                    chunk_content=ctx.content,
                    relevance_score=ctx.score,
                    position=CitationPosition.INDIRECT,
                    verified=True,  # Context is valid, just not cited
                    quote_in_answer=None,
                    verification_message="检索到的相关文档但未在回答中引用",
                )
                citations.append(citation)

        return citations

    def _parse_source_desc(self, source_desc: str) -> tuple[str | None, int | None]:
        """
        Parse source description like '文件名称#页码' into components.

        Returns:
            tuple of (file_name, page_number)
        """
        if "#" in source_desc:
            parts = source_desc.rsplit("#", 1)
            file_name = parts[0] if parts[0] else None
            try:
                page_number = int(parts[1]) if len(parts) > 1 else None
            except ValueError:
                page_number = None
            return file_name, page_number
        return source_desc, None

    def _verify_citation(
        self,
        cited_file: str | None,
        cited_page: int | None,
        ctx_file: str,
        ctx_page: int | None,
    ) -> bool:
        """
        Verify citation against retrieval context.

        Checks:
        1. File name matching (fuzzy for Chinese filenames)
        2. Page number consistency if both provided

        Args:
            cited_file: File name from citation in answer.
            cited_page: Page number from citation.
            ctx_file: File name from retrieval context metadata.
            ctx_page: Page number from retrieval context metadata.

        Returns:
            True if citation is verified, False otherwise.
        """
        if not ctx_file:
            return False

        # If no cited file provided, can't verify
        if not cited_file:
            return True

        # Normalize file names for comparison
        cited_normalized = self._normalize_filename(cited_file)
        ctx_normalized = self._normalize_filename(ctx_file)

        # Check if file names match (allowing partial matching for Chinese filenames)
        if not self._files_match(cited_normalized, ctx_normalized):
            return False

        # Page number verification if both are provided
        if cited_page is not None and ctx_page is not None:
            if cited_page != ctx_page:
                return False

        return True

    def _normalize_filename(self, filename: str) -> str:
        """
        Normalize filename for comparison.

        - Remove common suffixes like #页码
        - Lowercase for comparison
        - Remove extensions
        """
        if not filename:
            return ""

        # Remove page suffix like "#1", "#2"
        normalized = re.sub(r"#\d+$", "", filename)

        # Lowercase
        normalized = normalized.lower()

        # Remove file extension
        normalized = re.sub(r"\.(md|txt|pdf|docx)$", "", normalized)

        return normalized.strip()

    def _files_match(self, file1: str, file2: str) -> bool:
        """
        Check if two file names match (allowing fuzzy matching).

        Handles cases like:
        - "糖尿病指南" matches "糖尿病指南.md"
        - "指南" is substring of "糖尿病指南"
        """
        if not file1 or not file2:
            return False

        # Exact match (after normalization)
        if file1 == file2:
            return True

        # Substring match (one contains the other)
        if file1 in file2 or file2 in file1:
            return True

        # Remove common medical keywords and check again
        common_suffixes = ["指南", "共识", "标准", "规范", "手册", "方案"]
        for suffix in common_suffixes:
            f1_base = file1.replace(suffix, "").strip()
            f2_base = file2.replace(suffix, "").strip()
            if f1_base and f2_base and (f1_base in f2_base or f2_base in f1_base):
                return True

        return False

    def extract_citations_only(self, contexts: list[dict[str, Any]]) -> list[Citation]:
        """
        Extract citations from contexts without verifying against answer text.
        Used when answer doesn't contain citation patterns.

        Args:
            contexts: List of context dicts with content, source, page keys

        Returns:
            list[Citation]: Basic citations from contexts
        """
        citations = []
        for i, ctx in enumerate(contexts, 1):
            citation = Citation(
                source_id=str(i),
                document_id=ctx.get("node_id"),
                file_name=ctx.get("source", "未知来源"),
                page_number=ctx.get("page"),
                chunk_content=ctx.get("content", ""),
                relevance_score=ctx.get("score", 0.0),
                position=CitationPosition.DIRECT,
                verified=True,
                quote_in_answer=None,
                verification_message=None,
            )
            citations.append(citation)
        return citations
