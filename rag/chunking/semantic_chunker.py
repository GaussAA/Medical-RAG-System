import re
import uuid

from app.models.schemas import Chunk, ChunkMetadata
from config.settings import get_settings
from rag.chunking.chunker import BaseChunker

MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class SemanticChunker(BaseChunker):
    def __init__(self):
        settings = get_settings()
        self.config = settings.rag.chunking

        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.separators = self.config.separator
        self.preserve_tables = self.config.preserve_tables
        self.min_chunk_length = self.config.min_chunk_length
        self.max_chunk_length = self.config.max_chunk_length
        self.markdown_aware = getattr(self.config, "markdown_aware", False)

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text.strip():
            return []

        metadata = metadata or {}

        # Check if Markdown-aware chunking is enabled and text is Markdown
        if self.markdown_aware and self._is_markdown(text):
            chunks = self._chunk_by_markdown_headings(text, metadata)
        elif self.preserve_tables and "tables" in metadata:
            chunks = self._chunk_with_tables(text, metadata)
        else:
            chunks = self._chunk_by_separators(text, metadata)

        chunks = self._enforce_size_limits(chunks)

        return self._merge_small_chunks(chunks)

    def _is_markdown(self, text: str) -> bool:
        """Detect if text contains Markdown heading syntax.

        Args:
            text: Text to check

        Returns:
            True if text contains Markdown heading patterns
        """
        lines = text.split("\n")
        heading_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and MARKDOWN_HEADING_RE.match(stripped):
                heading_count += 1
                if heading_count >= 2:
                    return True
        return False

    def _chunk_by_markdown_headings(self, text: str, metadata: dict) -> list[Chunk]:
        """Split text by Markdown headings to preserve semantic units.

        Args:
            text: Markdown text to chunk
            metadata: Metadata dict

        Returns:
            List of Chunk objects
        """
        lines = text.split("\n")
        chunks = []
        current_heading = ""
        current_heading_level = 1
        current_content: list[str] = []
        current_position = 0

        def parse_heading(line: tuple[int, str]) -> tuple[str, int] | None:
            """Parse a Markdown heading line.

            Returns:
                Tuple of (heading_text, heading_level) or None if not a heading
            """
            match = MARKDOWN_HEADING_RE.match(line[1].strip())
            if match:
                level = len(match.group(1))
                return match.group(2), level
            return None

        def create_chunk(content: str, heading: str, level: int, position: int) -> Chunk:
            """Create a chunk with Markdown content."""
            chunk_metadata = ChunkMetadata(
                source_file=metadata.get("source_file", ""),
                page_number=metadata.get("page_number"),
                section_title=heading or metadata.get("section_title"),
                char_count=len(content),
                position=position,
                heading_level=level if heading else None,
            )
            return Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=metadata.get("doc_id", ""),
                content=content.strip(),
                token_count=self.count_tokens(content),
                metadata=chunk_metadata,
            )

        for i, line in enumerate(lines):
            heading_info = parse_heading((i, line))

            if heading_info:
                # Save current chunk if it has content
                if current_content:
                    chunk_text = "\n".join(current_content)
                    chunks.append(
                        create_chunk(
                            chunk_text, current_heading, current_heading_level, current_position
                        )
                    )
                    current_position += 1

                # Start new heading section
                current_heading, current_heading_level = heading_info
                current_content = [line]
            else:
                current_content.append(line)

        # Don't forget the last chunk
        if current_content:
            chunk_text = "\n".join(current_content)
            chunks.append(
                create_chunk(chunk_text, current_heading, current_heading_level, current_position)
            )

        return chunks

    def _chunk_by_separators(self, text: str, metadata: dict) -> list[Chunk]:
        parts = self._split_by_separators(text)
        chunks = []

        for i, part in enumerate(parts):
            chunk_metadata = ChunkMetadata(
                source_file=metadata.get("source_file", ""),
                page_number=metadata.get("page_number"),
                section_title=metadata.get("section_title"),
                char_count=len(part),
                position=i,
            )

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=metadata.get("doc_id", ""),
                content=part.strip(),
                token_count=self.count_tokens(part),
                metadata=chunk_metadata,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_with_tables(self, text: str, metadata: dict) -> list[Chunk]:
        chunks = []
        tables = metadata.get("tables", [])

        text_parts = text.split("\n\n")
        current_text_parts = []

        for part in text_parts:
            if self._contains_table(part, tables):
                if current_text_parts:
                    combined_text = "\n\n".join(current_text_parts)
                    text_chunks = self._chunk_by_separators(combined_text, metadata)
                    chunks.extend(text_chunks)
                    current_text_parts = []

                table_chunk = self._create_table_chunk(part, tables, metadata, len(chunks))
                if table_chunk:
                    chunks.append(table_chunk)
            else:
                current_text_parts.append(part)

        if current_text_parts:
            combined_text = "\n\n".join(current_text_parts)
            text_chunks = self._chunk_by_separators(combined_text, metadata)
            chunks.extend(text_chunks)

        return chunks

    def _split_by_separators(self, text: str) -> list[str]:
        parts = [text]

        for sep in self.separators:
            new_parts = []
            for part in parts:
                splits = part.split(sep)
                new_parts.extend(splits)
            parts = new_parts

        parts = [p.strip() for p in parts if p.strip()]
        return parts

    def _contains_table(self, text: str, tables: list[dict]) -> bool:
        for table in tables:
            if table.get("caption") and table["caption"] in text:
                return True
        return False

    def _create_table_chunk(
        self, text: str, tables: list[dict], metadata: dict, position: int
    ) -> Chunk | None:
        for table in tables:
            if table.get("caption") and table["caption"] in text:
                table_content = self._format_table(table)
                chunk_metadata = ChunkMetadata(
                    source_file=metadata.get("source_file", ""),
                    page_number=metadata.get("page_number"),
                    section_title=f"表格: {table.get('caption', '')}",
                    char_count=len(table_content),
                    position=position,
                )

                return Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=metadata.get("doc_id", ""),
                    content=table_content,
                    token_count=self.count_tokens(table_content),
                    metadata=chunk_metadata,
                )

        return None

    def _format_table(self, table: dict) -> str:
        headers = table.get("headers", [])
        rows = table.get("rows", [])

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def _enforce_size_limits(self, chunks: list[Chunk]) -> list[Chunk]:
        result = []

        for chunk in chunks:
            if chunk.metadata.char_count <= self.max_chunk_length:
                result.append(chunk)
                continue

            # If Markdown-aware and this is a heading-based chunk, split by paragraphs
            if self.markdown_aware and chunk.metadata.heading_level is not None:
                sub_parts = self._split_markdown_by_paragraphs(chunk)
            else:
                sub_parts = self._split_large_chunk(chunk)

            for i, part in enumerate(sub_parts):
                sub_metadata = ChunkMetadata(
                    source_file=chunk.metadata.source_file,
                    page_number=chunk.metadata.page_number,
                    section_title=chunk.metadata.section_title,
                    char_count=len(part),
                    position=chunk.metadata.position * 100 + i,
                    heading_level=chunk.metadata.heading_level,
                )

                sub_chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=chunk.doc_id,
                    content=part,
                    token_count=self.count_tokens(part),
                    metadata=sub_metadata,
                )
                result.append(sub_chunk)

        return result

    def _split_markdown_by_paragraphs(self, chunk: Chunk) -> list[str]:
        """Split a Markdown chunk by paragraphs while preserving heading context.

        Args:
            chunk: Chunk to split

        Returns:
            List of text parts
        """
        parts = []
        content = chunk.content

        # If it starts with a heading, keep it and split the rest
        lines = content.split("\n")
        heading_lines: list[str] = []
        body_lines: list[str] = []

        for line in lines:
            if MARKDOWN_HEADING_RE.match(line.strip()):
                heading_lines.append(line)
            else:
                body_lines.append(line)

        # Reconstruct heading if we have one
        heading_text = "\n".join(heading_lines) if heading_lines else ""

        # Split body by double newlines (paragraphs)
        paragraphs = "\n".join(body_lines).split("\n\n")

        current_part = heading_text + "\n" if heading_text else ""
        current_size = len(current_part)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_size + len(para) + 2 <= self.max_chunk_length:
                current_part += para + "\n\n"
                current_size += len(para) + 2
            else:
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = (
                    heading_text + "\n" + para + "\n\n" if heading_text else para + "\n\n"
                )
                current_size = len(current_part)

        if current_part.strip():
            parts.append(current_part.strip())

        return parts if parts else [content]

    def _split_large_chunk(self, chunk: Chunk) -> list[str]:
        parts = []
        content = chunk.content

        while len(content) > self.max_chunk_length:
            split_point = self.max_chunk_length
            for sep in self.separators:
                idx = content.rfind(sep, 0, self.max_chunk_length)
                if idx > 0:
                    split_point = idx + len(sep)
                    break

            parts.append(content[:split_point])
            content = content[split_point:]

        if content.strip():
            parts.append(content)

        return parts

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        merged = [chunks[0]]

        for chunk in chunks[1:]:
            last = merged[-1]

            if last.metadata.char_count + chunk.metadata.char_count < self.max_chunk_length:
                combined_content = last.content + "\n\n" + chunk.content
                combined_metadata = ChunkMetadata(
                    source_file=last.metadata.source_file,
                    page_number=last.metadata.page_number,
                    section_title=last.metadata.section_title,
                    char_count=last.metadata.char_count + chunk.metadata.char_count + 2,
                    position=last.metadata.position,
                    heading_level=last.metadata.heading_level,
                )

                merged[-1] = Chunk(
                    chunk_id=last.chunk_id,
                    doc_id=last.doc_id,
                    content=combined_content,
                    token_count=self.count_tokens(combined_content),
                    metadata=combined_metadata,
                )
            else:
                merged.append(chunk)

        return merged
