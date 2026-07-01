"""
Document chunking strategies - combined module.

Migrated from: rag/chunking/chunker.py, rag/chunking/hierarchical_chunker.py, rag/chunking/semantic_chunker.py
"""

import re
import uuid
from abc import ABC, abstractmethod
from functools import lru_cache

from src.common.config.settings import get_settings
from src.common.models import Chunk, ChunkMetadata

# ==================== Base Chunker ====================


class BaseChunker(ABC):
    """Abstract base class for all chunkers."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        pass

    @lru_cache(maxsize=1024)
    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def should_split(self, text: str, separators: list[str]) -> tuple[bool, str | None]:
        for sep in separators:
            if sep in text:
                return True, sep
        return False, None


# ==================== Hierarchical Chunker ====================

MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_CAPTION_RE = re.compile(r"^\*\*(表[一二三四五六七八九十\d]+[^\*]*)\*\*$")


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical-aware chunker for Markdown medical documents.

    Strategy:
    1. Split by heading boundaries to preserve semantic units
    2. For each heading section:
       - If content length <= max_chunk_size -> single chunk with heading context
       - If content length > max_chunk_size -> split by semantic boundaries
    3. Attach full heading tree context to each chunk
    """

    def __init__(self):
        settings = get_settings()
        self.config = settings.rag.chunking

        self.chunk_size = self.config.chunk_size
        self.chunk_overlap = self.config.chunk_overlap
        self.max_chunk_length = self.config.max_chunk_length
        self.min_chunk_length = self.config.min_chunk_length
        self.separators = self.config.separator

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text.strip():
            return []

        metadata = metadata or {}
        heading_tree = metadata.get("heading_tree", {})
        tables = metadata.get("tables", [])

        sections = self._split_by_headings(text)

        chunks = []
        position = 0

        for section in sections:
            heading_info = section.get("heading", {})
            content = section.get("content", "")
            heading_level = heading_info.get("level", 0)
            heading_title = heading_info.get("title", "")

            if not content.strip():
                continue

            section_heading_tree = self._build_section_heading_tree(heading_tree, heading_level, heading_title)
            content_type = self._detect_content_type(content, tables)

            if len(content) <= self.max_chunk_length:
                chunk_metadata = ChunkMetadata(
                    source_file=metadata.get("source_file", ""),
                    section_title=heading_title,
                    heading_tree=section_heading_tree,
                    content_type=content_type,
                    char_count=len(content),
                    position=position,
                    heading_level=heading_level,
                )
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=metadata.get("doc_id", ""),
                        content=content.strip(),
                        token_count=self.count_tokens(content),
                        metadata=chunk_metadata,
                    )
                )
                position += 1
            else:
                sub_chunks = self._split_large_content(
                    content, section_heading_tree, heading_title, heading_level,
                    tables, metadata.get("source_file", ""), position,
                )
                chunks.extend(sub_chunks)
                position += len(sub_chunks)

        chunks = self._merge_small_chunks(chunks)
        return chunks

    def _split_by_headings(self, text: str) -> list[dict]:
        """Split text by Markdown headings."""
        lines = text.split("\n")
        sections = []
        current_heading = None
        current_content_lines: list[str] = []
        current_start_line = 0

        for line_num, line in enumerate(lines, start=1):
            match = MARKDOWN_HEADING_RE.match(line.strip())
            if match:
                if current_heading is not None and current_content_lines:
                    content = "\n".join(current_content_lines)
                    if content.strip():
                        sections.append({
                            "heading": current_heading,
                            "content": content,
                            "start_line": current_start_line,
                        })
                level = len(match.group(1))
                title = match.group(2).strip()
                current_heading = {"level": level, "title": title}
                current_content_lines = []
                current_start_line = line_num + 1
            elif current_heading is not None:
                current_content_lines.append(line)

        if current_content_lines:
            content = "\n".join(current_content_lines)
            if content.strip():
                sections.append({
                    "heading": current_heading or {"level": 0, "title": ""},
                    "content": content,
                    "start_line": current_start_line,
                })

        if not sections and text.strip() and current_heading is None:
            sections.append({
                "heading": {"level": 0, "title": ""},
                "content": text.strip(),
                "start_line": 1,
            })

        return sections

    def _build_section_heading_tree(self, full_tree: dict[int, str], level: int, title: str) -> dict[int, str]:
        result = {}
        for lvl, h_title in full_tree.items():
            if lvl < level:
                result[lvl] = h_title
            elif lvl == level:
                result[lvl] = title
                break
        return result

    def _detect_content_type(self, content: str, tables: list[dict]) -> str:
        stripped = content.strip()
        if stripped.startswith("|") or "|--" in stripped:
            return "table"
        if TABLE_CAPTION_RE.search(stripped[:100]):
            return "table"

        lines = stripped.split("\n")
        list_lines = sum(1 for line in lines if re.match(r"^[·\-\*]\s+", line.strip()))
        if list_lines > len(lines) * 0.3:
            return "list"
        return "text"

    def _split_large_content(
        self, content, heading_tree, section_title,
        heading_level, tables, source_file, start_position,
    ):
        """Split large content into smaller chunks."""
        chunks = []
        lines = content.split("\n")
        current_block = []
        current_block_type = "text"
        current_size = 0
        position = start_position

        for line in lines:
            line_size = len(line)
            is_table_row = line.strip().startswith("|")
            is_list_item = bool(re.match(r"^[·\-\*]\s+", line.strip()))
            block_type = "table" if is_table_row else ("list" if is_list_item else "text")

            if current_block and (
                (block_type != current_block_type and current_block_type in ("table", "list"))
                or current_size + line_size > self.max_chunk_length
            ):
                block_content = "\n".join(current_block)
                if block_content.strip():
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id="",
                        content=block_content.strip(),
                        token_count=self.count_tokens(block_content),
                        metadata=ChunkMetadata(
                            source_file=source_file,
                            section_title=section_title,
                            heading_tree=heading_tree,
                            content_type=current_block_type,
                            char_count=len(block_content),
                            position=position,
                            heading_level=heading_level,
                        ),
                    ))
                    position += 1
                current_block = []
                current_block_type = block_type
                current_size = 0

            current_block.append(line)
            current_size += line_size
            if current_block_type == "text":
                current_block_type = block_type

        if current_block:
            block_content = "\n".join(current_block)
            if block_content.strip():
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id="",
                    content=block_content.strip(),
                    token_count=self.count_tokens(block_content),
                    metadata=ChunkMetadata(
                        source_file=source_file,
                        section_title=section_title,
                        heading_tree=heading_tree,
                        content_type=current_block_type,
                        char_count=len(block_content),
                        position=position,
                        heading_level=heading_level,
                    ),
                ))
        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge consecutive small chunks."""
        if not chunks:
            return []
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            last = merged[-1]
            can_merge = (
                last.metadata.content_type == chunk.metadata.content_type
                and last.metadata.heading_level == chunk.metadata.heading_level
                and last.metadata.char_count + chunk.metadata.char_count < self.max_chunk_length
            )
            if can_merge:
                combined = last.content + "\n\n" + chunk.content
                merged[-1] = Chunk(
                    chunk_id=last.chunk_id,
                    doc_id=last.doc_id,
                    content=combined,
                    token_count=self.count_tokens(combined),
                    metadata=ChunkMetadata(
                        source_file=last.metadata.source_file,
                        section_title=last.metadata.section_title,
                        heading_tree=last.metadata.heading_tree,
                        content_type=last.metadata.content_type,
                        char_count=len(combined),
                        position=last.metadata.position,
                        heading_level=last.metadata.heading_level,
                    ),
                )
            else:
                merged.append(chunk)
        return merged


# ==================== Semantic Chunker ====================


class SemanticChunker(BaseChunker):
    """Semantic boundary-based chunker."""

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

        if self.markdown_aware:
            return self._markdown_aware_chunk(text, metadata)

        return self._semantic_chunk(text, metadata)

    def _markdown_aware_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """Markdown-aware chunking that handles tables and headings."""
        lines = text.split("\n")
        chunks: list[Chunk] = []
        current_chunk_lines: list[str] = []
        current_size = 0
        inside_table = False
        position = 0

        for line in lines:
            stripped = line.strip()

            is_table = stripped.startswith("|")

            if is_table:
                inside_table = True
                current_chunk_lines.append(line)
                current_size += len(line)
                continue

            if inside_table and not is_table:
                inside_table = False
                self._flush_chunk(chunks, current_chunk_lines, metadata, position)
                position += 1
                current_chunk_lines = []
                current_size = 0

            # Check for heading
            heading_match = MARKDOWN_HEADING_RE.match(stripped) if self.markdown_aware else None
            if heading_match:
                if current_chunk_lines and current_size >= self.min_chunk_length:
                    self._flush_chunk(chunks, current_chunk_lines, metadata, position)
                    position += 1
                    current_chunk_lines = []
                    current_size = 0
                current_chunk_lines.append(line)
                current_size += len(line)
                continue

            current_chunk_lines.append(line)
            current_size += len(line)

            if current_size >= self.chunk_size:
                self._flush_chunk(chunks, current_chunk_lines, metadata, position)
                position += 1
                current_chunk_lines = []
                current_size = 0

        if current_chunk_lines:
            self._flush_chunk(chunks, current_chunk_lines, metadata, position)

        return chunks

    def _semantic_chunk(self, text: str, metadata: dict) -> list[Chunk]:
        """Generic semantic chunking."""
        chunks: list[Chunk] = []
        paragraphs = text.split("\n\n")
        current_chunk: list[str] = []
        current_size = 0
        position = 0

        for para in paragraphs:
            if not para.strip():
                continue

            if current_size + len(para) > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=metadata.get("doc_id", ""),
                    content=chunk_text,
                    token_count=self.count_tokens(chunk_text),
                    metadata=ChunkMetadata(
                        source_file=metadata.get("source_file", ""),
                        section_title=metadata.get("section_title"),
                        heading_tree=metadata.get("heading_tree"),
                        content_type="text",
                        char_count=len(chunk_text),
                        position=position,
                    ),
                ))
                position += 1

                if self.chunk_overlap > 0:
                    overlap_size = 0
                    overlap_lines: list[str] = []
                    for p in reversed(current_chunk):
                        if overlap_size + len(p) > self.chunk_overlap:
                            break
                        overlap_lines.insert(0, p)
                        overlap_size += len(p)
                    current_chunk = overlap_lines
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += len(para)

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=metadata.get("doc_id", ""),
                content=chunk_text,
                token_count=self.count_tokens(chunk_text),
                metadata=ChunkMetadata(
                    source_file=metadata.get("source_file", ""),
                    section_title=metadata.get("section_title"),
                    heading_tree=metadata.get("heading_tree"),
                    content_type="text",
                    char_count=len(chunk_text),
                    position=position,
                ),
            ))

        return chunks

    def _flush_chunk(self, chunks: list, lines: list, metadata: dict, position: int) -> None:
        """Flush current lines as a chunk."""
        chunk_text = "\n".join(lines)
        content_type = "table" if any(line.strip().startswith("|") for line in lines) else "text"
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=metadata.get("doc_id", ""),
            content=chunk_text,
            token_count=self.count_tokens(chunk_text),
            metadata=ChunkMetadata(
                source_file=metadata.get("source_file", ""),
                section_title=metadata.get("section_title"),
                heading_tree=metadata.get("heading_tree"),
                content_type=content_type,
                char_count=len(chunk_text),
                position=position,
            ),
        ))
