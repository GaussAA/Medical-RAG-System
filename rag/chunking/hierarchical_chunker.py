import re
import uuid

from app.models.schemas import Chunk, ChunkMetadata
from config.settings import get_settings
from rag.chunking.chunker import BaseChunker

MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
TABLE_CAPTION_RE = re.compile(r"^\*\*(表[一二三四五六七八九十\d]+[^\*]*)\*\*$")


class HierarchicalChunker(BaseChunker):
    """
    Hierarchical-aware chunker for Markdown medical documents.

    Strategy:
    1. Split by heading boundaries to preserve semantic units
    2. For each heading section:
       - If content length <= max_chunk_size -> single chunk with heading context
       - If content length > max_chunk_size -> split by semantic boundaries:
         - Tables -> independent chunks
         - Paragraphs -> merge until near max size
         - Lists -> keep together if possible
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

        # Split text by headings
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

            # Build heading tree context for this section
            section_heading_tree = self._build_section_heading_tree(heading_tree, heading_level, heading_title)

            # Determine content type
            content_type = self._detect_content_type(content, tables)

            # Check if content needs splitting
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
                # Split large content
                sub_chunks = self._split_large_content(
                    content,
                    section_heading_tree,
                    heading_title,
                    heading_level,
                    tables,
                    metadata.get("source_file", ""),
                    position,
                )
                chunks.extend(sub_chunks)
                position += len(sub_chunks)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _split_by_headings(self, text: str) -> list[dict]:
        """Split text by Markdown headings, preserving heading context."""
        lines = text.split("\n")
        sections = []

        current_heading = None
        current_content_lines: list[str] = []
        current_start_line = 0

        for line_num, line in enumerate(lines, start=1):
            match = MARKDOWN_HEADING_RE.match(line.strip())

            if match:
                # Save previous section if exists
                if current_heading is not None and (current_content_lines or len(current_content_lines) > 0):
                    content = "\n".join(current_content_lines)
                    if content.strip():
                        sections.append(
                            {
                                "heading": current_heading,
                                "content": content,
                                "start_line": current_start_line,
                            }
                        )
                    # Clear after saving
                # Start new section - heading line is NOT included in content
                level = len(match.group(1))
                title = match.group(2).strip()
                current_heading = {"level": level, "title": title}
                current_content_lines = []
                current_start_line = line_num + 1  # Content starts after heading
            elif current_heading is not None:
                # Only add content lines after a heading has been established
                current_content_lines.append(line)

        # Don't forget the last section
        if current_content_lines:
            content = "\n".join(current_content_lines)
            if content.strip():
                sections.append(
                    {
                        "heading": current_heading or {"level": 0, "title": ""},
                        "content": content,
                        "start_line": current_start_line,
                    }
                )

        # If no sections created at all AND no heading was ever set, treat as plain text
        if not sections and text.strip() and current_heading is None:
            sections.append(
                {
                    "heading": {"level": 0, "title": ""},
                    "content": text.strip(),
                    "start_line": 1,
                }
            )

        return sections

    def _build_section_heading_tree(self, full_tree: dict[int, str], level: int, title: str) -> dict[int, str]:
        """Build heading tree up to and including the current heading."""
        result = {}
        for lvl, h_title in full_tree.items():
            if lvl < level:
                result[lvl] = h_title
            elif lvl == level:
                result[lvl] = title
                break
        return result

    def _detect_content_type(self, content: str, tables: list[dict]) -> str:
        """Detect the primary content type of a chunk."""
        stripped = content.strip()

        # Check if it's a table
        if stripped.startswith("|") or "|--" in stripped:
            return "table"

        # Check for table caption pattern
        if TABLE_CAPTION_RE.search(stripped[:100]):
            return "table"

        # Check for list patterns
        lines = stripped.split("\n")
        list_lines = 0
        for line in lines:
            if re.match(r"^[·\-\*]\s+", line.strip()) or re.match(r"^（[0-9]+）", line.strip()):
                list_lines += 1

        if list_lines > len(lines) * 0.3:
            return "list"

        return "text"

    def _split_large_content(
        self,
        content: str,
        heading_tree: dict[int, str],
        section_title: str,
        heading_level: int,
        tables: list[dict],
        source_file: str,
        start_position: int,
    ) -> list[Chunk]:
        """Split large content into smaller chunks while preserving structure."""
        chunks = []
        lines = content.split("\n")

        current_block: list[str] = []
        current_block_type = "text"
        current_size = 0
        position = start_position

        for line in lines:
            line = line
            line_size = len(line)

            # Check if this line is a table row
            is_table_row = line.strip().startswith("|")

            # Check if this line is a list item
            is_list_item = bool(re.match(r"^[·\-\*]\s+", line.strip()) or re.match(r"^（[0-9]+）", line.strip()))

            # Determine block type
            block_type = "table" if is_table_row else ("list" if is_list_item else "text")

            # If block type changes or adding this line exceeds limit
            if current_block and (
                (block_type != current_block_type and current_block_type in ("table", "list"))
                or current_size + line_size > self.max_chunk_length
            ):
                # Create chunk from current block
                block_content = "\n".join(current_block)
                if block_content.strip():
                    chunk_metadata = ChunkMetadata(
                        source_file=source_file,
                        section_title=section_title,
                        heading_tree=heading_tree,
                        content_type=current_block_type,
                        char_count=len(block_content),
                        position=position,
                        heading_level=heading_level,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_id="",
                            content=block_content.strip(),
                            token_count=self.count_tokens(block_content),
                            metadata=chunk_metadata,
                        )
                    )
                    position += 1

                current_block = []
                current_block_type = block_type
                current_size = 0

            current_block.append(line)
            current_size += line_size

            # Update block type if not already set
            if not current_block_type or current_block_type == "text":
                current_block_type = block_type

        # Don't forget the last block
        if current_block:
            block_content = "\n".join(current_block)
            if block_content.strip():
                chunk_metadata = ChunkMetadata(
                    source_file=source_file,
                    section_title=section_title,
                    heading_tree=heading_tree,
                    content_type=current_block_type,
                    char_count=len(block_content),
                    position=position,
                    heading_level=heading_level,
                )
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id="",
                        content=block_content.strip(),
                        token_count=self.count_tokens(block_content),
                        metadata=chunk_metadata,
                    )
                )

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge consecutive small chunks to meet minimum size."""
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
                combined_content = last.content + "\n\n" + chunk.content
                combined_metadata = ChunkMetadata(
                    source_file=last.metadata.source_file,
                    section_title=last.metadata.section_title,
                    heading_tree=last.metadata.heading_tree,
                    content_type=last.metadata.content_type,
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
