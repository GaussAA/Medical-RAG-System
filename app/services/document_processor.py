from pathlib import Path
from typing import Any

from app.models.schemas import Chunk as SchemaChunk
from app.models.schemas import ParsedDocument, RetrievedNode
from rag.chunking.hierarchical_chunker import HierarchicalChunker
from rag.parser import parse_document, parse_document_with_headings


class DocumentProcessor:
    """Handles document parsing, chunking, and vectorization."""

    def __init__(self):
        self.chunker = HierarchicalChunker()

    async def parse(self, file_path: Path) -> ParsedDocument:
        """Parse document and return structured content."""
        return await parse_document(file_path)

    async def parse_with_headings(self, file_path: Path) -> tuple[ParsedDocument, list[dict]]:
        """Parse Markdown document and extract heading tree structure."""
        return await parse_document_with_headings(file_path)

    def chunk(self, text_content: str, metadata: dict[str, Any]) -> list[SchemaChunk]:
        """Split text content into semantic chunks."""
        return self.chunker.chunk(text_content, metadata=metadata)

    def create_retrieved_nodes(
        self,
        doc_id: str,
        chunks: list[SchemaChunk],
        source_file: str,
    ) -> list[RetrievedNode]:
        """Convert chunks to RetrievedNode list for indexing."""
        nodes = []
        for i, chunk in enumerate(chunks):
            node = RetrievedNode(
                node_id=chunk.chunk_id,  # Use UUID from chunk (matches PostgreSQL)
                content=chunk.content,
                score=1.0,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": chunk.chunk_id,
                    "source_file": source_file,
                    "section_title": chunk.metadata.section_title,
                    "heading_tree": chunk.metadata.heading_tree,
                    "content_type": chunk.metadata.content_type,
                    "char_count": chunk.metadata.char_count,
                    "position": chunk.metadata.position,
                    "heading_level": chunk.metadata.heading_level,
                },
            )
            nodes.append(node)
        return nodes

    def save_processed_text(self, file_path: Path, text_content: str) -> Path:
        """Save processed text to data/processed directory."""
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        original_name = file_path.stem
        processed_file_path = processed_dir / f"{original_name}.txt"
        with open(processed_file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return processed_file_path
