from pathlib import Path

from app.models.schemas import ParsedDocument
from rag.parser.markdown_parser import MarkdownParser


async def parse_document(file_path: str | Path) -> ParsedDocument:
    """Parse a Markdown document and return structured content."""
    parser = MarkdownParser()
    return await parser.parse(file_path)


async def parse_document_with_headings(file_path: str | Path) -> tuple[ParsedDocument, list[dict]]:
    """
    Parse a Markdown document and extract heading tree structure.

    Returns:
        tuple: (ParsedDocument, heading_tree_list)
        heading_tree_list contains heading metadata for DB storage
    """
    parser = MarkdownParser()
    return await parser.parse_with_headings(file_path)
