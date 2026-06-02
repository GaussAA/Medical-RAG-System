import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import markdown  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

from app.models.schemas import ParsedDocument, TableData
from rag.parser.base import BaseParser


@dataclass
class HeadingNode:
    """Represents a heading in the Markdown document tree."""

    level: int  # 1-6 for H1-H6
    title: str
    line_number: int
    parent: "HeadingNode | None" = None
    children: list["HeadingNode"] = field(default_factory=list)
    content_start: int = 0
    content_end: int = 0


class MarkdownParser(BaseParser):
    supported_extensions = [".md", ".markdown"]

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
    TABLE_CAPTION_RE = re.compile(r"^\*\*(表[一二三四五六七八九十\d]+[^\*]*)\*\*$")
    LIST_BULLET_RE = re.compile(r"^[·\-\*]\s+")
    LIST_NUMBERED_RE = re.compile(r"^（([0-9]+)）")
    DRUG_DOSAGE_RE = re.compile(r"每次?\s*[XxXx·\d\.]+\s*mg|每日?[Xx\d]+次|剂量?\s*[Xx\d\.]+\s*mg")

    async def parse(self, file_path: str | Path) -> ParsedDocument:
        file_path = Path(file_path)

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        html = markdown.markdown(content, extensions=["tables"])

        text_parts = []
        tables = []

        soup = BeautifulSoup(html, "html.parser")

        for element in soup.children:
            if element.name == "table":  # type: ignore[attr-defined]
                table_data = self._parse_html_table(element)
                if table_data:
                    tables.append(table_data)
            elif element.get_text().strip():
                text_parts.append(element.get_text())

        return ParsedDocument(
            doc_id=str(uuid.uuid4()),
            title=file_path.stem,
            source=str(file_path),
            content_type="mixed" if tables else "text",
            text_content="\n\n".join(text_parts),
            tables=tables,
            metadata={"original_length": len(content)},
        )

    async def parse_with_headings(self, file_path: str | Path) -> tuple[ParsedDocument, list[dict[str, Any]]]:
        """
        Parse Markdown file and extract heading tree structure.

        Returns:
            tuple: (ParsedDocument, heading_tree_list)
            heading_tree_list is a list of dicts with heading info for DB storage
        """
        file_path = Path(file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Build heading tree
        root_headings = []
        heading_stack: list[HeadingNode] = []

        for line_num, line in enumerate(lines, start=1):
            match = self.HEADING_RE.match(line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                node = HeadingNode(
                    level=level,
                    title=title,
                    line_number=line_num,
                )

                # Find parent (last heading with lower level)
                while heading_stack and heading_stack[-1].level >= level:
                    heading_stack.pop()

                if heading_stack:
                    node.parent = heading_stack[-1]
                    node.parent.children.append(node)
                else:
                    root_headings.append(node)

                heading_stack.append(node)

        # Flatten heading tree with position info for DB
        heading_tree_list = []
        position = 0

        def flatten_tree(nodes: list[HeadingNode], parent_position: int | None = None) -> None:
            nonlocal position
            for node in nodes:
                heading_info = {
                    "level": node.level,
                    "title": node.title,
                    "position": position,
                    "parent_position": parent_position,
                    "line_number": node.line_number,
                }
                heading_tree_list.append(heading_info)
                current_position = position
                position += 1
                flatten_tree(node.children, parent_position=current_position)

        flatten_tree(root_headings)

        # Extract content blocks with table info
        tables = self._extract_tables_with_captions(lines)
        content = "".join(lines)

        html = markdown.markdown(content, extensions=["tables"])
        soup = BeautifulSoup(html, "html.parser")

        text_parts = []
        for element in soup.children:
            if element.name == "table":  # type: ignore[attr-defined]
                continue  # Tables handled separately
            elif element.get_text().strip():
                text_parts.append(element.get_text())

        parsed_doc = ParsedDocument(
            doc_id=str(uuid.uuid4()),
            title=file_path.stem,
            source=str(file_path),
            content_type="mixed" if tables else "text",
            text_content=content,
            tables=tables,
            metadata={
                "original_length": len(content),
                "line_count": len(lines),
            },
        )

        return parsed_doc, heading_tree_list

    def _extract_tables_with_captions(self, lines: list[str]) -> list[TableData]:
        """Extract tables with their captions from Markdown source."""
        tables = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check for table caption pattern like **表2 xxx**
            caption_match = self.TABLE_CAPTION_RE.match(line)
            if caption_match:
                caption = caption_match.group(1)
                table_lines = [lines[i]]

                # Collect table lines following the caption
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and (lines[i].strip().startswith("|") or lines[i].strip().startswith("-"))
                ):
                    table_lines.append(lines[i])
                    i += 1

                # Parse the table
                table_text = "".join(table_lines)
                table_data = self._parse_markdown_table(table_text, caption)
                if table_data:
                    tables.append(table_data)
                continue

            i += 1

        return tables

    def _parse_markdown_table(self, text: str, caption: str | None = None) -> TableData:
        """Parse a Markdown table from text."""
        lines = text.strip().split("\n")
        if not lines:
            return TableData(headers=[], rows=[], caption=caption)

        # Filter out separator lines (|---|---|)
        content_lines = [line for line in lines if not re.match(r"^\|[\s\-:|]+\|$", line.strip())]

        if not content_lines:
            return TableData(headers=[], rows=[], caption=caption)

        rows = []
        headers = []

        for idx, line in enumerate(content_lines):
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = [c for c in cells if c]

            if idx == 0:
                headers = cells
            else:
                if cells:
                    rows.append(cells)

        return TableData(
            headers=headers,
            rows=rows,
            caption=caption,
        )

    def extract_tables(self, content: Any) -> list[TableData]:
        if isinstance(content, (str, Path)):
            with open(content, encoding="utf-8") as f:
                content = f.read()

        html = markdown.markdown(content, extensions=["tables"])
        soup = BeautifulSoup(html, "html.parser")

        tables = []
        for table in soup.find_all("table"):
            table_data = self._parse_html_table(table)
            if table_data:
                tables.append(table_data)

        return tables

    def _parse_html_table(self, table) -> TableData:
        headers = []
        rows = []

        header_row = table.find("tr")
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all(["th", "td"])]

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text().strip() for td in row.find_all("td")]
            if cells:
                rows.append(cells)

        if not rows:
            return TableData(headers=[], rows=[], caption=None)

        return TableData(
            headers=headers,
            rows=rows,
            caption=None,
        )
