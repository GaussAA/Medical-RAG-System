import pytest

from rag.parser.markdown_parser import MarkdownParser


class TestMarkdownParser:
    """Test MarkdownParser for Markdown-only document processing."""

    @pytest.mark.asyncio
    async def test_markdown_parser_can_parse_md(self):
        parser = MarkdownParser()
        assert parser.can_parse("test.md")
        assert parser.can_parse("test.markdown")
        assert not parser.can_parse("test.pdf")
        assert not parser.can_parse("test.docx")
        assert not parser.can_parse("test.txt")

    @pytest.mark.asyncio
    async def test_markdown_parser_parse_with_headings(self, tmp_path):
        """Test parsing Markdown with heading tree extraction."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "# Main Title\n\n## Section 1\n\nContent one.\n\n## Section 2\n\nContent two.\n",
            encoding="utf-8",
        )

        parser = MarkdownParser()
        parsed_doc, heading_tree = await parser.parse_with_headings(str(test_file))

        assert len(parsed_doc.text_content) > 0
        assert len(heading_tree) == 3  # H1 + 2 H2s
        assert heading_tree[0]["title"] == "Main Title"
        assert heading_tree[1]["title"] == "Section 1"
        assert heading_tree[2]["title"] == "Section 2"

    @pytest.mark.asyncio
    async def test_markdown_parser_table_detection(self, tmp_path):
        """Test table detection with caption in Markdown."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "**表1 数据表**\n\n| Col1 | Col2 |\n| --- | --- |\n| A | B |",
            encoding="utf-8",
        )

        parser = MarkdownParser()
        parsed_doc, _ = await parser.parse_with_headings(str(test_file))

        # Table with caption is detected
        assert len(parsed_doc.tables) == 1
        assert parsed_doc.tables[0].caption is not None

    @pytest.mark.asyncio
    async def test_markdown_parser_parse_empty_file(self, tmp_path):
        """Test parsing an empty file."""
        test_file = tmp_path / "empty.md"
        test_file.write_text("", encoding="utf-8")

        parser = MarkdownParser()
        parsed_doc, heading_tree = await parser.parse_with_headings(str(test_file))

        assert parsed_doc.text_content == ""
        assert len(heading_tree) == 0

    @pytest.mark.asyncio
    async def test_markdown_parser_nested_headings(self, tmp_path):
        """Test nested heading levels."""
        test_file = tmp_path / "test.md"
        test_file.write_text(
            "# H1\n\n## H2\n\n### H3\n\n#### H4\n",
            encoding="utf-8",
        )

        parser = MarkdownParser()
        _, heading_tree = await parser.parse_with_headings(str(test_file))

        assert len(heading_tree) == 4
        assert heading_tree[0]["level"] == 1
        assert heading_tree[1]["level"] == 2
        assert heading_tree[2]["level"] == 3
        assert heading_tree[3]["level"] == 4
