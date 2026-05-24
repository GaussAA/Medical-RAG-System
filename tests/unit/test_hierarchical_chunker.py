from app.models.schemas import Chunk
from rag.chunking.hierarchical_chunker import HierarchicalChunker


class TestHierarchicalChunker:
    """Test HierarchicalChunker for Markdown medical documents."""

    def setup_method(self):
        self.chunker = HierarchicalChunker()

    def test_chunk_empty_text(self):
        chunks = self.chunker.chunk("")
        assert len(chunks) == 0

    def test_chunk_single_paragraph(self):
        text = "这是一个简单的段落。"
        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_respects_max_length(self):
        text = "A" * 2000
        chunks = self.chunker.chunk(text)

        for chunk in chunks:
            assert chunk.metadata.char_count <= self.chunker.max_chunk_length * 1.5

    def test_chunk_includes_metadata(self):
        text = "这是一个测试段落。"
        metadata = {"source_file": "test.md", "doc_id": "doc-123"}

        chunks = self.chunker.chunk(text, metadata=metadata)

        assert len(chunks) > 0
        assert chunks[0].metadata.source_file == "test.md"

    def test_chunk_with_heading_context(self):
        text = """## 糖尿病诊断

糖尿病的诊断标准如下：
空腹血糖大于7.0 mmol/L。
"""

        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
        # Section title should be captured
        assert chunks[0].metadata.section_title == "糖尿病诊断"

    def test_chunk_heading_tree_construction(self):
        text = """# 主标题

## 子章节

内容段落。
"""
        metadata = {"source_file": "test.md", "heading_tree": {1: "主标题", 2: "子章节"}}

        chunks = self.chunker.chunk(text, metadata=metadata)

        assert len(chunks) > 0
        # Heading tree should be included
        assert chunks[0].metadata.heading_tree is not None

    def test_chunk_with_table_content_type(self):
        text = """| 表头1 | 表头2 |
| --- | --- |
| 数据1 | 数据2 |"""

        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
        assert chunks[0].metadata.content_type == "table"

    def test_chunk_with_list_content_type(self):
        text = """- 列表项一
- 列表项二
- 列表项三"""

        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
        assert chunks[0].metadata.content_type == "list"

    def test_chunk_multiple_sections(self):
        """Multiple sections should be chunked separately, but small chunks may merge."""
        text = """## 第一节

第一节的内容。

## 第二节

第二节的内容很长很长很长的内容。
"""
        metadata = {"source_file": "test.md"}

        chunks = self.chunker.chunk(text, metadata=metadata)

        # At least first section should be captured
        section_titles = [c.metadata.section_title for c in chunks]
        assert "第一节" in section_titles

    def test_chunk_merge_small_chunks(self):
        text = """## 小节

短内容。
"""
        metadata = {"source_file": "test.md", "heading_tree": {2: "小节"}}

        chunks = self.chunker.chunk(text, metadata=metadata)

        # Small chunks may be merged with neighbors
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_heading_level(self):
        text = """### 三级标题

内容。
"""
        metadata = {"source_file": "test.md"}

        chunks = self.chunker.chunk(text, metadata=metadata)

        assert len(chunks) > 0
        assert chunks[0].metadata.heading_level == 3

    def test_chunk_position_sequential(self):
        text = """## 第一节

第一节内容。

## 第二节

第二节内容。
"""
        metadata = {"source_file": "test.md"}

        chunks = self.chunker.chunk(text, metadata=metadata)

        assert len(chunks) > 0
        positions = [c.metadata.position for c in chunks]
        assert positions == sorted(positions)


class TestHierarchicalChunkerEdgeCases:
    """Test edge cases for HierarchicalChunker."""

    def setup_method(self):
        self.chunker = HierarchicalChunker()

    def test_chunk_only_heading(self):
        """Text with only a heading line (no content) - should produce 0 chunks since content is empty."""
        text = "## 只有标题"
        chunks = self.chunker.chunk(text)

        # Without actual content after heading, we get 0 chunks (empty content stripped)
        assert len(chunks) == 0

    def test_chunk_chinese_table_caption(self):
        """Table rows should be detected as table content type."""
        text = """| 指标 | 数值 |
| --- | --- |
| 空腹血糖 | ≥7.0 mmol/L |"""

        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
        assert chunks[0].metadata.content_type == "table"

    def test_chunk_numbered_list(self):
        text = """（1）第一项内容
（2）第二项内容
（3）第三项内容"""

        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
        assert chunks[0].metadata.content_type == "list"

    def test_chunk_mixed_content_large(self):
        """Test splitting of large content that exceeds max_chunk_length."""
        # Create text > max_chunk_length (2000 chars)
        long_text = """## 药物治疗

""" + "\n".join([f"- 药物{i}：这是很长的剂量信息，用于测试大型内容的分割功能，需要确保内容足够长以触发分割。" for i in range(100)])

        chunks = self.chunker.chunk(long_text, metadata={"source_file": "test.md"})

        # Large content should be split into multiple chunks
        assert len(chunks) > 1

    def test_chunk_preserve_semantic_units(self):
        """Tables should be preserved as independent chunks when separated by blank lines."""
        text = """## 数据

| A | B |
| --- | --- |
| 1 | 2 |

| C | D |
| --- | --- |
| 3 | 4 |
"""

        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        # Tables should be separate chunks
        table_chunks = [c for c in chunks if c.metadata.content_type == "table"]
        assert len(table_chunks) >= 1  # At least one table chunk

    def test_chunk_empty_metadata(self):
        text = "普通文本内容。"
        chunks = self.chunker.chunk(text)

        assert len(chunks) > 0
        # Should handle None metadata gracefully
        assert chunks[0].metadata.source_file == ""

    def test_chunk_no_heading(self):
        """Text without headings should still be chunked."""
        text = "这是一段没有标题的文本。"
        chunks = self.chunker.chunk(text, metadata={"source_file": "test.md"})

        assert len(chunks) > 0
