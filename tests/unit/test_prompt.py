from rag.generation.prompt import (
    build_system_prompt,
    build_user_prompt,
    format_contexts,
    _sanitize_content,
    format_history_message,
    SYSTEM_PROMPT,
)


class TestBuildSystemPrompt:
    def test_returns_system_prompt_constant(self):
        result = build_system_prompt()
        assert result == SYSTEM_PROMPT
        assert "专业的医疗知识问答助手" in result
        assert "绝对禁止编造信息" in result


class TestBuildUserPrompt:
    def test_formats_with_history_none(self):
        result = build_user_prompt(
            question="什么是糖尿病？",
            contexts="参考信息内容",
            history=None,
        )
        assert "什么是糖尿病？" in result
        assert "参考信息内容" in result
        assert "（无历史记录）" in result
        assert "## 对话历史" in result
        assert "## 参考信息" in result
        assert "## 用户问题" in result

    def test_formats_with_history_string(self):
        result = build_user_prompt(
            question="治疗方法有哪些？",
            contexts="上下文A",
            history="**用户**: 什么是高血压？\n**助手**: 高血压是指...",
        )
        assert "治疗方法有哪些？" in result
        assert "上下文A" in result
        assert "**用户**: 什么是高血压？" in result
        assert "（无历史记录）" not in result


class TestFormatContexts:
    def test_empty_list_returns_warning(self):
        result = format_contexts([])
        assert result == "无可用参考信息"

    def test_single_context_with_all_fields(self):
        contexts = [
            {
                "content": "糖尿病是一种代谢性疾病",
                "source": "糖尿病诊疗指南.pdf",
                "page": 15,
                "content_type": "text",
            }
        ]
        result = format_contexts(contexts)
        assert "「来源1」" in result
        assert "糖尿病诊疗指南.pdf" in result
        assert "#第15页" in result
        assert "糖尿病是一种代谢性疾病" in result
        # text type should NOT have a type prefix
        assert "[TEXT]" not in result

    def test_multiple_contexts_numbered_correctly(self):
        contexts = [
            {"content": "内容A", "source": "来源A"},
            {"content": "内容B", "source": "来源B"},
            {"content": "内容C", "source": "来源C"},
        ]
        result = format_contexts(contexts)
        assert "「来源1」" in result
        assert "「来源2」" in result
        assert "「来源3」" in result
        lines = result.split("\n")
        # Verify order
        idx_1 = next(i for i, line in enumerate(lines) if "来源1" in line)
        idx_2 = next(i for i, line in enumerate(lines) if "来源2" in line)
        idx_3 = next(i for i, line in enumerate(lines) if "来源3" in line)
        assert idx_1 < idx_2 < idx_3

    def test_content_type_table_shows_table_prefix(self):
        contexts = [
            {
                "content": "| 药物 | 剂量 |\n| 阿司匹林 | 100mg |",
                "source": "用药指南.pdf",
                "page": 3,
                "content_type": "table",
            }
        ]
        result = format_contexts(contexts)
        assert "[TABLE]" in result

    def test_content_type_list_shows_list_prefix(self):
        contexts = [
            {
                "content": "- 高血压\n- 糖尿病\n- 冠心病",
                "source": "风险因素.pdf",
                "page": 7,
                "content_type": "list",
            }
        ]
        result = format_contexts(contexts)
        assert "[LIST]" in result

    def test_content_type_text_no_prefix(self):
        contexts = [
            {
                "content": "正常血压范围是收缩压<120mmHg",
                "source": "血压指南.pdf",
                "content_type": "text",
            }
        ]
        result = format_contexts(contexts)
        assert "[TEXT]" not in result

    def test_content_exceeds_500_chars_truncated(self):
        long_content = "A" * 600
        contexts = [
            {"content": long_content, "source": "test.pdf"}
        ]
        result = format_contexts(contexts)
        assert "..." in result
        assert "A" * 500 + "..." in result
        # The full 600 chars should NOT be present
        assert long_content not in result

    def test_content_exactly_500_chars_not_truncated(self):
        exact_content = "B" * 500
        contexts = [
            {"content": exact_content, "source": "test.pdf"}
        ]
        result = format_contexts(contexts)
        # Should not have trailing "..."
        assert exact_content in result
        assert "B" * 500 + "..." not in result

    def test_content_under_500_chars_not_truncated(self):
        short_content = "短内容"
        contexts = [
            {"content": short_content, "source": "test.pdf"}
        ]
        result = format_contexts(contexts)
        assert short_content in result
        assert "短内容..." not in result

    def test_missing_source_defaults_to_unknown(self):
        contexts = [
            {"content": "一些内容"}
        ]
        result = format_contexts(contexts)
        assert "未知来源" in result

    def test_missing_page_shows_no_page_info(self):
        contexts = [
            {"content": "内容", "source": "test.pdf"}
            # page not provided
        ]
        result = format_contexts(contexts)
        assert "#第" not in result

    def test_missing_content_type_defaults_to_text(self):
        contexts = [
            {"content": "内容", "source": "test.pdf"}
            # content_type not provided
        ]
        result = format_contexts(contexts)
        assert "[TEXT]" not in result

    def test_missing_content_shows_empty(self):
        contexts = [
            {"source": "test.pdf", "page": 1}
            # content not provided
        ]
        result = format_contexts(contexts)
        # Should not crash, should have the source info
        assert "「来源1」" in result
        assert "test.pdf" in result

    def test_content_at_501_chars_truncated(self):
        content_501 = "C" * 501
        contexts = [
            {"content": content_501, "source": "test.pdf"}
        ]
        result = format_contexts(contexts)
        assert content_501[:500] + "..." in result


class TestSanitizeContent:
    def test_removes_markdown_asterisks(self):
        result = _sanitize_content("**加粗文本**")
        assert "*" not in result
        assert "加粗文本" in result

    def test_removes_markdown_hashes(self):
        result = _sanitize_content("## 标题")
        assert "#" not in result
        assert "标题" in result

    def test_removes_square_brackets(self):
        result = _sanitize_content("[链接文本](url)")
        assert "[" not in result
        assert "]" not in result
        assert "链接文本" in result

    def test_removes_backticks(self):
        result = _sanitize_content("`代码`")
        assert "`" not in result
        assert "代码" in result

    def test_removes_parentheses(self):
        result = _sanitize_content("文本(注释)")
        assert "(" not in result
        assert ")" not in result
        assert "文本" in result
        assert "注释" in result

    def test_removes_backslashes(self):
        result = _sanitize_content(r"转义符\字符")
        assert "\\" not in result
        assert "转义符" in result
        assert "字符" in result

    def test_handles_empty_string(self):
        result = _sanitize_content("")
        assert result == ""

    def test_preserves_text_without_special_chars(self):
        text = "这是一段普通的中文文本123"
        result = _sanitize_content(text)
        assert result == text


class TestFormatHistoryMessage:
    def test_user_role_shows_user_label(self):
        result = format_history_message("user", "你好")
        assert "用户" in result
        assert "助手" not in result

    def test_assistant_role_shows_assistant_label(self):
        result = format_history_message("assistant", "你好，有什么可以帮助你的？")
        assert "助手" in result
        assert "用户" not in result

    def test_unknown_role_defaults_to_assistant_label(self):
        result = format_history_message("bot", "响应内容")
        assert "助手" in result

    def test_sanitization_applied_to_content(self):
        result = format_history_message("user", "**加粗** `code` [link]")
        # The format wraps with **role_label**: so asterisks exist from the wrapper.
        # Verify the content portion is sanitized — the role markers use **, not *
        assert "加粗" in result
        assert "code" in result
        assert "link" in result
        # The markdown chars inside the content should have been stripped
        after_colon = result.split(": ", 1)[1]
        assert "*" not in after_colon
        assert "`" not in after_colon
        assert "[" not in after_colon

    def test_format_includes_bold_markers(self):
        result = format_history_message("user", "测试消息")
        assert result.startswith("**用户**: ")
        assert "测试消息" in result

    def test_assistant_format_includes_bold_markers(self):
        result = format_history_message("assistant", "回复内容")
        assert result.startswith("**助手**: ")

    def test_combined_role_and_sanitized_content(self):
        result = format_history_message("user", "查看 [文档](link) `代码`")
        assert "**用户**: 查看 文档link 代码" in result


class TestBuildUserPromptIntegration:
    """Integration-style tests that exercise build_user_prompt with
    format_contexts output."""

    def test_full_flow_with_real_contexts(self):
        contexts = [
            {
                "content": "糖尿病诊断标准：空腹血糖>=7.0mmol/L",
                "source": "糖尿病诊疗指南2023.pdf",
                "page": 42,
                "content_type": "text",
            }
        ]
        formatted_ctx = format_contexts(contexts)
        result = build_user_prompt(
            question="糖尿病的诊断标准是什么？",
            contexts=formatted_ctx,
            history="**用户**: 糖尿病有哪些类型？\n**助手**: 主要有1型和2型。",
        )
        assert "糖尿病的诊断标准是什么？" in result
        assert "糖尿病诊疗指南2023.pdf" in result
        assert "糖尿病有哪些类型？" in result
        assert "主要有1型和2型" in result
        assert "## 对话历史" in result
        assert "## 参考信息" in result
        assert "## 用户问题" in result
        assert "## 回答要求" in result
