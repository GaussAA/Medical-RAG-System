SYSTEM_PROMPT = """你是一个专业的医疗知识问答助手。你的职责是：
1. 基于提供的参考信息**准确、完整**地回答用户问题
2. 在回答中**明确标注信息来源**，使用「来源X」格式
3. 如果参考信息不足以回答问题，**明确告知用户**哪些信息缺失
4. **绝对禁止编造信息**，只基于提供的参考内容回答
5. 对于涉及药物、诊断、紧急情况的内容，**必须**添加适当的风险提示

回答质量要求：
- 答案必须**直接针对**用户问题，不要泛泛而谈
- **直接引用原文中的关键陈述**，不要过度概括或改写
- 每个医学陈述都需要有对应的来源标注
- 引用时尽量使用原文表述，而非重新组织语言
- 如果多个上下文提供的信息不一致，需要标注这种差异
- 答案应包含足够的细节来解答问题

风险提示格式：
⚠️ 重要提示：[相关风险说明]
"""

USER_PROMPT_TEMPLATE = """
## 对话历史
{history}

## 参考信息
{contexts}

## 用户问题
{question}

## 回答要求
1. **直接回答问题**：答案开头必须针对问题给出明确回答，不要先解释概念
2. **每个医学陈述都必须标注来源**，格式为「来源X」，不得遗漏
3. 如果参考信息中有多个来源同时支持一个论点，也要逐一标注
4. 答案应包含足够的具体细节（剂量、疗程、注意事项等）以解决用户的问题
5. 不得省略参考信息中的重要细节，如数值（剂量、时间等）、具体方法名称等
6. 禁止在回答中自行评估置信度，后端系统会自动计算
7. 如果问题超出参考信息范围，明确说明"根据提供的信息，无法完全回答此问题"
8. **重要**：对于高危人群、筛查方法、诊断标准等关键信息，必须尽可能完整地列出参考信息中的相关内容
"""

FALLBACK_RESPONSES = {
    "no_results": {
        "answer": "抱歉，我在知识库中没有找到与您问题相关的信息。",
        "suggestions": [
            "请尝试使用更通用的关键词",
            "检查问题是否与医学相关",
            "联系医疗专业人员获取帮助",
        ],
        "confidence": 0.0,
    },
    "low_confidence": {
        "answer": "根据找到的部分信息，您的问题可能的答案是：\n{partial_answer}",
        "warning": "⚠️ 警告：此答案的置信度较低({confidence})，仅供参考，请以专业医疗建议为准。",
        "confidence": "{calculated_confidence}",
    },
}


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(question: str, contexts: str, history: str | None = None) -> str:
    return USER_PROMPT_TEMPLATE.format(
        question=question,
        contexts=contexts,
        history=history or "（无历史记录）",
    )


def format_contexts(contexts: list[dict]) -> str:
    if not contexts:
        return "无可用参考信息"

    formatted = []
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get("source", "未知来源")
        page = ctx.get("page", "")
        content = ctx.get("content", "")
        content_type = ctx.get("content_type", "text")  # text, table, list

        page_info = f"#第{page}页" if page else ""
        type_hint = f"[{content_type.upper()}] " if content_type != "text" else ""

        # Truncate very long content but preserve完整性
        if len(content) > 500:
            content = content[:500] + "..."

        formatted.append(f"「来源{i}」{type_hint}{source}{page_info}:\n{content}\n")

    return "\n".join(formatted)


def _sanitize_content(content: str) -> str:
    """Remove markdown special characters to prevent prompt injection."""
    import re

    return re.sub(r"[*#\[\]()`\\]", "", content)


def format_history_message(role: str, content: str) -> str:
    role_label = "用户" if role == "user" else "助手"
    safe_content = _sanitize_content(content)
    return f"**{role_label}**: {safe_content}"
