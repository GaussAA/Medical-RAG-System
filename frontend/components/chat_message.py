"""Chat message rendering component."""

import streamlit as st


def render_message(role: str, content: str, metadata: dict | None = None):
    """
    Render a single chat message.

    Args:
        role: Message role ("user" or "assistant")
        content: Message content
        metadata: Optional metadata dict containing confidence, citations, warnings
    """
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)
            if metadata:
                _render_message_metadata(metadata)


def _render_message_metadata(metadata: dict):
    """Render assistant message metadata (confidence, citations, warnings)."""
    # Confidence badge
    if metadata.get("confidence"):
        _render_confidence_indicator(metadata["confidence"])

    # Citations
    if metadata.get("citations"):
        _render_citations_list(metadata["citations"])

    # Warnings
    if metadata.get("warnings"):
        _render_warnings_list(metadata["warnings"])


def _render_confidence_indicator(confidence: float):
    """Render confidence indicator."""
    if confidence >= 0.8:
        color = "green"
        label = "高置信度"
    elif confidence >= 0.5:
        color = "orange"
        label = "中等置信度"
    elif confidence >= 0.3:
        color = "yellow"
        label = "低置信度"
    else:
        color = "red"
        label = "不可靠"

    st.caption(f":{color}[{label}: {confidence:.2f}]")


def _render_citations_list(citations: list[dict]):
    """Render citations in an expandable section."""
    with st.expander("来源引用", expanded=False):
        for i, citation in enumerate(citations, 1):
            file_name = citation.get("file_name", "未知来源")
            st.markdown(f"**[{i}] {file_name}**")

            if page := citation.get("page_number"):
                st.caption(f"页码: {page}")

            if score := citation.get("relevance_score"):
                st.progress(float(score), text=f"相关度: {score:.2f}")

            if chunk := citation.get("chunk_content"):
                with st.expander("查看文档片段"):
                    with st.container(height=200):
                        st.markdown(chunk)


def _render_warnings_list(warnings: list[dict]):
    """Render warnings list."""
    for warning in warnings:
        wtype = warning.get("type", "general")
        message = warning.get("message", "")

        icon_map = {
            "medication": "💊",
            "diagnosis": "🏥",
            "emergency": "🚨",
            "hallucination": "🔍",
            "input_truncation": "📝",
        }
        icon = icon_map.get(wtype, "ℹ️")

        if wtype == "emergency" or wtype == "hallucination":
            st.error(f"{icon} {message}")
        else:
            st.warning(f"{icon} {message}")


def render_citations_inline(citations: list[dict]):
    """Render citations inline (compact view)."""
    if not citations:
        return

    st.markdown("**来源:** ")
    for citation in citations:
        file_name = citation.get("file_name", "未知")
        st.markdown(f"- {file_name}", unsafe_allow_html=False)
