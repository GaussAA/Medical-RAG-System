import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st


def render_message(role: str, content: str):
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)


def render_citations(citations: list[dict]):
    if not citations:
        return

    with st.expander("📚 引用来源", expanded=bool(citations)):
        for i, citation in enumerate(citations, 1):
            st.markdown(f"**[{i}] {citation.get('file_name', '未知来源')}**")

            if citation.get("page_number"):
                st.write(f"页码: {citation['page_number']}")

            # Relevance score progress bar
            score = citation.get("relevance_score", 0.0)
            st.progress(float(score), text=f"相关度: {score:.2f}")

            # Show verification message if present
            if verification_msg := citation.get("verification_message"):
                st.info(f"说明: {verification_msg}")

            # Show chunk content - scrollable collapsible
            chunk = citation.get("chunk_content", "")
            if chunk:
                with st.expander("📄 查看文档片段"):
                    with st.container(height=200):
                        st.markdown(chunk)

            st.divider()


def render_confidence_badge(confidence: float):
    if confidence >= 0.8:
        color = "#28a745"
        label = "高置信度"
    elif confidence >= 0.5:
        color = "#f0ad4e"
        label = "中等置信度"
    elif confidence >= 0.3:
        color = "#fd7e14"
        label = "低置信度"
    else:
        color = "#dc3545"
        label = "不可靠"

    st.markdown(
        f'<span style="background-color:{color};color:white;padding:8px 16px;border-radius:8px;font-size:16px;font-weight:bold;">'
        f"{label}: {confidence:.2f}</span>",
        unsafe_allow_html=True,
    )


def render_warnings(warnings: list[dict]):
    if not warnings:
        return

    for warning in warnings:
        wtype = warning.get("type", "general")
        message = warning.get("message", "")

        if wtype == "medication":
            st.warning(f"💊 {message}")
        elif wtype == "diagnosis":
            st.warning(f"🏥 {message}")
        elif wtype == "emergency":
            st.error(f"🚨 {message}")
        elif wtype == "hallucination":
            st.error(f"🔍 {message}")
        elif wtype == "input_truncation":
            st.info(f"📝 {message}")
        else:
            st.info(f"ℹ️ {message}")


def render_answer(
    answer: str,
    confidence: float,
    citations: list[dict],
    warnings: list[dict],
    processing_time: float,
):
    st.markdown("### 回答")

    render_confidence_badge(confidence)

    st.markdown(answer)

    render_warnings(warnings)

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"⏱️ 耗时: {processing_time:.2f}s")
    with col2:
        if citations:
            st.info(f"📚 引用来源: {len(citations)}条")

    if citations:
        render_citations(citations)


def render_answer_streaming(answer_placeholder, confidence: float = 0.0, processing_time: float = 0.0):
    """Render answer during streaming with typing cursor."""
    st.markdown("### 回答")
    render_confidence_badge(confidence)
    return answer_placeholder


def finalize_streaming_answer(
    answer_placeholder,
    answer: str,
    confidence: float,
    citations: list[dict],
    warnings: list[dict],
    processing_time: float,
):
    """Replace streaming placeholder with complete answer."""
    with answer_placeholder.container():
        st.markdown("### 回答")
        render_confidence_badge(confidence)
        st.markdown(answer)
        render_warnings(warnings)

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"⏱️ 耗时: {processing_time:.2f}s")
        with col2:
            if citations:
                st.info(f"📚 引用来源: {len(citations)}条")

        if citations:
            render_citations(citations)


def render_history_item(session_id: str, title: str, message_count: int, updated_at: str):
    return {
        "session_id": session_id,
        "title": title or "新对话",
        "message_count": message_count,
        "updated_at": updated_at,
    }
