import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# CSS for citation anchor links
CITATION_LINK_CSS = """
<style>
.citation-link {
    color: #1a73e8;
    text-decoration: none;
    cursor: pointer;
    border-bottom: 1px dashed #1a73e8;
    padding: 0 2px;
    transition: all 0.15s ease;
}
.citation-link:hover {
    background-color: #1a73e8;
    color: white;
    border-bottom-color: #1a73e8;
}
.citation-anchor {
    scroll-margin-top: 80px;
}
</style>
"""


def _process_citation_tags(text: str) -> str:
    """Convert citation tags in answer text to clickable links.

    Uses JavaScript scrollIntoView for reliable navigation in Streamlit,
    since HTML fragment anchors (#citation-N) can conflict with Streamlit's
    DOM structure when multiple answers with citations exist on the same page.

    Handles both 「来源N」 and 【来源N】 formats.
    """
    # Streamlit wraps each st.markdown in its own container, and expander
    # content renders inside a <details> element. Native #fragment anchors
    # can't reliably cross these boundaries to find the target <span>.
    # Instead we use onclick scrollIntoView which queries the whole document.
    citation_link = (
        '<a href="#" onclick="'
        "document.getElementById('citation-\\1').scrollIntoView("
        "{behavior:'smooth',block:'center'});"
        'return false;" class="citation-link">'
    )

    text = re.sub(
        r'「来源(\d+)」',
        citation_link + r'「来源\1」</a>',
        text,
    )
    text = re.sub(
        r'【来源(\d+)】',
        citation_link + r'【来源\1】</a>',
        text,
    )
    return text


def render_message(role: str, content: str):
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)


def render_citations(citations: list[dict], show_anchors: bool = True):
    if not citations:
        return

    st.markdown(CITATION_LINK_CSS, unsafe_allow_html=True)

    total = len(citations)
    with st.expander("📚 引用来源", expanded=bool(citations)):
        for i, citation in enumerate(citations, 1):
            source_id = citation.get("source_id", i)
            # Hidden anchor target for citation link scrolling
            if show_anchors:
                st.markdown(
                    f'<span id="citation-{source_id}" class="citation-anchor"></span>',
                    unsafe_allow_html=True,
                )
            st.markdown(f"**[{i}] {citation.get('file_name', '未知来源')}**")

            if citation.get("page_number"):
                st.write(f"页码: {citation['page_number']}")

            # Show ranking position instead of ambiguous normalized score
            rank_progress = 1.0 - (i - 1) / total if total > 1 else 1.0
            st.progress(rank_progress, text=f"来源排名: #{i}/{total}")

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

    style = f"background-color:{color};color:white;padding:8px 16px;border-radius:8px;font-size:16px;font-weight:bold;"
    st.markdown(
        f'<span style="{style}">{label}: {confidence:.2f}</span>',
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

    # Render answer with clickable citation links
    st.markdown(_process_citation_tags(answer), unsafe_allow_html=True)

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
        st.markdown(_process_citation_tags(answer), unsafe_allow_html=True)
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
