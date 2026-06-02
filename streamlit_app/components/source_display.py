"""Source document display component."""
import streamlit as st


def render_source_card(
    file_name: str,
    source_type: str | None = None,
    relevance_score: float | None = None,
    chunk_content: str | None = None,
    page_number: int | None = None,
    verification_status: str | None = None,
    expanded: bool = False,
):
    """
    Render a single source document card.

    Args:
        file_name: Name of the source file
        source_type: Type of content (text/table/list)
        relevance_score: Relevance score (0-1)
        chunk_content: Content snippet from the document
        page_number: Page number in original document
        verification_status: Verification status string
        expanded: Whether to expand by default
    """
    with st.container():
        col1, col2 = st.columns([4, 1])

        with col1:
            icon = _get_source_icon(source_type)
            st.markdown(f"{icon} **{file_name}**")

            if page_number:
                st.caption(f"页码: {page_number}")

        with col2:
            if relevance_score is not None:
                score_pct = int(relevance_score * 100)
                if score_pct >= 80:
                    color = "green"
                elif score_pct >= 50:
                    color = "orange"
                else:
                    color = "red"
                st.markdown(f":{color}[{score_pct}%]")

        # Verification status
        if verification_status:
            status_color = "green" if verification_status == "verified" else "orange"
            st.markdown(f":{status_color}[{verification_status}]")

        # Chunk content - scrollable collapsible
        if chunk_content:
            with st.expander("查看内容", expanded=expanded):
                with st.container(height=200):
                    st.markdown(chunk_content)

        st.divider()


def _get_source_icon(source_type: str | None) -> str:
    """Get icon for source type."""
    icon_map = {
        "text": "📄",
        "table": "📊",
        "list": "📋",
        "heading": "📑",
    }
    return icon_map.get(source_type, "📄")


def render_sources_panel(sources: list[dict], title: str = "参考来源"):
    """
    Render a panel containing multiple source cards.

    Args:
        sources: List of source dicts
        title: Panel title
    """
    if not sources:
        return

    with st.expander(f"{title} ({len(sources)})", expanded=True):
        for source in sources:
            render_source_card(
                file_name=source.get("file_name", "未知"),
                source_type=source.get("source_type"),
                relevance_score=source.get("relevance_score"),
                chunk_content=source.get("chunk_content"),
                page_number=source.get("page_number"),
                verification_status=source.get("verification_status"),
            )


def render_comparison_view(sources: list[dict], reference_answer: str | None = None):
    """
    Render sources in a comparison view for evaluation.

    Args:
        sources: List of source dicts
        reference_answer: Optional reference answer for comparison
    """
    st.markdown("### 检索来源对比")

    if reference_answer:
        with st.expander("参考答案"):
            st.markdown(reference_answer)

    for idx, source in enumerate(sources, 1):
        st.markdown(f"**来源 {idx}:**")
        render_source_card(
            file_name=source.get("file_name", f"来源 {idx}"),
            source_type=source.get("source_type"),
            relevance_score=source.get("relevance_score"),
            chunk_content=source.get("chunk_content"),
            page_number=source.get("page_number"),
        )