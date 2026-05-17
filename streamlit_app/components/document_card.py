"""Document card component for displaying document info."""
import streamlit as st
from datetime import datetime


def render_document_card(
    doc_id: str,
    title: str,
    status: str,
    file_type: str | None = None,
    total_chunks: int | None = None,
    tags: list[str] | None = None,
    created_at: str | None = None,
    on_delete_key: str | None = None,
    on_select_key: str | None = None,
):
    """
    Render a document card.

    Args:
        doc_id: Document ID
        title: Document title
        status: Processing status (pending/processing/completed/failed)
        file_type: File type (pdf/docx/md/txt)
        total_chunks: Number of chunks
        tags: List of tags
        created_at: Creation timestamp
        on_delete_key: Key for delete button
        on_select_key: Key for select checkbox
    """
    col_select, col1, col2, col3 = st.columns([0.5, 3, 1, 1])

    # Selection checkbox
    with col_select:
        if on_select_key:
            selected = st.checkbox("", key=on_select_key, label_visibility="collapsed")
        else:
            selected = False

    # Document info
    with col1:
        icon = _get_file_icon(file_type)
        st.markdown(f"{icon} **{title}**")
        st.caption(f"ID: {doc_id[:8]}...")

        if tags:
            st.markdown(" ".join([f"`{t}`" for t in tags[:5]]))

        if created_at:
            st.caption(f"创建于: {created_at[:10]}")

    # Status
    with col2:
        _render_status_badge(status)

    # Chunks count
    with col3:
        if total_chunks:
            st.metric("分片", total_chunks)

    # Delete button
    if on_delete_key and st.button("🗑️", key=on_delete_key):
        return doc_id

    st.divider()
    return None


def _render_status_badge(status: str):
    """Render status badge with appropriate styling."""
    status_config = {
        "completed": ("✅ 完成", "green"),
        "processing": ("⏳ 处理中", "orange"),
        "failed": ("❌ 失败", "red"),
        "pending": ("📋 待处理", "blue"),
        "archived": ("📦 已归档", "gray"),
    }

    label, color = status_config.get(status, ("❓ 未知", "gray"))
    st.markdown(f":{color}[{label}]")


def _get_file_icon(file_type: str | None) -> str:
    """Get icon for file type."""
    icon_map = {
        "pdf": "📕",
        "docx": "📘",
        "doc": "📘",
        "md": "📗",
        "markdown": "📗",
        "txt": "📄",
    }
    return icon_map.get(file_type, "📄")


def render_document_stats(documents: list[dict]) -> dict:
    """
    Calculate and render document statistics.

    Returns:
        Dict with stats: total, by_status, by_type
    """
    stats = {
        "total": len(documents),
        "by_status": {},
        "by_type": {},
    }

    for doc in documents:
        status = doc.get("status", "unknown")
        file_type = doc.get("file_type", "unknown")

        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1

    # Render as metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总数", stats["total"])
    col2.metric("已完成", stats["by_status"].get("completed", 0))
    col3.metric("处理中", stats["by_status"].get("processing", 0))
    col4.metric("失败", stats["by_status"].get("failed", 0))

    return stats


def render_batch_result(result: dict):
    """
    Render batch upload result summary.

    Args:
        result: Batch upload result dict
    """
    col1, col2, col3 = st.columns(3)

    col1.metric("成功", result.get("succeeded", 0))
    col2.metric("失败", result.get("failed", 0))
    col3.metric("重复", result.get("duplicate", 0))

    items = result.get("items", [])
    if items:
        with st.expander("详细结果"):
            for item in items:
                status = item.get("status", "unknown")
                file_name = item.get("file_name", "unknown")
                error = item.get("error_message", "")

                status_icon = {
                    "processing": "⏳",
                    "completed": "✅",
                    "failed": "❌",
                    "duplicate": "⚠️",
                }.get(status, "❓")

                if status == "failed" and error:
                    st.markdown(f"{status_icon} {file_name}: {error}")
                else:
                    st.markdown(f"{status_icon} {file_name}: {status}")