import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st


def render_document_list(documents: list[dict]):
    if not documents:
        st.info("暂无文档，请上传医疗文档开始使用。")
        return

    for doc in documents:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{doc.get('title', '未知标题')}**")
                doc_id = doc.get("id", "")
                st.caption(f"ID: {doc_id[:8]}...")

            with col2:
                status = doc.get("status", "pending")
                if status == "completed":
                    st.success("✅ 完成")
                elif status == "processing":
                    st.warning("⏳ 处理中")
                elif status == "failed":
                    st.error("❌ 失败")
                else:
                    st.info("📋 待处理")

            with col3:
                if st.button("🗑️", key=f"del_{doc_id}"):
                    st.session_state[f"delete_{doc_id}"] = True

            total_chunks = doc.get("total_chunks")
            if total_chunks:
                st.caption(f"分片数: {total_chunks}")

            st.divider()


def render_upload_area():
    st.markdown("### 📤 上传文档")

    uploaded_file = st.file_uploader(
        "选择医疗文档",
        type=["pdf", "docx", "md", "txt"],
        help="支持 PDF、Word、Markdown、TXT 格式",
    )

    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"已选择: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        return uploaded_file

    return None


def render_upload_result(result: dict):
    if not result:
        return

    st.success(f"✅ {result.get('message', '上传成功')}")

    with st.expander("📋 上传详情"):
        doc_id = result.get("document_id", "")
        st.write(f"文档ID: {doc_id[:8]}...")
        st.write(f"文件名: {result.get('file_name', '')}")
        st.write(f"状态: {result.get('status', '')}")
