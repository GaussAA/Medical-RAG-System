import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="文档管理", page_icon="📁")
st.title("📁 知识库管理")

# 初始化 session state
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = set()

# ========== 侧边栏：过滤搜索 ==========
with st.sidebar:
    st.markdown("### 🔍 过滤搜索")

    status_filter = st.selectbox(
        "状态",
        options=[None, "pending", "processing", "completed", "failed", "archived"],
        format_func=lambda x: "全部" if x is None else x,
        key="status_filter",
    )

    tags_filter = st.text_input("标签 (逗号分隔)", "", key="tags_filter_input")

    file_type_filter = st.selectbox(
        "文件类型",
        options=[None, "pdf", "docx", "md", "txt"],
        format_func=lambda x: "全部" if x is None else x,
        key="file_type_filter",
    )

    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("开始日期", value=None, key="date_from")
    with col2:
        date_to = st.date_input("结束日期", value=None, key="date_to")

    page_size = st.selectbox("每页数量", [20, 50, 100], index=0, key="page_size_select")

    if st.button("🔄 应用过滤", use_container_width=True, key="apply_filter"):
        st.rerun()

# ========== 主区域：文档列表 ==========
st.markdown("### 📚 文档列表")

# 构建查询参数
params = []
if status_filter:
    params.append(f"status={status_filter}")
if tags_filter:
    params.append(f"tags={tags_filter}")
if file_type_filter:
    params.append(f"file_type={file_type_filter}")
if date_from:
    params.append(f"date_from={date_from.isoformat()}")
if date_to:
    params.append(f"date_to={date_to.isoformat()}")
params.append(f"page_size={page_size}")

query_string = "&".join(params) if params else ""

try:
    response = requests.get(f"{API_BASE}/documents?{query_string}", timeout=10)
    if response.status_code == 200:
        data = response.json()
        documents = data.get("documents", [])
        total = data.get("total", 0)
        current_page = data.get("page", 1)
        page_size_resp = data.get("page_size", 20)
    else:
        documents = []
        total = 0
        current_page = 1
        page_size_resp = page_size
except Exception as e:
    st.error(f"加载文档列表失败: {str(e)}")
    documents = []
    total = 0
    current_page = 1
    page_size_resp = page_size

# 显示统计信息
col1, col2, col3 = st.columns(3)
col1.metric("总文档数", total)
col2.metric("当前页", current_page)
col3.metric("每页", page_size_resp)

st.markdown("---")

# ========== 批量操作工具栏 ==========
st.markdown("#### 批量操作")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    select_all = st.checkbox("全选", key="select_all")

with col2:
    pass  # 占位

with col3:
    pass  # 占位

selected_docs = st.session_state.selected_docs

if st.button("🗑️ 批量删除", use_container_width=True, key="batch_delete_btn") and selected_docs:
    with st.container():
        st.warning(f"确定要删除 {len(selected_docs)} 个文档吗？")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ 确认删除", key="confirm_batch_delete"):
                try:
                    response = requests.post(
                        f"{API_BASE}/documents/batch-delete",
                        json={"ids": list(selected_docs)},
                        timeout=30,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        deleted = len(result.get("deleted", []))
                        st.success(f"成功删除 {deleted} 个文档")
                        st.session_state.selected_docs = set()
                        st.rerun()
                except Exception as e:
                    st.error(f"批量删除失败: {str(e)}")
        with col_cancel:
            if st.button("❌ 取消", key="cancel_batch_delete"):
                st.rerun()

new_status = st.selectbox(
    "批量更新状态",
    options=[None, "archived", "completed", "pending"],
    format_func=lambda x: "选择状态..." if x is None else x,
    key="batch_status",
)
if (
    st.button("📝 更新状态", use_container_width=True, key="batch_update_status_btn")
    and selected_docs
    and new_status
):
    try:
        response = requests.patch(
            f"{API_BASE}/documents/batch-update",
            json={"ids": list(selected_docs), "status": new_status},
            timeout=30,
        )
        if response.status_code == 200:
            st.success(f"成功更新 {len(selected_docs)} 个文档状态")
            st.rerun()
    except Exception as e:
        st.error(f"批量更新失败: {str(e)}")

tags_input = st.text_input("批量添加标签", key="batch_tags_input", placeholder="逗号分隔")
if (
    st.button("🏷️ 添加标签", use_container_width=True, key="batch_add_tags_btn")
    and selected_docs
    and tags_input
):
    tags = [t.strip() for t in tags_input.split(",")]
    try:
        response = requests.patch(
            f"{API_BASE}/documents/batch-update",
            json={"ids": list(selected_docs), "tags": tags, "operation": "add"},
            timeout=30,
        )
        if response.status_code == 200:
            st.success(f"成功添加标签到 {len(selected_docs)} 个文档")
            st.rerun()
    except Exception as e:
        st.error(f"批量添加标签失败: {str(e)}")

# ========== 文档列表 ==========
if not documents:
    st.info("暂无文档，请上传医疗文档开始使用。")
else:
    # 处理全选
    if select_all:
        st.session_state.selected_docs = {doc.get("id", "") for doc in documents}
    else:
        # 移除不在当前页的文档
        current_ids = {doc.get("id", "") for doc in documents}
        st.session_state.selected_docs = {
            doc_id for doc_id in st.session_state.selected_docs if doc_id in current_ids
        }

    for doc in documents:
        doc_id = doc.get("id", "")

        with st.container():
            col_select, col1, col2, col3, col4 = st.columns([0.5, 3, 1, 1, 1])

            with col_select:
                cb = st.checkbox("选择", key=f"sel_{doc_id}", label_visibility="collapsed")
                if cb:
                    st.session_state.selected_docs.add(doc_id)
                elif doc_id in st.session_state.selected_docs:
                    st.session_state.selected_docs.discard(doc_id)

            with col1:
                st.markdown(f"**{doc.get('title', '未知标题')}**")
                st.caption(f"ID: {doc_id[:8]}...")

                # 显示标签
                tags = doc.get("tags", [])
                if tags:
                    st.markdown(" ".join([f"`{t}`" for t in tags[:5]]))

            with col2:
                status = doc.get("status", "pending")
                if status == "completed":
                    st.success("✅ 完成")
                elif status == "processing":
                    st.warning("⏳ 处理中")
                elif status == "failed":
                    st.error("❌ 失败")
                elif status == "archived":
                    st.info("📋 已归档")
                else:
                    st.info("📋 待处理")

            with col3:
                total_chunks = doc.get("total_chunks")
                if total_chunks:
                    st.metric("分片", total_chunks)

            with col4:
                # 删除按钮直接显示在列中
                if st.button("🗑️", key=f"del_{doc_id}"):
                    try:
                        del_response = requests.delete(
                            f"{API_BASE}/documents/{doc_id}",
                            timeout=5,
                        )
                        if del_response.status_code == 200:
                            st.success("删除成功")
                            st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {str(e)}")

            st.divider()

# ========== 分页 ==========
if total > page_size_resp:
    st.markdown("---")
    col_prev, col_next = st.columns(2)
    with col_prev:
        if current_page > 1:
            if st.button("◀️ 上一页"):
                st.rerun()
    with col_next:
        if len(documents) == page_size_resp:
            if st.button("下一页 ▶️"):
                st.rerun()

# ========== 上传区域 ==========
st.markdown("---")
st.markdown("### 📤 上传文档")

uploaded_files = st.file_uploader(
    "选择医疗文档（支持批量上传）",
    type=["md", "markdown"],
    accept_multiple_files=True,
    help="支持 Markdown 格式，批量上传时统一向量化",
)

if uploaded_files:
    file_count = len(uploaded_files)
    total_size = sum(f.size for f in uploaded_files) / 1024

    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"已选择 {file_count} 个文件，共 {total_size:.1f} KB")
    with col2:
        if st.button("📤 上传", type="primary"):
            with st.spinner("上传中..."):
                try:
                    # Build multipart form with multiple files
                    files = []
                    for f in uploaded_files:
                        files.append(
                            ("files", (f.name, f.getvalue(), f.type))
                        )

                    response = requests.post(
                        f"{API_BASE}/documents/upload/batch",
                        files=files,
                        timeout=300,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        batch_id = result.get("batch_id", "")

                        # Poll batch status until all processing complete or failed
                        if result.get("succeeded", 0) > 0:
                            with st.spinner("处理中，请等待..."):
                                import time
                                max_wait = 120  # 最大等待120秒
                                interval = 2
                                waited = 0
                                last_status = None

                                while waited < max_wait:
                                    try:
                                        status_resp = requests.get(
                                            f"{API_BASE}/documents/upload/batch/{batch_id}/status",
                                            timeout=10,
                                        )
                                        if status_resp.status_code == 200:
                                            batch_status = status_resp.json()
                                            processing = batch_status.get("processing", 0)
                                            completed = batch_status.get("completed", 0)
                                            failed = batch_status.get("failed", 0)

                                            # Calculate current completion
                                            total_items = completed + failed
                                            current_status = f"完成 {completed}/{total_items}"

                                            # Only update status display if changed
                                            if current_status != last_status:
                                                st.info(f"⏳ 处理中: {current_status}")
                                                last_status = current_status

                                            if processing == 0:
                                                break
                                    except Exception:
                                        pass

                                    time.sleep(interval)
                                    waited += interval

                        st.success(f"✅ 上传完成！批次ID: {batch_id[:8]}...")

                        col_s, col_f, col_d = st.columns(3)
                        col_s.metric("成功", result.get("succeeded", 0))
                        col_f.metric("失败", result.get("failed", 0))
                        col_d.metric("重复", result.get("duplicate", 0))

                        # Show detailed results
                        if result.get("items"):
                            st.markdown("#### 上传详情")
                            for item in result.get("items", []):
                                status_icon = {
                                    "processing": "⏳",
                                    "completed": "✅",
                                    "failed": "❌",
                                    "duplicate": "⚠️",
                                }.get(item.get("status", ""), "❓")

                                if item.get("status") in ["failed", "duplicate"]:
                                    st.markdown(f"{status_icon} {item.get('file_name', 'unknown')}: {item.get('error_message', item.get('status', 'unknown'))}")
                                else:
                                    st.markdown(f"{status_icon} {item.get('file_name', 'unknown')}: {item.get('status', 'unknown')}")

                        st.rerun()
                    else:
                        st.error(f"上传失败: {response.status_code}")

                except Exception as e:
                    st.error(f"上传出错: {str(e)}")
