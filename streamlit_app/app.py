import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import requests

from config.settings import get_settings

settings = get_settings()

st.set_page_config(
    page_title=settings.streamlit.page_title,
    page_icon=settings.streamlit.page_icon,
    initial_sidebar_state=settings.streamlit.initial_sidebar_state,
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

# ========== Sidebar Navigation ==========
with st.sidebar:
    st.title("导航")

    st.markdown("---")

    # Main navigation
    pages = {
        "💬 医疗问答": "pages/query.py",
        "📚 文档管理": "pages/documents.py",
        "📊 历史记录": "pages/history.py",
        "📈 评估中心": "pages/evaluation.py",
        "💬 聊天（新页面）": "pages/chat.py",
    }

    for label, page_path in pages.items():
        if st.button(label, use_container_width=True):
            st.switch_page(page_path)

    st.markdown("---")

    # System status
    st.markdown("### 系统状态")

    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            st.success("🟢 后端服务正常")
        else:
            st.warning("🟡 后端服务异常")
    except Exception:
        st.error("🔴 后端服务离线")

    st.markdown("---")
    st.markdown(f"版本: {settings.app.version}")

# ========== Main Content ==========
st.title(f"{settings.streamlit.page_icon} {settings.streamlit.page_title}")

# Welcome section
st.markdown("""
欢迎使用 **医疗文档检索增强生成系统**。

本系统提供以下功能：
- 💬 **智能问答** - 基于医学知识库的自动问答
- 📚 **文档管理** - 上传和管理医学指南文档
- 📊 **历史记录** - 查看和继续历史对话
- 📈 **评估中心** - 评估系统性能和答案质量
""")

# Quick stats
col1, col2, col3 = st.columns(3)

try:
    doc_response = requests.get("http://localhost:8000/api/v1/documents?page_size=1", timeout=5)
    if doc_response.status_code == 200:
        total_docs = doc_response.json().get("total", 0)
        col1.metric("已上传文档", total_docs)
except Exception:
    col1.metric("已上传文档", "N/A")

try:
    session_response = requests.get("http://localhost:8000/api/v1/sessions", timeout=5)
    if session_response.status_code == 200:
        total_sessions = len(session_response.json())
        col2.metric("对话会话", total_sessions)
except Exception:
    col2.metric("对话会话", "N/A")

col3.metric("系统状态", "运行中")

# Quick actions
st.markdown("### 快速开始")

col1, col2 = st.columns(2)

with col1:
    if st.button("开始新对话", type="primary", use_container_width=True):
        st.switch_page("pages/query.py")

with col2:
    if st.button("上传文档", use_container_width=True):
        st.switch_page("pages/documents.py")

# Usage tips
with st.expander("使用提示"):
    st.info("""
    1. **上传文档**：先在文档管理页面上传医学指南文档
    2. **开始问答**：在问答页面输入医疗相关问题
    3. **查看来源**：AI回答会附带引用来源供核实
    4. **注意警告**：关注系统给出的医疗安全警告
    """)
