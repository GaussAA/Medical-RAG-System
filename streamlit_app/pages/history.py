import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

st.title("📊 历史记录")

if "del_session_id" not in st.session_state:
    st.session_state.del_session_id = None

# 获取会话列表
try:
    response = requests.get("http://localhost:8000/api/v1/sessions", timeout=5)
    sessions = response.json() if response.status_code == 200 else []
except Exception:
    sessions = []

# 处理删除请求
if st.session_state.del_session_id:
    session_id = st.session_state.del_session_id
    try:
        response = requests.delete(f"http://localhost:8000/api/v1/sessions/{session_id}", timeout=5)
        if response.status_code == 200:
            st.success("会话已删除")
        else:
            st.error(f"删除失败: {response.status_code}")
    except Exception as e:
        st.error(f"删除失败: {e}")
    st.session_state.del_session_id = None
    st.rerun()

if not sessions:
    st.info("暂无历史对话记录。")
else:
    st.markdown(f"共 **{len(sessions)}** 个对话会话")

    for session in sessions:
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            session_id = session.get("session_id", "")
            title = session.get("session_title", "新对话") or "新对话"
            msg_count = session.get("msg_count", 0) or 0

            if st.button(f"**{title}**", key=f"enter_{session_id}"):
                st.session_state.session_id = session_id
                st.session_state.messages = []
                st.session_state.message_metadata = {}
                try:
                    msg_response = requests.get(
                        f"http://localhost:8000/api/v1/sessions/{session_id}/messages",
                        timeout=5,
                    )
                    if msg_response.status_code == 200:
                        messages = msg_response.json()
                        st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in messages]
                        # Store metadata for rendering
                        st.session_state.message_metadata = {
                            m["message_id"]: {
                                "confidence": m.get("confidence"),
                                "citations": m.get("citations") or [],
                                "warnings": m.get("warnings") or [],
                            }
                            for m in messages
                        }
                except Exception:
                    pass
                st.switch_page("pages/query.py")

            st.caption(f"💬 {msg_count} 条消息 · ID: {session_id[:8]}...")

        with col3:
            if st.button("🗑️", key=f"delete_btn_{session_id}"):
                st.session_state.del_session_id = session_id
                st.rerun()

        st.divider()

st.markdown("---")

st.markdown("### 💡 使用提示")
st.info("""
历史对话记录帮助您回顾之前的问答内容。
系统会自动保存最近的对话历史。
点击会话标题可进入问答页面继续对话。
""")
