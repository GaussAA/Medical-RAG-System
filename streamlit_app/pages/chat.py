"""Chat page with streaming support."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import requests
import streamlit as st

from streamlit_app.components.chat_message import render_message

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="医疗问答", page_icon="💬")
st.title("💬 医疗问答")


def answer_stream(question: str, session_id: str | None, result_container: dict):
    """
    Stream answer from the query API.

    Args:
        question: User question
        session_id: Optional session ID
        result_container: Dict to store results

    Yields:
        Content chunks
    """
    payload = {
        "question": question,
        "session_id": session_id,
        "options": {
            "include_citations": True,
            "include_confidence": True,
            "include_warnings": True,
        },
    }

    try:
        response = requests.post(
            f"{API_BASE}/query/stream",
            json=payload,
            stream=True,
            timeout=120,
        )

        if response.status_code != 200:
            yield f"请求失败: {response.status_code}"
            return

        full_answer = ""
        event_type = None

        for line in response.iter_lines():
            if line.startswith(b"event: "):
                event_type = line[6:].decode("utf-8").strip()
            elif line.startswith(b"data: "):
                try:
                    data = json.loads(line[6:].decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                if event_type == "metadata":
                    if data.get("session_id"):
                        st.session_state.session_id = data["session_id"]
                        result_container["session_id"] = data["session_id"]

                elif event_type == "chunk":
                    content = data.get("content", "")
                    full_answer += content
                    yield content

                elif event_type == "done":
                    result_container["confidence"] = data.get("confidence", 0.0)
                    result_container["citations"] = data.get("citations", [])
                    result_container["warnings"] = data.get("warnings", [])
                    result_container["processing_time"] = data.get("processing_time", 0.0)
                    result_container["answer"] = full_answer

                elif event_type == "error":
                    yield f"\n\n处理出错: {data.get('message', '未知错误')}"
                    return

    except Exception as e:
        yield f"\n\n发生错误: {str(e)}"


def render():
    """Render the chat page."""
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Load existing session messages
    session_id = st.session_state.get("session_id")
    if session_id and not st.session_state.messages:
        try:
            response = requests.get(
                f"{API_BASE}/sessions/{session_id}/messages",
                timeout=10,
            )
            if response.status_code == 200:
                messages = response.json()
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                ]
        except Exception:
            pass

    # Display existing messages
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("metadata"))

    # Chat input
    if prompt := st.chat_input("请输入您的问题..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)

        # Stream assistant response
        session_id = st.session_state.get("session_id")
        result_container = {}

        with st.spinner("正在检索知识库..."):
            try:
                # Use streaming
                answer_placeholder = st.empty()
                full_answer = ""

                for chunk in answer_stream(prompt, session_id, result_container):
                    full_answer += chunk
                    answer_placeholder.markdown(f"### 回答\n{full_answer}▌")

                # Finalize answer display
                answer_placeholder.markdown(f"### 回答\n{full_answer}")

                # Add assistant message with metadata
                metadata = {
                    "confidence": result_container.get("confidence", 0.0),
                    "citations": result_container.get("citations", []),
                    "warnings": result_container.get("warnings", []),
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "metadata": metadata,
                })

                # Display metadata
                if result_container.get("citations"):
                    _render_streaming_metadata(result_container)

            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                # Try fallback to sync endpoint
                st.info("尝试同步接口...")
                _fallback_query(prompt)


def _render_streaming_metadata(result_container: dict):
    """Render metadata after streaming completes."""
    from streamlit_app.components.chat import render_confidence_badge, render_warnings

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"耗时: {result_container.get('processing_time', 0):.2f}s")
    with col2:
        citations = result_container.get("citations", [])
        if citations:
            st.info(f"引用来源: {len(citations)}条")

    render_confidence_badge(result_container.get("confidence", 0.0))
    render_warnings(result_container.get("warnings", []))


def _fallback_query(question: str):
    """Fallback to synchronous query endpoint."""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.get("session_id"),
            "options": {
                "include_citations": True,
                "include_confidence": True,
                "include_warnings": True,
            },
        }

        response = requests.post(f"{API_BASE}/query", json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()

            if result.get("session_id"):
                st.session_state.session_id = result["session_id"]

            from streamlit_app.components.chat import render_answer

            render_answer(
                answer=result["answer"],
                confidence=result.get("confidence", 0.0),
                citations=result.get("citations", []),
                warnings=result.get("warnings", []),
                processing_time=result.get("processing_time", 0.0),
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": {
                    "confidence": result.get("confidence", 0.0),
                    "citations": result.get("citations", []),
                    "warnings": result.get("warnings", []),
                },
            })
        else:
            st.error(f"请求失败: {response.status_code}")

    except Exception as e:
        st.error(f"同步接口失败: {str(e)}")


# Sidebar with settings and examples
with st.sidebar:
    st.title("对话设置")

    if st.button("新对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()

    st.markdown("### 使用提示")
    st.info("""
    1. 输入医疗相关问题
    2. 系统从知识库检索
    3. AI生成带引用回答
    4. 请注意风险提示
    """)

    st.divider()

    st.markdown("### 示例问题")
    examples = [
        "糖尿病的诊断标准是什么？",
        "高血压患者饮食建议？",
        "阿司匹林的副作用？",
    ]

    for example in examples:
        if st.button(example, key=f"example_{example}"):
            st.session_state.example_question = example
            st.rerun()


if __name__ == "__main__":
    render()