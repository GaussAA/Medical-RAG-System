import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

st.title("💬 智能问答")

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "message_metadata" not in st.session_state:
    st.session_state.message_metadata = {}


# 页面加载时，如果存在 session_id，则从数据库加载历史消息
session_id = st.session_state.get("session_id")
if session_id:
    try:
        response = requests.get(
            f"http://localhost:8000/api/v1/sessions/{session_id}/messages",
            timeout=10
        )
        if response.status_code == 200:
            messages = response.json()
            if messages and not st.session_state.messages:
                st.session_state.messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in messages
                ]
                # Store metadata for rendering history附加信息
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


def answer_stream(question, session_id, result_container):
    """同步迭代器，yield 内容片段，结果存入 result_container."""
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
            "http://localhost:8000/api/v1/query/stream",
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


def render_streaming_answer(full_answer):
    """Render partial streaming answer."""
    st.markdown("### 回答")
    st.markdown(full_answer + "▌")


# 显示已加载的消息
from streamlit_app.components.chat import render_confidence_badge, render_warnings, render_citations

for idx, message in enumerate(st.session_state.get("messages", [])):
    st.chat_message(message["role"]).write(message["content"])
    # 如果是 assistant 消息，渲染附加信息
    if message["role"] == "assistant" and st.session_state.get("message_metadata"):
        # 找到对应的 metadata
        metadata_keys = list(st.session_state.message_metadata.keys())
        if idx < len(metadata_keys):
            msg_id = metadata_keys[idx]
            meta = st.session_state.message_metadata.get(msg_id, {})
            confidence = meta.get("confidence")
            citations = meta.get("citations", [])
            warnings = meta.get("warnings", [])
            if confidence is not None:
                render_confidence_badge(confidence)
            if warnings:
                render_warnings(warnings)
            if citations:
                render_citations(citations)

question = st.chat_input("输入您的医疗问题...")

if question:
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    session_id_to_use = st.session_state.get("session_id")

    # 使用 st.write_stream 流式显示回答
    with st.spinner("正在检索知识库..."):
        try:
            result_container = {}
            st.write_stream(answer_stream(question, session_id_to_use, result_container))

            # 流结束后检查 result_container
            if result_container.get("answer"):
                full_answer = result_container.get("answer", "")
                confidence = result_container.get("confidence", 0.0)
                citations = result_container.get("citations", [])
                warnings = result_container.get("warnings", [])
                processing_time = result_container.get("processing_time", 0.0)

                # 流结束后只渲染元数据（置信度、警告、引用），answer 已由 st.write_stream 显示
                from streamlit_app.components.chat import render_confidence_badge, render_warnings, render_citations

                st.markdown("### 回答")
                render_confidence_badge(confidence)
                render_warnings(warnings)

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"⏱️ 耗时: {processing_time:.2f}s")
                with col2:
                    if citations:
                        st.info(f"📚 引用来源: {len(citations)}条")

                if citations:
                    render_citations(citations)

                st.session_state.messages.append({"role": "assistant", "content": full_answer})
            else:
                raise Exception("流式接口未返回完整数据")

        except (StopIteration, TypeError) as e:
            # 流式接口失败或未返回 final_data，回退到同步接口
            fallback_placeholder = st.empty()
            fallback_placeholder.info("流式输出未完成，回退到同步接口...")

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

                sync_response = requests.post(
                    "http://localhost:8000/api/v1/query",
                    json=payload,
                    timeout=120,
                )

                fallback_placeholder.empty()

                if sync_response.status_code == 200:
                    result = sync_response.json()

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
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                else:
                    st.error(f"请求失败: {sync_response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("无法连接到后端服务，请确保 FastAPI 服务正在运行。")
            except Exception as sync_error:
                st.error(f"发生错误: {str(sync_error)}")

        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            st.session_state.messages.pop()  # 移除刚添加的用户消息


with st.sidebar:
    st.title("对话设置")

    if st.button("🆕 新建对话"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

    st.divider()

    st.markdown("### 使用提示")
    st.info("""
    1. 输入医疗相关问题
    2. 系统会从知识库中检索相关信息
    3. AI生成带引用的回答
    4. 请注意查看置信度和风险提示
    """)

    st.divider()

    st.markdown("### 示例问题")
    examples = [
        "糖尿病的诊断标准是什么？",
        "高血压患者应该如何饮食？",
        "阿司匹林有什么副作用？",
    ]

    for example in examples:
        if st.button(example, key=f"example_{example}"):
            st.session_state.example_question = example
            st.rerun()