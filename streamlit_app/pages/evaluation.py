"""Evaluation page for RAG system assessment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"


@st.cache_data(ttl=60)
def _load_document_titles() -> list[str]:
    """Load document titles from API for the document selector."""
    try:
        resp = requests.get(f"{API_BASE}/documents?page_size=100", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get("documents", [])
            return [doc["title"] for doc in docs]
    except Exception:
        pass
    return []

st.set_page_config(page_title="评估中心", page_icon="📊")
st.title("📊 RAG 评估中心")


def render_evaluation_metrics(result: dict):
    """Render evaluation metrics from a single result."""
    col1, col2, col3 = st.columns(3)

    overall = result.get("overall_score", 0.0)
    col1.metric("综合评分", f"{overall:.2%}")

    retrieval = result.get("retrieval", {})
    mrr = retrieval.get("mrr", 0.0)
    col2.metric("检索 MRR", f"{mrr:.2%}")

    generation = result.get("generation", {})
    faithfulness = generation.get("faithfulness", 0.0)
    col3.metric("答案忠实度", f"{faithfulness:.2%}")


def render_retrieval_metrics(retrieval: dict):
    """Render retrieval metrics."""
    st.markdown("#### 检索指标")

    col1, col2, col3 = st.columns(3)
    col1.metric("命中率", f"{retrieval.get('hit_rate', 0):.2%}")
    col2.metric("MRR@10", f"{retrieval.get('mrr', 0):.2%}")

    ndcg_at_k = retrieval.get("ndcg_at_k", {})
    if 10 in ndcg_at_k:
        col3.metric("NDCG@10", f"{ndcg_at_k[10]:.2%}")


def render_generation_metrics(generation: dict):
    """Render generation metrics."""
    st.markdown("#### 生成指标")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("忠实度", f"{generation.get('faithfulness', 0):.2%}")
    col2.metric("答案相关度", f"{generation.get('answer_relevancy', 0):.2%}")
    col3.metric("引用准确率", f"{generation.get('citation_accuracy', 0):.2%}")
    col4.metric("幻觉率", f"{generation.get('hallucination_ratio', 0):.2%}")


def render_medical_safety_metrics(safety: dict):
    """Render medical safety metrics."""
    st.markdown("#### 医疗安全指标")

    col1, col2, col3 = st.columns(3)
    col1.metric("安全评分", f"{safety.get('safety_score', 0):.2%}")

    entity_acc = safety.get("entity_accuracy")
    if entity_acc is not None:
        col2.metric("实体准确率", f"{entity_acc:.2%}")
    else:
        col2.metric("实体准确率", "N/A")

    contradiction = safety.get("contradiction_detected", False)
    status = "是" if contradiction else "否"
    col3.metric("检测到矛盾", status)

    warning_coverage = safety.get("warning_coverage", {})
    if warning_coverage:
        with st.expander("警告覆盖率"):
            for wtype, covered in warning_coverage.items():
                icon = "✅" if covered else "❌"
                st.markdown(f"{icon} {wtype}")


def render_single_evaluation(query_id: str, result: dict):
    """Render a single evaluation result."""
    with st.container():
        st.markdown(f"### 查询: {query_id}")

        render_evaluation_metrics(result)

        tabs = st.tabs(["检索", "生成", "医疗安全", "详情"])

        with tabs[0]:
            retrieval = result.get("retrieval", {})
            render_retrieval_metrics(retrieval)

        with tabs[1]:
            generation = result.get("generation", {})
            render_generation_metrics(generation)

        with tabs[2]:
            safety = result.get("medical_safety", {})
            render_medical_safety_metrics(safety)

        with tabs[3]:
            st.json(result.get("details", {}))

        st.divider()


def render_benchmark_results(results: list[dict]):
    """Render benchmark test results summary."""
    if not results:
        st.info("暂无评估结果")
        return

    # Summary stats
    total = len(results)
    avg_overall = sum(r.get("overall_score", 0) for r in results) / total
    avg_retrieval = sum(r.get("retrieval", {}).get("mrr", 0) for r in results) / total
    avg_faithfulness = sum(r.get("generation", {}).get("faithfulness", 0) for r in results) / total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("评估数量", total)
    col2.metric("平均综合分", f"{avg_overall:.2%}")
    col3.metric("平均 MRR", f"{avg_retrieval:.2%}")
    col4.metric("平均忠实度", f"{avg_faithfulness:.2%}")

    st.markdown("---")

    # Individual results
    for result in results:
        query_id = result.get("query_id", "unknown")
        render_single_evaluation(query_id, result)


def run_single_evaluation():
    """Form to run a single evaluation."""
    st.markdown("### 单次评估")

    with st.form("single_eval_form"):
        query = st.text_area("输入查询", height=100)
        reference_answer = st.text_area("参考答案（可选）", height=100)

        # Load existing documents for the document selector
        doc_titles = _load_document_titles()
        selected_docs = st.multiselect(
            "期望文档（选择后自动匹配其所有分块作为 ground truth）",
            options=doc_titles,
            placeholder="搜索或选择文档...",
        )

        submitted = st.form_submit_button("运行评估")
        if submitted and query:
            with st.spinner("评估中..."):
                try:
                    payload = {
                        "query": query,
                        "expected_answer": reference_answer if reference_answer else None,
                        "relevant_doc_ids": selected_docs,
                    }

                    response = requests.post(
                        f"{API_BASE}/evaluation/evaluate",
                        json=payload,
                        timeout=120,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("评估完成")
                        render_single_evaluation(query, result)
                    else:
                        st.error(f"评估失败: {response.status_code}")

                except Exception as e:
                    st.error(f"评估出错: {str(e)}")


def run_benchmark():
    """Form to run benchmark tests."""
    import json  # Local import for json parsing

    st.markdown("### 基准测试")

    # JSONL format for dataset
    st.caption("数据集格式：JSONL，每行包含 query, expected_answer, relevant_doc_ids 字段")

    with st.form("benchmark_form"):
        dataset_path = st.text_area(
            "数据集路径",
            placeholder="""/path/to/evaluation_dataset.jsonl
格式示例（每行JSON）：
{"query": "糖尿病诊断标准", "expected_answer": "...", "relevant_doc_ids": ["doc1"]}
{"query": "高血压用药", "expected_answer": "...", "relevant_doc_ids": ["doc2"]}""",
            height=120,
        )
        limit = st.number_input("限制数量", min_value=1, max_value=1000, value=50)

        submitted = st.form_submit_button("运行基准测试")
        if submitted and dataset_path:
            with st.spinner("基准测试运行中..."):
                try:
                    # Parse JSONL data from the path/content
                    if dataset_path.strip().startswith("{"):
                        # It's JSONL content, not a path
                        lines = dataset_path.strip().split("\n")
                        dataset = [json.loads(line) for line in lines if line.strip()]
                    else:
                        # It's a file path, read the file
                        # Security: validate path is within allowed data directory
                        from pathlib import Path
                        safe_base = Path(__file__).parent.parent.parent / "data"
                        requested_path = Path(dataset_path.strip()).resolve()
                        try:
                            requested_path.relative_to(safe_base.resolve())
                        except ValueError:
                            st.error("路径不允许：只能在 data/ 目录下读取文件")
                            return
                        with open(requested_path) as f:
                            dataset = [json.loads(line) for line in f if line.strip()]

                    dataset = dataset[:limit]  # Apply limit

                    response = requests.post(
                        f"{API_BASE}/evaluation/benchmark",
                        json={"dataset": dataset},
                        timeout=600,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        st.success(f"基准测试完成，共 {len(results)} 条评估")
                        render_benchmark_results(results)
                    else:
                        st.error(f"基准测试失败: {response.status_code}")

                except json.JSONDecodeError as e:
                    st.error(f"数据集格式错误: {str(e)}")
                except FileNotFoundError:
                    st.error(f"文件不存在: {dataset_path}")
                except Exception as e:
                    st.error(f"基准测试出错: {str(e)}")


def view_history():
    """View evaluation history."""
    st.markdown("### 评估历史")

    try:
        response = requests.get(f"{API_BASE}/evaluation/history", timeout=10)
        if response.status_code == 200:
            data = response.json()
            history = data.get("history", [])
            total = data.get("total", 0)

            if not history:
                st.info("暂无评估历史")
                return

            # Top toolbar: total count + export button
            col_total, col_export = st.columns([3, 1])
            with col_total:
                st.caption(f"共 {total} 条评估记录")
            with col_export:
                import json as _json
                json_bytes = _json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    label="📥 导出 JSON",
                    data=json_bytes,
                    file_name="evaluation_history.json",
                    mime="application/json",
                    use_container_width=True,
                )

            st.divider()

            for item in history:
                timestamp = item.get("timestamp", "")
                query_id = item.get("query_id", "unknown")
                overall = item.get("overall_score", 0)

                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{query_id}**")
                        st.caption(f"时间: {timestamp[:19]}")
                    with col2:
                        st.metric("评分", f"{overall:.2%}")

                    if st.button("查看详情", key=f"view_{query_id}"):
                        render_single_evaluation(query_id, item)

                    st.divider()

        else:
            st.error(f"获取历史失败: {response.status_code}")

    except Exception as e:
        st.error(f"获取历史出错: {str(e)}")


# Main page layout
tab1, tab2, tab3 = st.tabs(["单次评估", "基准测试", "历史记录"])

with tab1:
    run_single_evaluation()

with tab2:
    run_benchmark()

with tab3:
    view_history()