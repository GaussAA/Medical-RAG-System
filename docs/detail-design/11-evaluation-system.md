# RAG 评估系统详细设计

> 本文档描述 Medical RAG System 的评估系统设计与实现

## 概述

RAG 评估系统用于量化评估 RAG系统的效果，覆盖**检索评估**、**生成评估**、**医疗安全评估**三个维度，支持单次评估和批量基准测试。

## 评估指标体系

### 1. 检索评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| `precision@k` | Top-K 中相关文档的比例 | ` Relevant / K` |
| `recall@k` | 相关文档被召回的比例 | `Relevant / TotalRelevant` |
| `ndcg@k` | 排序质量（相关文档排名越靠前越高） | DCG/IDCG |
| `mrr` | 首个相关文档排名的倒数 | `1 / rank_of_first_relevant` |
| `hit_rate` | 至少召回 1 篇相关文档的查询比例 | 无 ground truth 时可用 |

### 2. 生成评估指标

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| `faithfulness` | 答案陈述与上下文的一致程度 | LLM 法官 / 规则匹配 |
| `answer_relevancy` | 答案与问题的相关程度 | LLM 法官 / 词覆盖 |
| `citation_accuracy` | 引用验证通过比例 | `Citation.verified / total` |
| `hallucination_ratio` | 未验证引用 / 总引用 | `unverified / total` |

### 3. 医疗安全评估

| 指标 | 说明 |
|------|------|
| `entity_accuracy` | 医学实体（药物/疾病/手术）准确率 |
| `warning_coverage` | 安全警告覆盖率（药物/诊断/紧急） |
| `contradiction_detected` | 多上下文矛盾检测 |
| `safety_score` | 综合安全评分 (0-1) |

## 核心组件

### RAGEvaluator

主评估器类，聚合检索、生成、医疗安全三个维度的评估。

```python
from rag.evaluation import RAGEvaluator, RAGEvaluationResult, EvalGroundTruth

evaluator = RAGEvaluator(llm_generator=None)  # llm_generator 可选

result = await evaluator.evaluate(
    query="糖尿病患者如何选择降糖药物？",
    response=query_response,
    ground_truth=None,  # 可选，用于检索指标计算
    retrieved_doc_ids=None,
)
```

### RetrievalEvaluator

检索评估器，计算 Precision@K, Recall@K, NDCG@K, MRR。

```python
from rag.evaluation import RetrievalEvaluator

retrieval_evaluator = RetrievalEvaluator(k_values=[5, 10, 20])

metrics = retrieval_evaluator.evaluate(
    retrieved_ids=["doc1", "doc2", "doc3"],
    ground_truth_ids=["doc1", "doc4"],
    scores=[0.9, 0.8, 0.7],
)
```

### GenerationEvaluator

生成评估器，使用 LLM 法官或规则评估 Faithfulness 和 Answer Relevancy。

```python
from rag.evaluation import GenerationEvaluator

gen_evaluator = GenerationEvaluator(llm_generator=llm_generator)

metrics = await gen_evaluator.evaluate(
    query="...",
    answer="...",
    contexts=["..."],
    citations=citations,
)
```

### MedicalSafetyEvaluator

医疗安全评估器，检查医学实体准确性、警告覆盖、矛盾检测。

```python
from rag.evaluation import MedicalSafetyEvaluator

safety_evaluator = MedicalSafetyEvaluator()

metrics = await safety_evaluator.evaluate(
    query="...",
    answer="...",
    contexts=["..."],
    warnings=risk_warnings,
)
```

### BenchmarkRunner

批量基准测试运行器，支持数据集加载和批量评估。

```python
from rag.evaluation import BenchmarkRunner, BenchmarkConfig

runner = BenchmarkRunner(config=BenchmarkConfig(
    dataset_path="tests/evaluation/fixtures/eval_queries.json",
    output_dir="data/evaluation/reports",
))

result = await runner.run_with_responses(
    evaluator=evaluator,
    queries=query_list,
    responses=response_list,
)
```

### EvaluationReporter

评估报告生成器，支持 JSON 和文本格式输出。

```python
from rag.evaluation import EvaluationReporter

reporter = EvaluationReporter()
summary = reporter.generate_summary(results)
json_report = reporter.generate_json(results)
```

## 评估数据集格式

评估数据集为 JSON 格式，每条记录包含：

```json
{
  "query_id": "med_001",
  "query_text": "糖尿病患者如何选择降糖药物？",
  "query_type": "drug",
  "relevant_doc_ids": ["doc_abc", "doc_xyz"],
  "expected_keywords": ["二甲双胍", "血糖监测", "并发症"],
  "reference_answer": "糖尿病患者首选二甲双胍...",
  "difficulty": "medium",
  "safety_sensitive": true
}
```

## 评估结果结构

`RAGEvaluationResult` 包含完整的评估结果：

```python
@dataclass
class RAGEvaluationResult:
    query_id: str

    # 检索指标
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    retrieval_hit_rate: float

    # 生成指标
    faithfulness: float
    answer_relevancy: float
    citation_accuracy: float
    hallucination_ratio: float

    # 医疗安全
    entity_accuracy: float | None
    warning_coverage: dict[str, bool]
    contradiction_detected: bool
    safety_score: float

    # 综合评分
    overall_score: float
```

## 配置

评估相关配置在 `config/settings.py` 的 `EvaluationConfig` 中：

```python
class EvaluationConfig(BaseModel):
    enable: bool = True
    sample_rate: float = 1.0
    k_values: list[int] = [5, 10, 20]
    output_dir: str = "data/evaluation/reports"
    llm_judge_provider: str = "deepseek"
    llm_judge_model: str = "deepseek-chat"
    faithfulness_threshold: float = 0.8
    relevancy_threshold: float = 0.7
```

## 目录结构

```
rag/evaluation/
├── __init__.py
├── evaluator.py          # RAGEvaluator, RAGEvaluationResult, EvalGroundTruth
├── retrieval_eval.py      # RetrievalEvaluator, RetrievalMetrics
├── generation_eval.py     # GenerationEvaluator, GenerationMetrics
├── medical_safety_eval.py # MedicalSafetyEvaluator, MedicalSafetyMetrics
├── benchmark_runner.py    # BenchmarkRunner, BenchmarkConfig, BenchmarkResult
└── reporter.py            # EvaluationReporter, EvaluationSummary

tests/evaluation/
├── __init__.py
├── test_retrieval_eval.py
├── test_generation_eval.py
├── test_medical_safety.py
└── fixtures/
    └── eval_queries.json  # 评估数据集
```

## 使用示例

### 单次查询评估

```python
import asyncio
import httpx
from rag.evaluation import RAGEvaluator
from app.models.schemas import QueryResponse, Citation, RiskWarning, CitationPosition

async def evaluate_single_query():
    # 1. 获取 RAG 响应
    resp = httpx.post(
        'http://localhost:8000/api/v1/query',
        json={'question': '糖尿病患者如何选择降糖药物？'},
        timeout=120.0
    )
    data = resp.json()

    # 2. 构建 QueryResponse
    response = QueryResponse(
        answer=data['answer'],
        confidence=data['confidence'],
        citations=[Citation(**c) for c in data.get('citations', [])],
        warnings=[RiskWarning(**w) for w in data.get('warnings', [])],
        session_id=data.get('session_id', ''),
        processing_time=data.get('processing_time', 0.0),
        metadata=data.get('metadata', {}),
    )

    # 3. 评估
    evaluator = RAGEvaluator()
    result = await evaluator.evaluate(
        query='糖尿病患者如何选择降糖药物？',
        response=response,
    )

    print(f"Overall Score: {result.overall_score}")
    print(f"Faithfulness: {result.faithfulness}")
    print(f"Safety Score: {result.safety_score}")

asyncio.run(evaluate_single_query())
```

### 批量基准测试

```python
import asyncio
from rag.evaluation import RAGEvaluator, BenchmarkRunner

async def run_benchmark():
    evaluator = RAGEvaluator()
    runner = BenchmarkRunner()

    # 运行评估
    results = await runner.run(
        evaluator=evaluator,
        dataset_path="tests/evaluation/fixtures/eval_queries.json",
    )

    # 保存结果
    output_path = runner.save_results(results)
    print(f"Results saved to {output_path}")

asyncio.run(run_benchmark())
```

## 评估维度计算说明

### Faithfulness 计算

**LLM 法官模式**（当 `llm_generator` 提供时）：
- 使用专用 prompt 让 LLM 判断每个答案陈述是否可从上下文推导
- 计算 SUPPORTED / PARTIAL / UNSUPPORTED 比例

**规则模式**（默认）：
- 检查医学术语在答案和上下文中同时出现
- 检查剂量数值匹配
- 检查引用标记存在性
- 检查答案长度是否足够充分

### Answer Relevancy 计算

**LLM 法官模式**：
- LLM 生成 N 个反向问题
- 计算反向问题与原问题的语义相似度

**规则模式**：
- 计算查询词在答案中的覆盖率
- 检查答案是否以"根据..."开头（负面指标）

### Safety Score 计算

综合以下因素计算：
- 无矛盾检测：+0.3
- 警告覆盖完整：每缺失一项 -0.1
- 医学实体准确率 < 0.5：-0.2
- 医学实体准确率 < 0.8：-0.1

## 注意事项

1. **Ground Truth 可选**：检索指标（Precision/Recall/NDCG/MRR）需要 ground truth，无 ground truth 时仅计算 hit_rate
2. **LLM 法官可选**：不提供 `llm_generator` 时自动使用规则模式评估
3. **异步评估**：所有评估方法均为 async，支持批量并发评估
4. **评估数据集**：用户需根据实际场景准备评估数据集，建议至少 50 条查询