# Medical RAG System - 评估系统增强设计

> 设计日期：2026-05-20
> 状态：已批准

## 1. 背景与目标

### 1.1 现状问题

当前评估系统存在以下不足：
- **CLI 功能简陋**：仅支持检索评估，无法做端到端的 RAG 评估
- **BenchmarkRunner 是空壳**：run() 方法没有真正调用 RAG 引擎
- **缺少对比分析**：无法横向比较不同配置/版本的效果
- **报告形式单一**：只有文本和 JSON，缺少可视化
- **数据集管理缺失**：无版本管理、无合成数据生成

### 1.2 增强目标

构建完整的评估体系，支持：
- 完整的端到端 RAG 评估（检索 + 生成 + 医疗安全）
- 多模式数据来源（实时调用 / 离线回放 / 混合模式）
- 多维度对比（时间 / 配置 / 版本对比）
- 完整的数据集管理（CRUD / 版本管理 / 合成数据生成）
- 丰富的可视化报告（HTML / PDF / CSV / JSON）

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Evaluation CLI (Typer)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ evaluate    │  │ benchmark   │  │ dataset             │ │
│  │ (单次评估)   │  │ (批量测试)   │  │ (数据集管理)         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                          ┌─────────────────┐ │
│                                          │ compare         │ │
│                                          │ (A/B对比)       │ │
│                                          └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Evaluation Engine (核心引擎)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ interfaces.py (Protocol 定义)                       │   │
│  │  - RetrievalEvaluatorProtocol                       │   │
│  │  - GenerationEvaluatorProtocol                      │   │
│  │  - MedicalSafetyEvaluatorProtocol                   │   │
│  │  - ReporterProtocol                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  RAGEvaluator                       │   │
│  │  (默认紧耦合实现，适配 Protocol 接口)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              BenchmarkRunner                        │   │
│  │  - run()           在线模式                          │   │
│  │  - run_with_responses() 离线模式                     │   │
│  │  - run_hybrid()    混合模式                          │   │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Reporters (报告层)                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐    │
│  │ JSON    │  │ Text    │  │ CSV     │  │ HTML Report  │    │
│  │ Reporter│  │ Reporter│  │ Exporter│  │ (Plotly图表) │    │
│  └─────────┘  └─────────┘  └─────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 核心原则

- **渐进式插件化**：保持现有结构稳定，在关键点引入 Protocol 接口抽象
- **向后兼容**：现有 API 和 RAGEvaluator 使用方式保持不变
- **默认紧耦合**：默认实现使用紧耦合，接口抽象支持未来扩展

## 3. CLI 命令设计

### 3.1 命令结构

```
rag_cli.py evaluate      # 单次评估
rag_cli.py benchmark     # 批量基准测试
rag_cli.py dataset       # 数据集管理
rag_cli.py report        # 报告生成与对比
rag_cli.py compare       # A/B 版本对比
```

### 3.2 evaluate 命令

```bash
# 实时调用模式（默认）
rag_cli.py evaluate \
  --query "糖尿病患者如何选择降糖药物？" \
  --api-url "http://localhost:8000" \
  --output result.json

# 离线回放模式
rag_cli.py evaluate \
  --response-file history/response_20240101.json \
  --output result.json

# 混合模式
rag_cli.py evaluate \
  --query "..." \
  --fallback-file history/responses.jsonl \
  --output result.json
```

### 3.3 benchmark 命令

```bash
rag_cli.py benchmark \
  --dataset data/eval_queries.json \
  --mode online \
  --api-url "http://localhost:8000" \
  --output-dir data/evaluation/results \
  --report-format html \
  --sample 20
```

### 3.4 dataset 命令

```bash
# CRUD 操作
rag_cli.py dataset create --name "medical_qa_v1" --source data/raw_docs/
rag_cli.py dataset list
rag_cli.py dataset validate --dataset data/eval_queries.json
rag_cli.py dataset export --dataset data/eval_queries.json --format jsonl

# 合成数据生成
rag_cli.py dataset generate \
  --source data/guidelines/*.md \
  --count 100 \
  --output data/synthetic_qa.json

# 版本管理
rag_cli.py dataset version --dataset data/eval_queries.json --create --tag "v1.0"
rag_cli.py dataset version --dataset data/eval_queries.json --list
rag_cli.py dataset version --dataset data/eval_queries.json --rollback v0.9
```

### 3.5 compare 命令

```bash
# 时间对比
rag_cli.py compare timeline \
  --query "糖尿病用药" \
  --baseline results/baseline.json \
  --current results/current.json

# 配置对比
rag_cli.py compare config \
  --query "糖尿病用药" \
  --config-a config/rrf_weight_0.6.yaml \
  --config-b config/rrf_weight_0.8.yaml

# 版本对比
rag_cli.py compare version \
  --query "糖尿病用药" \
  --version-a "v1.0" \
  --version-b "v1.1"
```

### 3.6 混合模式流程

```
用户发起请求
     │
     ▼
┌─────────────────────────┐
│ 尝试实时调用 RAG API     │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌────────┐    ┌─────────────────┐
│ 调用成功│    │ 调用失败        │
└───┬────┘    └────────┬────────┘
    │                   │
    ▼                   ▼
┌────────────┐   ┌─────────────────┐
│ 执行评估   │   │ 检查 fallback    │
└─────┬──────┘   └────────┬────────┘
      │                   │
      │           ┌───────┴───────┐
      │           ▼               ▼
      │     ┌───────────┐   ┌──────────┐
      │     │ fallback  │   │ 无 fallback│
      │     │ 存在      │   │ 返回错误   │
      │     └─────┬─────┘   └──────────┘
      │           │
      │           ▼
      │     ┌────────────┐
      │     │ 加载离线响应│
      │     │ 执行评估    │
      │     └─────┬──────┘
      │           │
      ▼           ▼
┌─────────────────────┐
│ 输出评估报告        │
└─────────────────────┘
```

## 4. 报告与可视化设计

### 4.1 支持格式

| 格式 | 用途              | 生成速度 | 交互性 |
| ---- | ----------------- | -------- | ------ |
| JSON | 程序消费/数据导出 | 快       | 无     |
| CSV  | Excel 分析/统计   | 快       | 无     |
| Text | 控制台快速查看    | 最快     | 无     |
| HTML | 报告分享/交互查看 | 中       | 有     |

### 4.2 HTML 报告模块

- **雷达图**：展示 Retrieval/Generation/Safety 三维评分
- **直方图**：分数分布，识别异常值
- **对比表**：查询级别的详细结果对比
- **告警高亮**：显著下降的指标自动高亮

### 4.3 报告生成器接口

```python
class ReporterPlugin(Protocol):
    """报告生成器接口"""
    def generate(self, results: list[RAGEvaluationResult]) -> str: ...
    def supports_format(self, fmt: str) -> bool: ...

class MultiReporter:
    """组合报告生成器"""
    def generate_all(self, results, output_dir: Path) -> dict[str, Path]: ...
```

## 5. 数据集管理设计

### 5.1 存储结构

```
data/
└── datasets/
    ├── manifest.json              # 数据集索引清单
    ├── medical_qa_v1/
    │   ├── metadata.json
    │   ├── v1.0/data.jsonl
    │   └── v1.1/data.jsonl
    └── synthetic_qa_2026/
        ├── metadata.json
        └── data.jsonl
```

### 5.2 DatasetManager 功能

| 功能             | 说明           |
| ---------------- | -------------- |
| create_dataset   | 创建新数据集   |
| list_datasets    | 列出所有数据集 |
| get_dataset      | 获取数据集内容 |
| delete_dataset   | 删除数据集     |
| validate_dataset | 验证数据集格式 |
| create_version   | 创建新版本     |
| rollback         | 回滚到指定版本 |
| import/export    | JSONL 导入导出 |

### 5.3 合成数据生成

```python
class SyntheticDataGenerator:
    """使用 LLM 生成合成评估数据"""
    async def generate(
        self,
        source_docs: list[Document],
        count: int = 50,
        query_types: list[str] | None = None,
    ) -> list[EvalGroundTruth]: ...
```

## 6. 实施计划

### 6.1 阶段划分

| 阶段 | 内容                                                    | 预计时间 |
| ---- | ------------------------------------------------------- | -------- |
| 1    | 核心引擎增强（interfaces.py、BenchmarkRunner 混合模式） | 2-3 天   |
| 2    | CLI 完善（evaluate/benchmark/dataset/compare 命令）     | 1-2 天   |
| 3    | 报告与可视化（HTMLReporter、CSVExporter）               | 1-2 天   |
| 4    | 数据集管理（DatasetManager、SyntheticDataGenerator）    | 1 天     |

**总计**：约 5-8 个工作日

### 6.2 文件变更

**新增文件**：
```
rag/evaluation/interfaces.py           # 评估器接口定义
rag/evaluation/dataset_manager.py      # 数据集管理器
rag/evaluation/synthetic_generator.py  # 合成数据生成器
rag/evaluation/reporters/             # 报告生成器目录
│   ├── __init__.py
│   ├── json_reporter.py
│   ├── csv_reporter.py
│   └── html_reporter.py
rag_cli.py                            # 统一 CLI 入口
tests/evaluation/test_dataset_manager.py
```

**修改文件**：
```
rag/evaluation/benchmark_runner.py     # 增强混合模式
rag/evaluation/evaluator.py            # 适配接口抽象
rag/evaluation/cli.py                  # 重构为子命令
```

## 7. 风险与缓解

| 风险                  | 影响                          | 缓解措施                             |
| --------------------- | ----------------------------- | ------------------------------------ |
| 合成数据质量不可控    | 生成数据无法真实反映 RAG 效果 | 添加人工验证环节，支持半自动化生成   |
| HTML 报告图表渲染问题 | 不同浏览器显示不一致          | 使用 Plotly 生成自包含 HTML          |
| 批量测试耗时长        | 评估 100 条 query 可能超时    | 支持并发评估，提供 `--parallel` 参数 |

## 8.向后兼容性

- 现有 API 调用方式保持不变
- 现有 `RAGEvaluator` 使用方式保持兼容
- 新增接口作为可选扩展，不破坏现有功能
