# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

Medical Knowledge Base RAG Q&A System - 医疗文档检索增强生成系统，支持混合检索（BM25 + 向量相似度）、置信度评估和多轮对话。

**核心能力**: 用户上传医学指南文档 → 系统分块建索引 → 用户提问 → RAG 检索 → LLM 生成答案，附带安全检查和引用追溯。

**详细架构**: [docs/detail-design/01-architecture-overview.md](docs/detail-design/01-architecture-overview.md)

## 核心设计决策

理解这些决策对于正确地在此项目中工作至关重要：

### 1. Markdown-only 处理
- PDF/DOCX 支持已移除（解析可靠性问题）
- 所有文档格式：`.md`、`.markdown`
- [03-document-processing.md](docs/detail-design/03-document-processing.md) § Markdown-Only Processing

### 2. 层级感知分块 (Hierarchical Chunking)
- 按 H1-H6 标题边界切分文档
- 表格、列表作为独立语义单元保留
- 每个 Chunk 携带 `heading_tree`（完整标题路径）和 `content_type`（text/table/list）
- [03-document-processing.md](docs/detail-design/03-document-processing.md) § HierarchicalChunker

### 3. 查询类型 Boosting
- 检测查询中的内容类型意图，表查询优先返回表格，药物查询优先返回列表
- [04-retrieval-system.md](docs/detail-design/04-retrieval-system.md) § Query-Type Detection

### 4. 三存储同步策略
- 删除顺序：PostgreSQL（先）→ Qdrant → BM25（后）
- PostgreSQL 失败则中止，索引失败则记录不一致并用 `/cleanup-orphans` 修复
- [01-architecture-overview.md](docs/detail-design/01-architecture-overview.md) § Synchronization Rules

### 5. 向量/BM25 权重
- RRF 公式：`vector_weight=0.6`, `bm25_weight=0.4`, `rrf_k=60`
- [04-retrieval-system.md](docs/detail-design/04-retrieval-system.md) § Reciprocal Rank Fusion

### 6. 多轮对话上下文注入
- `QueryRequest` 可带 `session_id`，首次查询无 `session_id` 时自动创建
- 历史通过 `conversation_history` 参数注入 LLM prompt
- 消息持久化位置：`RAGEngine.query()` 内部
- [05-session-management.md](docs/detail-design/05-session-management.md) § Context Building

### 7. 批量上传统一向量化

- 批量上传接口 `POST /api/v1/documents/upload/batch` 支持多文件同时上传
- **统一向量化**：所有文档解析分块后，一次性调用 `rag_engine.process_document()`，embedding 模型只加载一次
- 单批次最大 50 个文件
- MD5 去重检测，重复文件返回 `duplicate` 状态
- [03-document-processing.md](docs/detail-design/03-document-processing.md) § Batch Upload

### 8. RAG 评估系统
- 评估维度：检索（Precision@K/Recall@K/NDCG@K/MRR）、生成（Faithfulness/Answer Relevancy）、医疗安全（实体准确率/警告覆盖率）
- 评估器入口： 类，支持单次评估和批量基准测试
- 评估数据集格式：
- [11-evaluation-system.md](docs/detail-design/11-evaluation-system.md) § RAG Evaluation System

## 文档索引

> 状态标记：`[完整]` 可信赖 | `[部分]` 包含关键信息 | `[待完善]` 占位符

| 文档                                                                          | 内容                                    | 状态   |
| ----------------------------------------------------------------------------- | --------------------------------------- | ------ |
| [01-architecture-overview.md](docs/detail-design/01-architecture-overview.md) | 系统组件关系、数据流、高层架构          | [完整] |
| [02-rag-pipeline.md](docs/detail-design/02-rag-pipeline.md)                   | 完整查询流程                            | [完整] |
| [03-document-processing.md](docs/detail-design/03-document-processing.md)     | 上传→解析→分块→建索引流程               | [完整] |
| [04-retrieval-system.md](docs/detail-design/04-retrieval-system.md)           | HybridRetriever、RRF、内容类型 Boosting | [完整] |
| [05-session-management.md](docs/detail-design/05-session-management.md)       | SessionManager、消息驱逐、上下文窗口    | [完整] |
| [06-data-models.md](docs/detail-design/06-data-models.md)                     | PostgreSQL 表结构、三存储映射           | [完整] |
| [07-configuration.md](docs/detail-design/07-configuration.md)                 | YAML 配置和 Pydantic settings           | [完整] |
| [08-gpu-memory-management.md](docs/detail-design/08-gpu-memory-management.md) | Embedding/Reranker GPU 懒加载策略       | [完整] |
| [09-query-type-detection.md](docs/detail-design/09-query-type-detection.md)   | 查询类型检测、内容类型增强机制          | [完整] |
| [10-citation-verification.md](docs/detail-design/10-citation-verification.md) | 引用验证、幻觉检测机制                  | [完整] |
| [11-evaluation-system.md](docs/detail-design/11-evaluation-system.md)         | RAG 评估系统、基准测试、指标体系        | [完整] |

## 常用命令

```bash
# 环境初始化
uv sync                                          # 安装依赖
uv run python scripts/init_db.py                  # 初始化 PostgreSQL
uv run python scripts/init_vector_db.py            # 初始化 Qdrant

# 启动服务
uv run uvicorn app.main:app --reload             # 后端 (port 8000)
uv run streamlit run streamlit_app/app.py         # 前端 (port 8501)

# 测试
uv run pytest tests/unit/                         # 单元测试
uv run pytest tests/integration/                   # 集成测试（需 PostgreSQL + Qdrant）
```

## 关键文件参考

| 文件                                                                   | 职责                             |
| ---------------------------------------------------------------------- | -------------------------------- |
| [app/main.py](app/main.py)                                             | FastAPI 应用工厂、CORS、路由注册 |
| [app/core/rag_engine.py](app/core/rag_engine.py)                       | RAG 查询编排入口                 |
| [app/services/session.py](app/services/session.py)                     | Session 状态、消息持久化、驱逐   |
| [app/api/routes/query.py](app/api/routes/query.py)                     | 查询 API 入口                    |
| [app/api/routes/documents.py](app/api/routes/documents.py)             | 文档上传/批量上传 API            |
| [rag/generation/llm_generator.py](rag/generation/llm_generator.py)     | LLM 调用、prompt 构建            |
| [rag/retrieval/hybrid_retriever.py](rag/retrieval/hybrid_retriever.py) | 混合检索 + RRF                   |
| [rag/evaluation/evaluator.py](rag/evaluation/evaluator.py)             | RAG 评估器入口                   |
| [streamlit_app/app.py](streamlit_app/app.py)                           | Streamlit 前端入口               |

## 代码组织

```
app/
├── api/routes/          # API 路由层
├── core/               # 核心业务（RAGEngine）
├── models/             # Pydantic schemas + SQLAlchemy models
├── services/           # 服务层（Document, Session, Consistency）
└── core/database.py    # 数据库连接管理

rag/
├── parser/             # 文档解析
├── chunking/           # 分块策略
├── retrieval/          # 检索器（Vector、BM25、Hybrid）
├── reranker/          # 交叉编码重排序
├── generation/        # LLM 生成
└── evaluation/        # RAG 评估（检索/生成/医疗安全）

streamlit_app/          # Streamlit 前端
```

## 数据库模型关系

```
Document ───1:N──→ Heading
Document ───1:N──→ Chunk
Heading ───1:N──→ Chunk
Conversation ───1:N──→ Message
```

详细 ER 图：[06-data-models.md](docs/detail-design/06-data-models.md)

## 补充说明

### msg_count 字段
`ConversationSession` 有一个 `msg_count` 字段，用于跟踪会话消息数量。这与 `len(messages)` 不同，因为消息可能在达到限制时被驱逐。

### 引用验证与幻觉检测
`RAGEngine._generate_warnings()` 直接生成风险警告，包含幻觉检测：
- 检测未验证引用（`verified=False`）的比例
- 超过阈值（默认 0.5）时触发 `hallucination` 警告
- 配置项：`generation.citation_verification.hallucination_threshold`

### 警告生成方式
`app/core/risk_warnings.py` 中的 `RiskWarningGenerator` 类**未被 RAGEngine 使用**。RAGEngine 直接在 `_generate_warnings()` 方法中生成警告，包含：
- `general` - 始终添加
- `medication` - 检测到药物关键词
- `diagnosis` - 检测到诊断关键词
- `emergency` - 检测到紧急症状关键词
- `hallucination` - 检测到未验证引用

## 注意事项

- 该文档专门给 AI 阅读，并起着目录索引功能。需要详细了解时，应引导去 `docs/detail-design/` 相关文档
- 所有 `datetime.utcnow()` 已标记为废弃，应使用 `datetime.now(UTC)`
- async session 必须在 `finally` 块中显式关闭以避免连接池 GC 警告
- 删除文档时必须遵循 PostgreSQL → Qdrant → BM25 的顺序
- 多轮对话消息持久化在 `RAGEngine.query()` 内部，不在 API 路由层

## 反模式警示

| 禁止                                              | 说明                            |
| ------------------------------------------------- | ------------------------------- |
| ❌ 在 `RAGEngine.query()` 外部调用 `add_message()` | 消息持久化应在 RAGEngine 内部   |
| ❌ 修改 `db_confirmed` 标志                        | 这是内部实现细节                |
| ❌ 打乱三存储删除顺序                              | 必须 PostgreSQL → Qdrant → BM25 |
| ❌ 使用 `datetime.utcnow()`                        | 已废弃，用 `datetime.now(UTC)`  |
| ❌ 在 async session 关闭后继续使用                 | 会导致连接池警告                |

## 故障排除

| 错误                               | 解决方案                                    |
| ---------------------------------- | ------------------------------------------- |
| `页面文件太小 (os error 1455)`     | 内存不足，增加可用内存或减少后台应用        |
| `Connection refused` (PostgreSQL)  | 检查服务 `pg_isready`                       |
| `Connection refused` (Qdrant)      | 检查服务 `curl http://localhost:6333`       |
| `ConversationSession has no field` | Pydantic 字段需用 `Field(default=...)` 定义 |

内存需求：Embedding ~1.5GB + Reranker ~1.8GB，建议可用内存 > 4GB。