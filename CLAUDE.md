# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Medical Knowledge Base RAG Q&A System — 医疗文档检索增强生成系统，支持混合检索（BM25 + 向量相似度）、置信度评估和多轮对话。

**核心能力**: 用户上传医学指南文档 → 系统分块建索引 → 用户提问 → RAG 检索 → LLM 生成答案，附带安全检查和引用追溯。

**详细架构**: [docs/detail-design/01-architecture-overview.md](docs/detail-design/01-architecture-overview.md)

## 架构信息

> 2026-07-01 重构完成：扁平结构 → **模块化单体 + 垂直切片**架构。
> 源码根目录为 `src/`，所有新代码在此目录下。
> 旧目录（app/、rag/、config/、streamlit_app/）已删除。

| 切片     | 路径                | 职责                                               |
| -------- | ------------------- | -------------------------------------------------- |
| 横切层   | `src/common/`       | config/database/cache/logging/monitoring/safety/DI |
| 文档管理 | `src/documents/`    | 上传/解析/分块/索引                                |
| RAG查询  | `src/query/`        | 检索/重排/生成/引用验证                            |
| 会话管理 | `src/conversation/` | 多轮对话上下文                                     |
| 评估系统 | `src/evaluation/`   | RAG评估/基准测试/报告                              |

**入口**: `src/main.py`（FastAPI应用工厂）
**DI容器**: `src/common/di/container.py`
**配置**: `src/common/config/config.yaml`
**模块公共API**: 每个切片的 `__init__.py` 导出 Protocol 接口，跨模块调用必须通过接口

## 核心设计决策

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
- 所有文档解析分块后，一次性调用 `rag_engine.process_document()`，embedding 模型只加载一次
- 单批次最大 50 个文件，MD5 去重检测
- [03-document-processing.md](docs/detail-design/03-document-processing.md) § Batch Upload

### 8. FP16 量化与 GPU 常驻策略

- Embedding 和 Reranker 均以 FP16 精度加载，两模型合计显存 ~1.7GB
- 模型在启动时懒加载到 CUDA 后即常驻，不再需要 `GPUMemoryManager`
- [相关配置: config.yaml](src/common/config/config.yaml)

### 9. 依赖倒置 + 模块接口

- 每个切片的 `__init__.py` 定义该模块对外暴露的 Protocol 接口
- 跨模块调用必须通过接口，禁止绕过接口直接访问模块内部
- DI Container (`src/common/di/container.py`) 负责在启动时组装所有服务
- 详见 `src/*/__init__.py`

### 10. RAG 评估系统

- 评估维度：检索（Precision@K/Recall@K/NDCG@K/MRR）、生成（Faithfulness/Answer Relevancy）、医疗安全
- 数据集位置：`data/evaluation_dataset.jsonl`
- [11-evaluation-system.md](docs/detail-design/11-evaluation-system.md)

## 文档索引

| 文档                                                                          | 内容                                    | 状态   |
| ----------------------------------------------------------------------------- | --------------------------------------- | ------ |
| [01-architecture-overview.md](docs/detail-design/01-architecture-overview.md) | 系统组件关系、数据流、高层架构          | [完整] |
| [02-rag-pipeline.md](docs/detail-design/02-rag-pipeline.md)                   | 完整查询流程                            | [完整] |
| [03-document-processing.md](docs/detail-design/03-document-processing.md)     | 上传→解析→分块→建索引流程               | [完整] |
| [04-retrieval-system.md](docs/detail-design/04-retrieval-system.md)           | HybridRetriever、RRF、内容类型 Boosting | [完整] |
| [05-session-management.md](docs/detail-design/05-session-management.md)       | SessionManager、消息驱逐、上下文窗口    | [完整] |
| [06-data-models.md](docs/detail-design/06-data-models.md)                     | PostgreSQL 表结构、三存储映射           | [完整] |
| [07-configuration.md](docs/detail-design/07-configuration.md)                 | YAML 配置和 Pydantic settings           | [完整] |
| [08-gpu-memory-management.md](docs/detail-design/08-gpu-memory-management.md) | [废弃] FP16 常驻 GPU 后不再需要         | [废弃] |
| [09-query-type-detection.md](docs/detail-design/09-query-type-detection.md)   | 查询类型检测、内容类型增强机制          | [完整] |
| [10-citation-verification.md](docs/detail-design/10-citation-verification.md) | 引用验证、幻觉检测机制                  | [完整] |
| [11-evaluation-system.md](docs/detail-design/11-evaluation-system.md)         | RAG 评估系统、基准测试、指标体系        | [完整] |
| [tech-stack-analysis.md](docs/tech-stack-analysis.md)                         | 技术栈分析、利用率评分、优化建议        | [完整] |

## 常用命令

```bash
# 环境初始化
uv sync                                          # 安装依赖
uv run python scripts/db/init_db.py               # 初始化 PostgreSQL
uv run python scripts/db/init_qdrant.py            # 初始化 Qdrant

# 启动服务
uv run uvicorn src.main:app --reload             # 后端 (port 8000)
uv run streamlit run frontend/app.py             # 前端 (port 8501)

# 测试
uv run pytest tests/unit/                         # 单元测试
uv run pytest tests/integration/                   # 集成测试（需 PostgreSQL + Qdrant）

# 代码检查
uv run ruff check src/                            # lint
uv run mypy src/                                  # 类型检查
```

## 关键文件参考

| 文件                                                                   | 职责                                      |
| ---------------------------------------------------------------------- | ----------------------------------------- |
| [src/main.py](src/main.py)                                             | FastAPI 应用工厂、CORS、路由注册、DI 容器 |
| [src/common/di/container.py](src/common/di/container.py)               | 依赖注入容器（服务组装）                  |
| [src/common/config/settings.py](src/common/config/settings.py)         | Pydantic Settings 配置加载                |
| [src/query/engine.py](src/query/engine.py)                             | RAG 查询编排入口                          |
| [src/query/retrieval/hybrid.py](src/query/retrieval/hybrid.py)         | 混合检索 + RRF                            |
| [src/query/generation/generator.py](src/query/generation/generator.py) | LLM 调用、prompt 构建                     |
| [src/query/generation/prompt.py](src/query/generation/prompt.py)       | Prompt 模板和常量                         |
| [src/query/confidence.py](src/query/confidence.py)                     | 置信度评估                                |
| [src/query/citation/verifier.py](src/query/citation/verifier.py)       | 引用验证、幻觉检测                        |
| [src/documents/service.py](src/documents/service.py)                   | 文档生命周期管理                          |
| [src/documents/processor.py](src/documents/processor.py)               | 文档解析编排                              |
| [src/documents/chunker.py](src/documents/chunker.py)                   | 层级感知分块                              |
| [src/documents/store.py](src/documents/store.py)                       | 文档数据访问层                            |
| [src/documents/indexer.py](src/documents/indexer.py)                   | 向量/BM25 索引管理                        |
| [src/conversation/manager.py](src/conversation/manager.py)             | 会话状态、消息持久化、驱逐                |
| [src/conversation/consistency.py](src/conversation/consistency.py)     | 三存储一致性检查                          |
| [src/evaluation/evaluator.py](src/evaluation/evaluator.py)             | RAG 评估器入口                            |
| [src/common/models.py](src/common/models.py)                           | 公共 Pydantic schemas                     |
| [src/common/safety/checker.py](src/common/safety/checker.py)           | 安全检查器                                |
| [src/common/cache/manager.py](src/common/cache/manager.py)             | Redis 缓存管理（连接池）                  |
| [frontend/app.py](frontend/app.py)                                     | Streamlit 前端入口                        |
| [frontend/api_client.py](frontend/api_client.py)                       | 统一 API 客户端                           |

## 代码组织

```
src/                            # 源码根目录（模块化单体 + 垂直切片）
├── common/                    # 横切关注点
│   ├── config/                # 配置管理（Pydantic Settings + YAML）
│   ├── database/              # 数据库连接 + ORM 模型
│   ├── cache/                 # Redis 缓存（ConnectionPool）
│   ├── logging/               # 日志持久化（文件轮转 + trace_id 注入）
│   ├── monitoring/            # Prometheus 指标
│   ├── safety/                # 安全检查器
│   ├── di/                    # DI容器 + FastAPI依赖
│   ├── models.py              # 公共 Pydantic schemas
│   └── api.py                 # 健康检查
├── documents/                 # 文档管理（垂直切片）
│   ├── __init__.py            # → DocumentStorePort, DocumentServicePort
│   ├── api.py                 # 文档 CRUD API 路由
│   ├── service.py             # 文档生命周期编排
│   ├── processor.py           # 解析编排
│   ├── store.py               # 数据访问层
│   ├── indexer.py             # 索引管理
│   ├── parser/                # 文档解析器
│   └── chunker.py             # 层级分块
├── query/                     # RAG 查询（垂直切片）
│   ├── __init__.py            # → RAGEnginePort, LLMGeneratorPort, HybridRetrieverPort
│   ├── api.py                 # 查询 API 路由（含 SSE 流式）
│   ├── engine.py              # RAG 查询编排
│   ├── confidence.py          # 置信度评估
│   ├── retrieval/             # 检索器（vector / bm25 / hybrid）
│   ├── reranker/              # 交叉编码重排序
│   ├── generation/            # LLM生成 + prompt 模板
│   └── citation/              # 引用验证
├── conversation/              # 会话管理（垂直切片）
│   ├── __init__.py            # → SessionManagerPort, ConsistencyCheckerPort
│   ├── api.py                 # 会话 CRUD API
│   ├── manager.py             # 会话状态 + Redis 缓存
│   └── consistency.py         # 跨存储一致性检查
├── evaluation/                # 评估系统（垂直切片）
│   └── evaluator + reporters + api
└── main.py                    # FastAPI 入口

frontend/                      # Streamlit 前端
├── api_client.py              # 统一 API 客户端
├── app.py                     # 首页
└── pages/                     # 多页面
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

`ConversationSession` 的 `msg_count` 跟踪会话消息数量，与 `len(messages)` 不同（消息可能在达到限制时被驱逐）。

### 引用验证与幻觉检测

`RAGEngine._generate_warnings()` 直接生成风险警告：

- 检测未验证引用（`verified=False`）的比例
- 超过阈值（默认 0.5）时触发 `hallucination` 警告

### 警告类型

| 类型            | 触发条件               |
| --------------- | ---------------------- |
| `general`       | 始终添加               |
| `medication`    | 检测到药物关键词       |
| `diagnosis`     | 检测到诊断关键词       |
| `emergency`     | 检测到紧急症状关键词   |
| `hallucination` | 未验证引用比例超过阈值 |

## 注意事项

- 该文档专门给 AI 阅读，并起目录索引功能。需要详细了解时，引导去 `docs/detail-design/` 相关文档
- 所有 `datetime.utcnow()` 已废弃，使用 `datetime.now(UTC)`
- async session 必须在 `finally` 块中关闭以避免连接池 GC 警告
- 删除文档时必须遵循 PostgreSQL → Qdrant → BM25 的顺序
- 跨模块调用必须通过 `__init__.py` 中的 Port 接口，禁止直接 import 模块内部
- DI Container 在 `src/common/di/container.py`，修改依赖时同步更新

## 反模式警示

| 禁止                                               | 说明                               |
| -------------------------------------------------- | ---------------------------------- |
| ❌ 在 `RAGEngine.query()` 外部调用 `add_message()` | 消息持久化应在 RAGEngine 内部      |
| ❌ 修改 `db_confirmed` 标志                        | 内部实现细节                       |
| ❌ 打乱三存储删除顺序                              | 必须 PostgreSQL → Qdrant → BM25    |
| ❌ 使用 `datetime.utcnow()`                        | 已废弃，用 `datetime.now(UTC)`     |
| ❌ 绕过接口访问模块内部                            | 必须通过 `__init__.py` 导出的 Port |
| ❌ 在 async session 关闭后继续使用                 | 会导致连接池警告                   |

## 故障排除

| 错误                               | 解决方案                                    |
| ---------------------------------- | ------------------------------------------- |
| `页面文件太小 (os error 1455)`     | 内存不足，增加可用内存或减少后台应用        |
| `Connection refused` (PostgreSQL)  | 检查服务 `pg_isready`                       |
| `Connection refused` (Qdrant)      | 检查服务 `curl http://localhost:6333`       |
| `ConversationSession has no field` | Pydantic 字段需用 `Field(default=...)` 定义 |

内存需求：Embedding ~1.5GB + Reranker ~1.8GB，建议可用内存 > 4GB。
