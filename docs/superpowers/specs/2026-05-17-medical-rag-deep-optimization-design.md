# Medical RAG System 深度优化与重构设计

**日期**: 2026-05-17
**状态**: 已批准
**版本**: v1.0

---

## 1. 目标与成功标准

### 目标

对 Medical RAG System 进行深度优化与全面重构，覆盖技术债清理、功能释放、性能优化三个维度，达到生产就绪标准。

### 成功标准

| 维度       | 标准                                  |
| ---------- | ------------------------------------- |
| 异步一致性 | 全链路 async/await，无阻塞调用        |
| 测试覆盖   | 全流程集成测试 + 核心组件单元测试     |
| 代码质量   | 无死代码，代码风格统一，模块职责单一  |
| 模块边界   | 清晰可独立测试，接口协议明确          |
| 安全       | CORS限制、敏感信息脱敏、输入校验完备  |
| 错误处理   | 显式异常处理，无空捕获块              |
| 可观测     | 结构化日志，完整Tracing               |
| 配置规范   | YAML配置 + Pydantic校验，环境变量优先 |

---

## 2. 技术债清理（Phase 1）

### 2.1 死代码清理

| 文件/组件                              | 问题                                        | 动作                                                |
| -------------------------------------- | ------------------------------------------- | --------------------------------------------------- |
| `app/core/risk_warnings.py`            | 定义但从未被RAGEngine使用                   | 评估：合并到RAGEngine `_generate_warnings()` 或移除 |
| `CitationVerifier.verify()`            | 空壳逻辑，无实际验证                        | 重构为真实验证逻辑或移除                            |
| `config/settings.py` Observer          | 初始加载不触发，仅 `reload_settings()` 生效 | 简化或移除观察者模式                                |
| `app/services/session.py` 中未使用导入 | 检查并清理                                  | 移除                                                |

### 2.2 异步模式统一

**问题**: `VectorRetriever.add()` 是 `async` 方法但内部调用同步的 `self.embedding_model.encode()`

**修复方案**:
```python
# 方案1: run_in_executor
def encode_sync(texts: list[str]) -> list[np.ndarray]:
    return self.embedding_model.encode(texts, ...)

async def add(self, chunks: list[Chunk]) -> None:
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, self.encode_sync, texts)
    # ... 其余逻辑
```

**检查范围**:
- [ ] `VectorRetriever.add()` 阻塞调用
- [ ] `BM25Retriever` 所有同步I/O
- [ ] `DocumentService.process_document()` 调用链
- [ ] `RAGEngine.process_document()` 到 `add()` 的完整链路

### 2.3 安全修复

**CORS 限制**
```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # 非 ["*"]
    allow_credentials=True,
)
```

**敏感信息脱敏**
- 日志中 Mask API Key、Token、密码
- 用户输入在异常信息中不回显

**输入校验**
- 所有 API 请求 Pydantic schema 严格校验
- 文件上传大小限制、类型校验

### 2.4 测试覆盖补全

**全流程集成测试** (`tests/integration/test_full_pipeline.py`)
```
上传文档 → 解析分块 → 建索引 → 查询 → 生成答案 → 引用验证
```

**单元测试**
- `tests/unit/test_bm25_retriever.py`
- `tests/unit/test_vector_retriever.py`
- `tests/unit/test_hybrid_retriever.py` — RRF融合逻辑
- `tests/unit/test_risk_warnings.py`

### 2.5 模块边界重划

**当前问题**: `rag/generation/llm_generator.py` 职责过重（LLM调用 + Citation提取 + Prompt构建）

**重构后**:
```
rag/generation/
├── llm_generator.py      # 仅LLM调用
├── prompt_builder.py     # Prompt模板构建
├── citation_extractor.py # 引用提取（从LLMGenerator分离）
└── warnings_generator.py  # 警告生成（从RAGEngine._generate_warnings分离）
```

**检索层**:
```
rag/retrieval/
├── vector_retriever.py   # 仅向量检索
├── bm25_retriever.py     # 仅BM25
├── hybrid_retriever.py   # 仅RRF编排，不含检索实现
├── reranker.py           # 重排序（从HybridRetriever分离）
└── query_boosting.py     # 查询类型Boosting逻辑（从HybridRetriever分离）
```

---

## 3. 功能释放（Phase 2）

### 3.1 Redis 缓存层

**缓存策略**:

| 缓存对象      | Key格式                           | TTL   | 淘汰策略 |
| ------------- | --------------------------------- | ----- | -------- |
| 查询结果      | `query:{query_hash}:{session_id}` | 5min  | LRU      |
| Session元数据 | `session:{session_id}`            | 24h   | LRU      |
| LLM生成缓存   | `llm:{prompt_hash}`               | 10min | LRU      |
| 文档Chunk     | `chunk:{chunk_id}`                | 1h    | LRU      |

**实现位置**: `app/services/cache.py`（新建）

**接口**:
```python
class CacheService:
    async def get_query_result(key: str) -> Optional[QueryResult]
    async def set_query_result(key: str, result: QueryResult, ttl: int = 300)
    async def get_session(session_id: str) -> Optional[SessionData]
    async def set_session(session_id: str, data: SessionData, ttl: int = 86400)
```

### 3.2 评估器 CLI

**入口**: `rag/evaluation/cli.py`

**命令**:
```bash
# 单次评估
uv run python -m rag.evaluation.cli evaluate --query "高血压用药" --expected-answer "..."

# 批量基准测试
uv run python -m rag.evaluation.cli benchmark --dataset tests/fixtures/eval_dataset.jsonl

# 输出格式
# --format json | table | markdown
```

**报告结构**:
```json
{
  "retrieval": {"precision_at_5": 0.8, "recall_at_5": 0.9, "ndcg_at_5": 0.85},
  "generation": {"faithfulness": 0.92, "answer_relevancy": 0.88},
  "safety": {"entity_accuracy": 0.95, "warning_coverage": 0.78}
}
```

### 3.3 Streamlit 完整 UI

**页面结构**:
```
streamlit_app/
├── app.py              # 主入口（重构当前骨架）
├── pages/
│   ├── chat.py        # 聊天界面
│   ├── documents.py   # 文档管理
│   └── evaluation.py  # 评估结果
├── components/
│   ├── chat_message.py
│   ├── source_display.py
│   └── document_card.py
└── services/
    └── api_client.py  # 统一API调用
```

**聊天界面功能**:
- 消息历史展示（来源折叠/展开）
- 实时流式响应（可选）
- 安全警告可视化
- 引用溯源点击跳转

---

## 4. 性能优化（Phase 3）

### 4.1 GPU 内存管理完善

**当前问题**: `torch.cuda.mem_get_info()` fallback 逻辑可能在显存压力下产生不一致结果

**改进**:
```python
# 统一显存查询接口
def get_gpu_memory_status() -> GPUMemoryStatus:
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return GPUMemoryStatus(allocated, reserved, total)
    except Exception:
        return GPUMemoryStatus.unknown()
```

**加载策略**:
- Embedding 模型：查询时加载，空闲30min后卸载
- Reranker 模型：查询时加载，首轮结果返回后卸载

### 4.2 并发优化

**批量上传并发向量化**:
```python
# 改进前: 串行
for doc in documents:
    chunks = parse(doc)
    await rag_engine.process_document(chunks)

# 改进后: 并发（embedding模型只加载一次）
all_chunks = []
for doc in documents:
    all_chunks.extend(parse(doc))
await rag_engine.process_document(all_chunks)  # 内部批量向量化
```

**查询并行化**:
```python
# Vector检索和BM25并行执行
vector_results, bm25_results = await asyncio.gather(
    self.vector_retriever.retrieve(query, top_k),
    self.bm25_retriever.retrieve(query, top_k)
)
```

### 4.3 检索性能调优

**Qdrant 预-filter**:
- 利用 `pre_filter` 尽早过滤不适合的文档
- 减少候选集大小

**BM25 按需加载**:
- 避免启动时全量加载
- LRU 缓存最近使用的索引分片

---

## 5. 实现顺序

```
Phase 1: 技术债清理
  ├── 1.1 死代码清理 + 异步统一
  ├── 1.2 安全修复（CORS、凭证、日志脱敏）
  ├── 1.3 测试覆盖补全
  └── 1.4 模块边界重划

Phase 2: 功能释放
  ├── 2.1 Redis 缓存层
  ├── 2.2 评估器 CLI
  └── 2.3 Streamlit 完整 UI

Phase 3: 性能优化
  ├── 3.1 GPU 内存管理完善
  ├── 3.2 并发优化
  └── 3.3 检索性能调优

Phase 4: 架构储备
  └── 仅在 Phase 1-3 后仍必要时触发
```

---

## 6. 关键技术决策

| 决策点           | 选择                                  | 原因            |
| ---------------- | ------------------------------------- | --------------- |
| Redis序列化格式  | JSON（简单场景）/ MessagePack（高频） | 优先级A先用JSON |
| 缓存失效策略     | 写入时失效（write-invalidate）        | 简化一致性逻辑  |
| 模块边界调整方式 | 逐步重构，不做Big Bang                | 风险可控        |
| 测试框架         | pytest + pytest-asyncio               | 已有基础        |
| 流式响应         | 可选功能，B阶段不强制                 | 复杂度考虑      |

---

## 7. 风险与缓解

| 风险                 | 概率 | 影响 | 缓解措施                    |
| -------------------- | ---- | ---- | --------------------------- |
| 异步改造破坏现有逻辑 | 中   | 高   | 充分单元测试 + 集成测试覆盖 |
| Redis引入单点故障    | 低   | 中   | 提供fallback（直接查询）    |
| Streamlit UI重构返工 | 中   | 低   | 先确认UI需求再动手          |
| 性能优化效果不及预期 | 中   | 低   | Phase 3后进行基准测试验证   |

---

## 8. 待明确事项

- [ ] Redis连接信息（host/port/password）是否已配置在环境变量中？是
- [ ] 批量上传max 50文件的限制是否需要调整为可配置？是
- [ ] Streamlit UI是否有品牌/样式要求？要求现代，美观
- [ ] 评估基准数据集是否有现成的？无现成的