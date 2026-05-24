# 医疗通用知识库 RAG 问答系统——技术设计详细说明

## 1. 数据流总览

整个系统的数据处理与问答流程如下，分**文档入库**与**在线问答**两条主线。

### 1.1 文档入库流程

```
用户上传文档 → 安全检测（文件类型/敏感内容）→ 格式解析（PyMuPDF/python-docx/markdown）
→ 提取文本和表格 → 表格转结构化文本 → 语义分片（基于 token 重叠）
→ 生成 chunk 元数据（来源、页码、分段号） → 向量化（BAAI/bge-m3）
→ 存入 Qdrant（向量 + payload） 与 PostgreSQL（chunk 原文 + 元数据）
→ 更新索引状态
```

### 1.2 在线问答流程

```
用户提问 → 安全检测（敏感词/PHI过滤或脱敏） → 多轮对话上下文改写（可选）
→ 混合检索（向量检索 + BM25 全文检索）→ 粗排候选集 top_k
→ 重排序（bge-reranker-v2-m3）→ 取 top_n 精排片段
→ 置信度预评估（基于检索分/片段相似度） → 拼接上下文提示
→ 大语言模型生成回答（DeepSeek API） → 答案置信度综合评估
→ 判断是否降级处理 → 组装引用来源、风险提示 → 返回结果并记录日志
```

## 2. API 端点详细设计

### 2.1 文档管理

#### `POST /api/v1/documents/upload`

- **描述**：上传医疗文档并触发自动入库流程。
- **请求**：`multipart/form-data`  
  - `file`: 文件（支持 pdf, docx, md, txt）
  - `meta`: JSON 字符串，可附加自定义元数据（如科室、来源类型）
- **响应**：
  ```json
  {
    "document_id": "doc_123",
    "filename": "高血压诊疗指南.pdf",
    "status": "processing",
    "chunks_count": null
  }
  ```
- **后端处理**：立即返回，后台异步执行解析、分片、向量化流程；通过 `GET /api/v1/documents/{doc_id}/status` 查询进度。

#### `GET /api/v1/documents/{doc_id}/status`

- **描述**：获取文档处理状态。
- **响应**：
  ```json
  {
    "document_id": "doc_123",
    "status": "ready",  // pending, processing, ready, error
    "chunks_count": 42,
    "error_message": null
  }
  ```

### 2.2 问答服务

#### `POST /api/v1/chat`

- **描述**：发送问题进行问答，支持多轮对话。
- **请求**：`application/json`
  ```json
  {
    "session_id": "sess_abc (可选，新会话则创建)",
    "question": "高血压患者用药应注意什么？",
    "history": [   // 可选，如果是多轮，可由前端维护或后端通过session_id获取
      {"role": "user", "content": "什么是高血压？"},
      {"role": "assistant", "content": "高血压是指..."}
    ],
    "options": {
      "retrieval_top_k": 10,
      "reranker_top_n": 5,
      "confidence_threshold": 0.7,
      "enable_risk_prompt": true
    }
  }
  ```
- **响应**：
  ```json
  {
    "answer": "高血压患者用药需注意...",
    "confidence": 0.87,
    "confidence_level": "high",  // high / medium / low
    "sources": [
      {
        "document_id": "doc_123",
        "filename": "高血压诊疗指南.pdf",
        "chunk_id": "chunk_456",
        "text": "用药原则包括...",
        "page": 15,
        "relevance_score": 0.92
      }
    ],
    "risk_warnings": [
      "本回答仅供参考，不构成医疗建议，请务必咨询专业医生。",
      "用药剂量须遵循医嘱，擅自调整可能引起危险。"
    ],
    "is_fallback": false,
    "session_id": "sess_abc"
  }
  ```

#### `GET /api/v1/chat/history/{session_id}`

- **描述**：获取会话的历史消息。
- **响应**：
  ```json
  {
    "session_id": "sess_abc",
    "messages": [
      {"role": "user", "content": "什么是高血压？"},
      {"role": "assistant", "content": "..."},
      ...
    ]
  }
  ```

### 2.3 系统健康检查

#### `GET /api/v1/health`

- 返回各服务连接状态（PostgreSQL, Qdrant, Redis, DeepSeek API）。

## 3. 数据模型设计

### 3.1 PostgreSQL 表结构

**文档表 (documents)**
```sql
CREATE TABLE documents (
    id VARCHAR(64) PRIMARY KEY,
    filename VARCHAR(512) NOT NULL,
    file_type VARCHAR(16),
    upload_time TIMESTAMP DEFAULT NOW(),
    status VARCHAR(16) DEFAULT 'pending',  -- pending, processing, ready, error
    error_message TEXT,
    meta JSONB DEFAULT '{}',
    chunks_count INT DEFAULT 0
);
```

**分片表 (chunks)**
```sql
CREATE TABLE chunks (
    id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,     -- 分片在文档中的序号
    text TEXT NOT NULL,
    page INT,                     -- PDF页码或word段落号
    metadata JSONB DEFAULT '{}',  -- 如表格标识、章节标题
    token_count INT
);
CREATE INDEX idx_chunks_doc_id ON chunks(document_id);
```

**对话日志表 (conversation_log)**（可选，用于分析）
```sql
CREATE TABLE conversation_log (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    role VARCHAR(16),
    content TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_log_session ON conversation_log(session_id, created_at);
```

### 3.2 Qdrant 集合结构

- **集合名称**：`medical_knowledge`
- **向量维度**：1024 (bge-m3 输出维度)
- **距离度量**：Cosine
- **Payload 示例**：
  ```json
  {
    "chunk_id": "chunk_456",
    "document_id": "doc_123",
    "text": "高血压用药原则包括...",
    "page": 15,
    "tags": ["高血压", "药物治疗"]
  }
  ```

### 3.3 Redis 缓存/状态设计

- **会话上下文**：
  - 键：`session:{session_id}:messages`
  - 值：JSON 数组，存放最近 N 轮对话，参与上下文改写或直接携带历史逻辑。
  - 过期：30 分钟无操作自动清除。

- **检索缓存**（可选）：
  - 键：`ret:{sha256(question+top_k)}`
  - 值：检索结果 JSON，过期 1 小时，减少重复检索调用。

- **速率限制**：基于 IP/会话，防止滥用。

## 4. 安全检测明确方案

### 4.1 输入安全

- **敏感词检测**：维护一份医疗场景下的正则/关键词列表，包括身份证号、手机号、姓名+病症组合、药品名称+剂量等模式。检测到后**脱敏**（替换为 `[已脱敏]`）而非直接拒绝，以免干扰真实医疗咨询。
- **提示注入防御**：过滤“忽略之前指令”等越狱语句，使用简单的模式匹配库（如 `re` 自定义规则）。
- **安全层实现**：以 FastAPI 中间件形式，对所有 `/chat` 请求的 `question` 字段应用。

### 4.2 输出安全

- 所有生成答案强制追加 **固定风险提示模板**（见下方场景示例）。
- 动态风险提示：在 prompt 中要求 LLM 判断是否涉及药物剂量、手术建议等，并在输出中标记，后端解析后附加特定风险语句。

## 5. 混合检索与重排序具体设计

### 5.1 混合检索策略

- **向量检索**：Qdrant 近邻搜索，返回余弦相似度最高的 `top_k=20` 个候选。
- **全文检索**：基于 PostgreSQL 的 `tsvector` 中文分词全文索引，对 `chunks.text` 进行检索，返回 `top_k=20`。如果没有 PG 分词，可借助 LlamaIndex 的 BM25 检索器（使用本地内存索引，简单不依赖 ES）。
- **融合算法**：**倒数排名融合 (RRF)**  
  `score(chunk) = Σ (1/(k + rank_i))`，对每个检索列表，`rank` 为 chunk 在该列表中的排名，`k=60`。取融合后得分最高的 `top_k=10` 作为粗排结果。

### 5.2 重排序

- 使用 `BAAI/bge-reranker-v2-m3` 模型，输入为 `(question, chunk_text)` 对，计算相关性分数。
- 对粗排的 10 个片段重打分，取 `top_n=5` 送入 LLM 上下文。
- 重排序服务本地部署（GPU 推荐），通过一个小型 FastAPI 接口或直接集成在检索服务中。

## 6. 置信度评估与降级策略

### 6.1 置信度计算

由于 DeepSeek API 不公开内部 logprobs，采用**多维度启发式评估**：
- **检索分加权**：取重排序后的最高分 (0~1) 和平均分，权重 0.4。
- **答案一致性**：生成答案后，将问题与答案拼接，计算与参考片段的相似度（可以用 bge-m3 向量相似度），权重 0.3。
- **LLM 自我评估**：在生成 prompt 中要求 LLM 在回答结尾输出 `[CONFIDENCE: xx%]` 并解析，权重 0.3。
- 最终置信度 = 0.4*ret_score + 0.3*vec_sim + 0.3*llm_conf，归一化至 [0,1]。

### 6.2 降级处理

- 若置信度 < 0.5，触发降级：
  - `is_fallback` 设为 `true`
  - 回答改为：“抱歉，我无法提供足够可靠的答案。以下是最相关的参考资料，请谨慎参考：” + 分段列表。
  - 保留风险提示和来源。
- 若 0.5 ≤ 置信度 < 0.7，`confidence_level=medium`，答案正常但强调风险。
- 若 ≥ 0.7，`high`，正常返回。

## 7. 大语言模型调用与提示词模板

### 7.1 DeepSeek API 调用参数

- 模型：`deepseek-chat`（或其他可用聊天模型）
- 温度：0.1（保证严谨性）
- max_tokens：1024
- 系统提示词：
```
你是专业医疗知识助手，请基于以下参考资料回答用户问题。回答需准确、有条理，并引用提供的来源。如果在参考资料中找不到相关信息，请明确说明。最后，请评估答案置信度并以 [CONFIDENCE: x%] 结尾。
参考资料：
{context}
```
- 用户消息构成：
  - 如有历史消息，拼接为对话格式（LlamaIndex 的 CondenseQuestion 模式或手动压缩）。
  - 加上当前问题。

## 8. 多轮对话实现方案

- **上下文维护**：Redis 存储每会话消息列表（最多保留最近 10 轮）。
- **上下文压缩**：在检索前，若存在历史，利用 LLM 将最近三轮对话总结为一个独立的查询（condensed question），以该查询进行检索，保证相关度。
- 生成阶段，将完整历史与当前压缩问题及检索上下文传入，使回答保持连续。

## 9. 前端界面概要（Streamlit）

- **主页面**：侧边栏可选文档上传（拖拽或文件选择），状态刷新；主区域为聊天窗口，类似对话气泡。
- **答案卡片**：
  - 置信度以进度条或颜色标记（绿/黄/红）。
  - 来源以可折叠列表展示，点击可查看片段原文。
  - 风险提示以醒目的黄色警告框显示。
- **新会话按钮**：重置 session_id。

## 10. 配置参数表

| 配置项                   | 默认值                    | 说明                               |
| ------------------------ | ------------------------- | ---------------------------------- |
| `CHUNK_SIZE`             | 512                       | 分片 token 数（用 tokenizer 近似） |
| `CHUNK_OVERLAP`          | 50                        | 相邻分片重叠 token 数              |
| `VECTOR_TOP_K`           | 20                        | 向量检索候选数                     |
| `BM25_TOP_K`             | 20                        | 全文检索候选数                     |
| `RRF_K`                  | 60                        | RRF 融合参数                       |
| `RERANKER_TOP_N`         | 5                         | 送入 LLM 片段数                    |
| `CONFIDENCE_THRESHOLD`   | 0.5                       | 降级触发阈值                       |
| `SESSION_EXPIRE_SECONDS` | 1800                      | Redis 会话过期时间                 |
| `DEEPSEEK_MODEL`         | `deepseek-chat`           | LLM 模型名                         |
| `EMBEDDING_MODEL`        | `BAAI/bge-m3`             | 嵌入模型（本地或远程加载）         |
| `RERANKER_MODEL`         | `BAAI/bge-reranker-v2-m3` | 重排序模型                         |

## 11. 关键场景示例

### 11.1 正常问答（Happy Path）

**用户**：高血压患者如何选择降压药？  
**系统内部**：  
1. 安全检测无敏感信息。  
2. 无历史，直接检索。  
3. 混合检索+重排序得到5个高度相关片段（涉及一线药物、联合用药）。  
4. 置信度计算 0.88。  
5. 生成回答并引用3个来源，附风险提示。  

**回答展示**：
> 根据指南，通常首选一线药物包括 ACEI、ARB、CCB 等，具体选择需考虑患者合并症。  
> 📊 置信度：高  
> 📄 来源：1.《中国高血压防治指南》 P23；2.《临床药物治疗学》 P102 
> ⚠️ 风险提示：本信息仅供参考，不构成医嘱，具体用药请咨询医生。

### 11.2 敏感信息过滤（Edge Case）

**用户**：张医生你好，我身份证号是 11010119900307663X，最近血压高。  
**系统内部**：  
- 安全中间件检测到身份证号模式，自动替换为 `[已脱敏]`。  
- 后续流程相同，问题变成“张医生你好，我身份证号是 [已脱敏]，最近血压高。”  

### 11.3 低置信度降级

**用户**：请问最新的人工肺移植后存活率
**系统内部**：  
- 知识库内找不到相关最新文献，检索片段不相关。  
- 重排序最高分仅 0.4，综合置信度 0.35。  
- 触发降级，回答变为：  
> 抱歉，我无法提供足够可靠的回答。以下是最相关的片段，供参考：  
> 1. ……  
> 2. ……  
> ⚠️ 风险提示：以上片段可能不能直接回答您的问题，请查阅最新文献。

### 11.4 多轮对话

**用户第一问**：什么是2型糖尿病？  
**回答**：……（正常返回）  
**用户第二问**：它的饮食控制原则是什么？  
**系统**：  
- Redis 保存历史。  
- 将最近3轮压缩为上下文“用户正在询问2型糖尿病的饮食控制原则”。  
- 检索到相关片段，生成连贯回答。

## 12. 工程部署与监控

- **容器化**：提供 Docker Compose 编排，包含 FastAPI 应用、PostgreSQL、Qdrant、Redis、Streamlit 前端。
- **日志**：结构化日志（JSON 格式），记录每次问答的 session、问题、置信度、检索耗时，便于监控。
- **降级开关**：当重排序或 LLM 服务不可用时，可配置降低到仅返回检索片段。