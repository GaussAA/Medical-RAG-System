# 医疗知识库RAG问答系统 - 需求规格说明书

## 1. 项目概述

### 1.1 项目背景

计算机科学与技术专业大学生毕业设计

### 1.2 项目目标

开发一个医疗通用知识库RAG问答Web系统，能够：
- 处理用户输入的医疗相关问题
- 从医疗文档中检索相关信息生成准确回答
- 结合文本检索和向量检索的混合检索方案
- 提供用户友好的Web界面

### 1.3 术语定义

| 术语       | 定义                                         |
| ---------- | -------------------------------------------- |
| RAG        | Retrieval-Augmented Generation，检索增强生成 |
| Chunk      | 文档分片后的文本片段                         |
| Rerank     | 对检索结果进行相关性重排序                   |
| Confidence | 答案置信度评分                               |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户界面层 (Streamlit)                    │
├─────────────────────────────────────────────────────────────────┤
│                         API网关层 (FastAPI)                       │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   安全检测    │   查询处理   │   文档管理    │     会话管理        │
│   模块       │   模块       │   模块        │     模块           │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                         RAG核心引擎 (LlamaIndex)                 │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   文档解析   │   语义分片    │   混合检索    │     答案生成        │
│   处理器     │   处理器      │   引擎        │     引擎           │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│                    数据存储层                                     │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  PostgreSQL  │   Qdrant     │    Redis     │    文件存储         │
│  (结构化数据) │  (向量数据)   │   (缓存)     │   (原始文档)        │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

### 2.2 模块职责说明

| 模块名称       | 职责                      | 关键接口                                     |
| -------------- | ------------------------- | -------------------------------------------- |
| 安全检测模块   | 检测敏感信息、风险内容    | `safety_check(text) -> SafetyResult`         |
| 文档解析模块   | 解析PDF/Word/Markdown文档 | `parse_document(file_path) -> Document`      |
| 语义分片模块   | 文档智能分片              | `chunk_document(doc, config) -> List[Chunk]` |
| 混合检索模块   | 文本+向量融合检索         | `hybrid_search(query, top_k) -> List[Node]`  |
| 重排序模块     | BM25+Reranker二次排序     | `rerank(query, nodes) -> List[Node]`         |
| 答案生成模块   | LLM生成带引用的回答       | `generate_answer(query, contexts) -> Answer` |
| 置信度评估模块 | 评估答案可靠性            | `evaluate_confidence(answer) -> float`       |
| 会话管理模块   | 多轮对话上下文维护        | `add_message(session_id, msg)`               |

### 2.3 数据流程

```
用户问题 → 安全检测 → 查询向量化 → 混合检索 → Rerank → LLM生成 → 后处理 → 用户回答
    ↓                                                              ↓
敏感词过滤                                                   引用来源标注
                                                              风险提示添加
```

---

## 3. 功能需求详细规格

### 3.1 安全检测模块

#### 功能描述
检测用户输入是否包含敏感信息，并进行适当处理。

#### 实现要求
```
输入: 用户问题文本
输出: {
    "passed": bool,           # 是否通过检测
    "flagged_types": [],      # 被标记的敏感类型
    "sanitized_text": str,    # 脱敏后的文本
    "risk_level": str          # low/medium/high
}
```

#### 检测范围
| 敏感类型 | 检测规则                   | 处理方式 |
| -------- | -------------------------- | -------- |
| 个人信息 | 姓名、手机号、身份证、地址 | 脱敏替换 |
| 医疗隐私 | 具体患者信息、病历号       | 完全过滤 |
| 政治敏感 | 特定关键词列表             | 拒绝回答 |
| 暴力色情 | 特定关键词列表             | 拒绝回答 |

#### 代码示例
```python
# 期望的实现方式
class SafetyChecker:
    def __init__(self):
        self.sensitive_patterns = [...]  # 正则表达式列表
        self.privacy_keywords = [...]    # 关键词列表
        
    def check(self, text: str) -> SafetyResult:
        # 实现敏感词检测逻辑
        # 返回检测结果和脱敏文本
```

---

### 3.2 文档解析模块

#### 功能描述
解析各种格式的医疗文档，提取文本和表格内容。

#### 支持格式
| 格式         | 库/工具     | 表格支持       | 图片OCR    |
| ------------ | ----------- | -------------- | ---------- |
| PDF          | PyMuPDF     | ✅ 提取表格结构 | ✅ 可选     |
| Word (.docx) | python-docx | ✅ 表格元素     | ❌          |
| Markdown     | markdown库  | ✅ 表格语法     | ✅ 图片链接 |
| TXT          | 内置        | ❌              | ❌          |

#### 输出数据结构
```python
class ParsedDocument(BaseModel):
    doc_id: str                          # 文档唯一ID
    title: str                            # 文档标题
    source: str                           # 文件来源
    created_at: datetime                  # 创建时间
    content_type: str                     # "text" | "table" | "mixed"
    text_content: str                     # 纯文本内容
    tables: List[Table]                   # 提取的表格列表
    metadata: Dict[str, Any]             # 元数据
    
class Table(BaseModel):
    headers: List[str]                    # 表头
    rows: List[List[str]]                  # 数据行
    caption: Optional[str]                 # 表格标题
```

#### 错误处理
| 错误类型   | 处理方式                   | 日志级别 |
| ---------- | -------------------------- | -------- |
| 格式不支持 | 返回错误信息，标记文档状态 | WARNING  |
| 解析失败   | 记录错误，返回部分结果     | ERROR    |
| 文件损坏   | 标记状态，通知管理员       | CRITICAL |

---

### 3.3 语义分片模块

#### 功能描述
将文档分成适当大小的语义片段。

#### 配置参数
```yaml
chunking:
  # 基础参数
  chunk_size: 512              # 片段token数
  chunk_overlap: 50             # 重叠token数
  
  # 分片策略
  strategy: "semantic"         # semantic | fixed | recursive
  separator: ["\n\n", "\n", "。", "！", "？"]  # 语义分隔符
  
  # 特殊处理
  preserve_tables: true        # 表格保持完整
  min_chunk_length: 50         # 最小片段长度（字符）
  max_chunk_length: 1000        # 最大片段长度（字符）
  
  # 元数据保留
  include_metadata: true
  metadata_fields: ["source", "page", "title", "heading"]
```

#### 输出数据结构
```python
class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str                    # 分片文本内容
    token_count: int                 # token数量
    metadata: ChunkMetadata
    embedding: Optional[List[float]] # 可选：预计算embedding
    
class ChunkMetadata(BaseModel):
    source_file: str
    page_number: Optional[int]
    section_title: Optional[str]
    char_count: int
    position: int                    # 在原文档中的位置
```

---

### 3.4 混合检索模块

#### 功能描述
融合稀疏检索（BM25）和稠密检索（向量相似度）。

#### 检索流程
```
用户查询
    ↓
查询向量化 (Embedding Model)
    ↓
┌─────────────────┐
↓                 ↓                 
BM25检索          向量检索          
↓                 ↓                 
Top-50            Top-50           
↓                 ↓                 
└────────┬────────┘
         ↓
    结果融合 (RRF)
         ↓
    Top-20 候选
         ↓
    Rerank重排
         ↓
    Top-5 最终结果
```

#### 配置参数
```yaml
retrieval:
  # 向量检索
  vector:
    top_k: 50                      # 召回数量
    similarity_threshold: 0.5      # 相似度阈值
    
  # BM25检索
  bm25:
    top_k: 50
    k1: 1.5
    b: 0.75
    
  # 结果融合
  fusion:
    method: "rrf"                  # rrf | weighted
    rrf_k: 60                       # RRF参数
    weights:
      vector: 0.6
      bm25: 0.4
      
  # 最终输出
  final_top_k: 5
```

#### 代码示例
```python
class HybridRetriever:
    def __init__(self, vector_db, bm25_index):
        self.vector_db = vector_db
        self.bm25_index = bm25_index
        
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[RetrievedNode]:
        # 1. 向量检索
        vector_results = await self.vector_db.search(
            query, top_k=50, filters=filters
        )
        
        # 2. BM25检索
        bm25_results = self.bm25_index.search(query, top_k=50)
        
        # 3. RRF融合
        fused_results = self._reciprocal_rank_fusion(
            vector_results, bm25_results
        )
        
        return fused_results[:top_k]
```

---

### 3.5 重排序模块

#### 功能描述
使用Cross-Encoder模型对检索结果进行相关性重排序。

#### 配置参数
```yaml
reranker:
  model_name: "BAAI/bge-reranker-v2-m3"
  device: "cuda"                    # cuda | cpu
  batch_size: 8
  max_length: 512
  
  # 重排策略
  return_documents: true            # 返回原始文档内容
  apply_normalization: true         # 归一化分数
```

#### 排序逻辑
```python
class Reranker:
    def rerank(
        self, 
        query: str, 
        candidates: List[RetrievedNode]
    ) -> List[RerankedNode]:
        # 构建查询-文档对
        pairs = [(query, node.content) for node in candidates]
        
        # 批量计算相关性分数
        scores = self.model.predict(pairs)
        
        # 按分数排序
        ranked = sorted(
            zip(candidates, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [node for node, score in ranked]
```

---

### 3.6 答案生成模块

#### 功能描述
使用LLM生成带引用来源的准确回答。

#### Prompt模板
```python
SYSTEM_PROMPT = """你是一个专业的医疗知识问答助手。你的职责是：
1. 基于提供的参考信息回答用户问题
2. 在回答中明确标注信息来源
3. 如果参考信息不足以回答问题，明确告知用户
4. 避免编造信息，只基于提供的参考内容回答

回答格式要求：
- 使用清晰的段落结构
- 关键医疗术语需要解释
- 引用格式：[来源X](文件名称#页码)

风险提示格式：
⚠️ 重要提示：[相关风险说明]
"""

USER_PROMPT_TEMPLATE = """
## 参考信息
{contexts}

## 用户问题
{question}

## 要求
1. 仅基于以上参考信息回答问题
2. 如果问题超出参考信息范围，请明确说明"根据提供的信息，无法完全回答此问题"
3. 给出置信度评分(0-1)和风险提示
"""
```

#### 配置参数
```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  api_base: "https://api.deepseek.com/v1"
  
  # 生成参数
  temperature: 0.3                  # 较低温度保证准确性
  max_tokens: 2000
  top_p: 0.9
  
  # 响应格式
  response_format:
    include_citations: true         # 包含引用来源
    include_confidence: true        # 包含置信度
    include_warnings: true           # 包含风险提示
```

---

### 3.7 置信度评估模块

#### 评估指标
| 指标         | 计算方式                   | 阈值 |
| ------------ | -------------------------- | ---- |
| 上下文相关性 | 检索结果与问题的平均相似度 | >0.6 |
| 答案完整性   | 回答覆盖问题要点的比例     | >0.7 |
| 一致性       | 多个检索结果的一致程度     | >0.5 |
| 来源可靠性   | 引用来源的权威性评分       | >0.5 |

#### 综合置信度计算
```python
def calculate_confidence(
    context_relevance: float,
    answer_completeness: float,
    consistency: float,
    source_reliability: float
) -> float:
    """
    加权平均计算综合置信度
    """
    weights = {
        "context_relevance": 0.3,
        "answer_completeness": 0.3,
        "consistency": 0.2,
        "source_reliability": 0.2
    }
    
    confidence = (
        weights["context_relevance"] * context_relevance +
        weights["answer_completeness"] * answer_completeness +
        weights["consistency"] * consistency +
        weights["source_reliability"] * source_reliability
    )
    
    return round(confidence, 2)
```

#### 置信度等级
| 等级   | 分数范围 | 显示方式 | 建议操作     |
| ------ | -------- | -------- | ------------ |
| 高     | 0.8-1.0  | 绿色标签 | 可直接使用   |
| 中     | 0.5-0.8  | 黄色标签 | 建议核实     |
| 低     | 0.3-0.5  | 橙色标签 | 需要补充信息 |
| 不可靠 | 0.0-0.3  | 红色标签 | 不建议使用   |

---

### 3.8 降级处理策略

#### 触发条件
| 条件           | 处理策略                 |
| -------------- | ------------------------ |
| 检索结果为空   | 返回"未找到相关信息"提示 |
| 置信度 < 0.5   | 显示低置信度警告         |
| 检索结果不完整 | 部分回答+信息缺口说明    |
| LLM调用失败    | 返回缓存结果或错误信息   |

#### 降级响应模板
```python
FALLBACK_RESPONSES = {
    "no_results": {
        "answer": "抱歉，我在知识库中没有找到与您问题相关的信息。",
        "suggestions": [
            "请尝试使用更通用的关键词",
            "检查问题是否与医学相关",
            "联系医疗专业人员获取帮助"
        ],
        "confidence": 0.0
    },
    
    "low_confidence": {
        "answer": "根据找到的部分信息，您的问题可能的答案是：\n{partial_answer}",
        "warning": "⚠️ 警告：此答案的置信度较低({confidence})，仅供参考，请以专业医疗建议为准。",
        "confidence": "{calculated_confidence}"
    }
}
```

---

### 3.9 引用来源标注

#### 引用格式
```python
class Citation(BaseModel):
    source_id: str                  # 来源ID
    file_name: str                   # 文件名
    page_number: Optional[int]       # 页码
    chunk_content: str               # 引用的原文片段
    relevance_score: float           # 相关性分数
    position: str                    # "direct" | "partial" | "related"
```

#### 显示格式示例
```
根据《临床诊疗指南》，糖尿病的诊断标准包括[1]：

**空腹血糖**
> 空腹血糖≥7.0mmol/L可诊断为糖尿病[1](指南.pdf#第15页)

[1] 《中国2型糖尿病防治指南(2020年版)》- 中华医学会糖尿病学分会
```

---

### 3.10 风险提示模块

#### 提示类型
| 类型     | 触发条件     | 提示内容示例                         |
| -------- | ------------ | ------------------------------------ |
| 一般风险 | 所有医疗回答 | "本回答仅供参考，不构成医疗建议"     |
| 用药风险 | 涉及药物内容 | "请在医生指导下使用药物"             |
| 诊断风险 | 涉及诊断建议 | "AI无法提供正式诊断，请咨询专业医生" |
| 紧急风险 | 涉及紧急情况 | "如有紧急症状，请立即就医"           |

#### 实现代码
```python
class RiskWarningGenerator:
    def generate_warnings(
        self, 
        answer: str, 
        context: List[RetrievedNode]
    ) -> List[RiskWarning]:
        warnings = []
        
        # 通用风险提示
        warnings.append(RiskWarning(
            type="general",
            message="本回答由AI生成，仅供参考，不能替代专业医疗建议。",
            priority="low"
        ))
        
        # 检查药物相关
        if self._contains_drug_info(context):
            warnings.append(RiskWarning(
                type="medication",
                message="涉及药物信息，请务必在医生或药师指导下使用。",
                priority="medium"
            ))
            
        # 检查诊断相关
        if self._contains_diagnostic_suggestions(answer):
            warnings.append(RiskWarning(
                type="diagnosis",
                message="AI无法提供正式医学诊断，请咨询医疗专业人员。",
                priority="high"
            ))
            
        return warnings
```

---

### 3.11 多轮对话模块

#### 对话状态管理
```python
class ConversationSession(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message]
    context_documents: List[str]     # 积累的上下文文档ID
    user_preferences: Dict           # 用户偏好设置
    
class Message(BaseModel):
    message_id: str
    role: str                        # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    metadata: Dict                  # 包含置信度、引用等信息
```

#### 上下文管理策略
```python
class ContextManager:
    def __init__(self):
        self.max_history: int = 10          # 最大历史消息数
        self.max_context_length: int = 4000 # 最大上下文token数
        self.relevance_threshold: float = 0.3  # 历史消息相关性阈值
        
    def build_context(
        self, 
        current_query: str,
        history: List[Message],
        retrieved_docs: List[Node]
    ) -> str:
        # 1. 筛选相关历史消息
        relevant_history = self._filter_relevant_history(
            current_query, history
        )
        
        # 2. 构建对话历史
        history_text = self._format_history(relevant_history)
        
        # 3. 构建参考文档
        docs_text = self._format_documents(retrieved_docs)
        
        # 4. 组合上下文（注意长度控制）
        context = f"{history_text}\n\n## 参考文档\n{docs_text}"
        
        if self._count_tokens(context) > self.max_context_length:
            context = self._truncate_context(context)
            
        return context
```

---

### 3.12 用户界面 (Streamlit)

#### 页面结构
```
┌─────────────────────────────────────────────────────────────┐
│  🏥 医疗知识库问答系统                                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 💬 智能问答 │  │ 📁 文档管理 │  │ 📊 历史记录  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                      │   │
│  │              问答区域                                 │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ 置信度: 0.85  │  │ 引用来源: 3条 │  │ 耗时: 2.3s   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 输入您的问题...                             [发送]   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 界面功能要求

| 功能区域 | 具体要求                                   |
| -------- | ------------------------------------------ |
| 对话展示 | 显示问答历史、引用来源折叠展开、置信度标签 |
| 输入区域 | 支持多行输入、输入字数统计、清空按钮       |
| 结果展示 | 回答内容高亮、引用可点击跳转、风险提示展示 |
| 文档管理 | 上传、列表、删除、状态查看                 |
| 历史记录 | 按时间排序、支持搜索、可导出               |

---

## 4. 数据模型设计

### 4.1 PostgreSQL 表结构

```sql
-- 文档表
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT,
    status VARCHAR(50) DEFAULT 'pending',  -- pending, processing, completed, failed
    total_pages INTEGER,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    metadata JSONB
);

-- 分片表
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    token_count INTEGER,
    char_count INTEGER,
    position INTEGER,
    page_number INTEGER,
    section_title VARCHAR(500),
    vector_id VARCHAR(255),           -- Qdrant中的向量ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- 会话表
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_title VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    confidence FLOAT,
    citations JSONB,
    warnings JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX idx_messages_session_id ON messages(session_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_messages_created_at ON messages(created_at);
```

### 4.2 Qdrant Collection 配置

```python
COLLECTION_CONFIG = {
    "name": "medical_knowledge",
    "vector_size": 1024,                    # BGE-M3 embedding维度
    "distance": "Cosine",
    
    # 向量参数
    "hnsw": {
        "m": 16,
        "ef_construct": 100
    },
    
    # Payload字段
    "payload_schema": {
        "doc_id": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "content": {"type": "text"},
        "source_file": {"type": "keyword"},
        "page_number": {"type": "integer"},
        "section_title": {"type": "text"},
        "char_count": {"type": "integer"},
        "created_at": {"type": "datetime"}
    }
}
```

### 4.3 Redis 缓存设计

```python
CACHE_CONFIG = {
    # 会话上下文缓存
    "session_context": {
        "key_pattern": "session:{session_id}:context",
        "ttl": 3600,                       # 1小时
        "max_size": 10000                   # 最大token数
    },
    
    # 检索结果缓存
    "retrieval_cache": {
        "key_pattern": "retrieve:{query_hash}",
        "ttl": 300,                        # 5分钟
    },
    
    # LLM响应缓存
    "llm_response": {
        "key_pattern": "llm:{input_hash}",
        "ttl": 1800,                       # 30分钟
    }
}
```

---

## 5. API 接口规范

### 5.1 接口列表

| 方法   | 路径                           | 描述             |
| ------ | ------------------------------ | ---------------- |
| POST   | /api/v1/query                  | 问答接口         |
| POST   | /api/v1/documents/upload       | 上传文档         |
| GET    | /api/v1/documents              | 获取文档列表     |
| DELETE | /api/v1/documents/{id}         | 删除文档         |
| GET    | /api/v1/documents/{id}/status  | 获取文档处理状态 |
| POST   | /api/v1/sessions               | 创建会话         |
| GET    | /api/v1/sessions/{id}/messages | 获取会话消息     |
| POST   | /api/v1/sessions/{id}/messages | 发送消息         |
| DELETE | /api/v1/sessions/{id}          | 删除会话         |

### 5.2 问答接口

**请求**
```python
# POST /api/v1/query
{
    "question": "糖尿病的诊断标准是什么？",
    "session_id": "uuid",           # 可选，用于多轮对话
    "filters": {                     # 可选，检索过滤条件
        "source_file": "*.pdf",
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        }
    },
    "options": {
        "include_citations": True,
        "include_confidence": True,
        "include_warnings": True,
        "top_k": 5
    }
}
```

**响应**
```python
{
    "answer": "根据《中国2型糖尿病防治指南》...",
    "confidence": 0.85,
    "citations": [
        {
            "source_id": "uuid",
            "file_name": "糖尿病指南.pdf",
            "page_number": 15,
            "content": "空腹血糖≥7.0mmol/L...",
            "relevance_score": 0.92
        }
    ],
    "warnings": [
        {
            "type": "medication",
            "message": "请在医生指导下参考此信息",
            "priority": "medium"
        }
    ],
    "session_id": "uuid",
    "processing_time": 2.3,
    "metadata": {
        "retrieved_chunks": 5,
        "model_used": "deepseek-chat",
        "tokens_used": 1500
    }
}
```

### 5.3 文档上传接口

**请求**
```python
# POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: (binary)  # 支持 PDF, DOCX, MD, TXT
title: "糖尿病防治指南"  # 可选，默认使用文件名
metadata: {}  # 可选，自定义元数据
```

**响应**
```python
{
    "document_id": "uuid",
    "title": "糖尿病防治指南",
    "file_name": "指南.pdf",
    "file_type": "pdf",
    "status": "processing",
    "message": "文档已上传，正在处理中..."
}
```

---

## 6. 配置参数汇总

### 6.1 环境变量
```bash
# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=medical_rag
POSTGRES_USER=postgres_admin
POSTGRES_PASSWORD=postgres_123

# 向量数据库
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=medical_knowledge

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# LLM配置
DEEPSEEK_API_KEY=***
DEEPSEEK_API_BASE=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cuda

# 应用配置
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```

### 6.2 完整配置YAML
```yaml
# config.yaml

app:
  name: "医疗知识库RAG系统"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

database:
  postgresql:
    host: "localhost"
    port: 5432
    database: "medical_rag"
    pool_size: 10
    
  qdrant:
    host: "localhost"
    port: 6333
    collection: "medical_knowledge"
    
  redis:
    host: "localhost"
    port: 6379
    db: 0

models:
  embedding:
    name: "BAAI/bge-m3"
    device: "cuda"
    dimension: 1024
    
  reranker:
    name: "BAAI/bge-reranker-v2-m3"
    device: "cuda"
    batch_size: 8
    
  llm:
    provider: "deepseek"
    model: "deepseek-chat"
    temperature: 0.3
    max_tokens: 2000

rag:
  chunking:
    chunk_size: 512
    chunk_overlap: 50
    strategy: "semantic"
    
  retrieval:
    vector_top_k: 50
    bm25_top_k: 50
    fusion_method: "rrf"
    final_top_k: 5
    
  generation:
    include_citations: true
    include_confidence: true
    include_warnings: true

safety:
  enable: true
  sensitive_words_check: true
  privacy_protection: true
```

---

## 7. 性能指标要求

### 7.1 响应时间要求
| 操作                   | 目标    | 最大 |
| ---------------------- | ------- | ---- |
| 文档解析 (单文件)      | < 10s   | 30s  |
| 向量生成 (1000 chunks) | < 5s    | 15s  |
| 单轮问答 (P50)         | < 2s    | 5s   |
| 单轮问答 (P95)         | < 5s    | 10s  |
| 文档检索               | < 500ms | 1s   |

### 7.2 准确率要求
| 指标         | 目标  | 测试方法       |
| ------------ | ----- | -------------- |
| 检索召回率@5 | > 85% | 人工标注测试集 |
| 答案相关性   | > 80% | 用户反馈评分   |
| 引用准确率   | > 90% | 人工验证       |

### 7.3 可用性要求
| 指标       | 要求   |
| ---------- | ------ |
| 系统可用性 | 99.5%  |
| 错误率     | < 0.5% |
| 缓存命中率 | > 30%  |

---

## 8. 项目结构

```
medical-rag/
├── config/
│   ├── __init__.py
│   ├── settings.py          # 配置加载
│   └── config.yaml          # 配置文件
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── query.py     # 问答API
│   │   │   ├── documents.py # 文档API
│   │   │   └── sessions.py  # 会话API
│   │   └── deps.py          # 依赖注入
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── safety.py        # 安全检测
│   │   ├── rag_engine.py    # RAG核心引擎
│   │   └── confidence.py    # 置信度评估
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document.py      # 文档服务
│   │   ├── retrieval.py     # 检索服务
│   │   ├── generation.py    # 生成服务
│   │   └── session.py       # 会话服务
│   │
│   └── models/
│       ├── __init__.py
│       ├── schemas.py       # Pydantic模型
│       └── database.py     # 数据库模型
│
├── rag/
│   ├── __init__.py
│   ├── parser/              # 文档解析
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pdf_parser.py
│   │   ├── docx_parser.py
│   │   └── markdown_parser.py
│   │
│   ├── chunking/           # 语义分片
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── semantic_chunker.py
│   │
│   ├── retrieval/          # 检索模块
│   │   ├── __init__.py
│   │   ├── vector_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── reranker/          # 重排序
│   │   ├── __init__.py
│   │   └── cross_encoder.py
│   │
│   └── generation/        # 生成模块
│       ├── __init__.py
│       ├── prompt.py
│       └── llm_generator.py
│
├── data/
│   ├── raw_documents/      # 原始文档存储
│   ├── processed/          # 处理后的数据
│   └── cache/              # 缓存数据
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│   ├── init_db.py          # 初始化数据库
│   ├── init_vector_db.py   # 初始化向量库
│   └── benchmark.py        # 性能测试
│
├── streamlit_app/
│   ├── app.py              # Streamlit主应用
│   ├── pages/
│   │   ├── query.py        # 问答页面
│   │   ├── documents.py    # 文档管理页面
│   │   └── history.py      # 历史记录页面
│   └── components/
│       ├── chat.py
│       └── document_list.py
│
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 9. 验收标准

### 9.1 功能验收清单
| 功能     | 验收条件                  | 测试方法     |
| -------- | ------------------------- | ------------ |
| 文档上传 | 支持PDF/Word/Markdown格式 | 上传测试文件 |
| 文档解析 | 正确提取文本和表格        | 对比原始文档 |
| 语义分片 | 分片大小符合配置          | 抽样检查     |
| 混合检索 | 返回相关结果              | 查询测试     |
| Rerank   | 结果相关性提升            | 人工评估     |
| 答案生成 | 生成准确回答              | 问答测试     |
| 置信度   | 正确评估和显示            | 人工核对     |
| 引用来源 | 正确标注来源              | 核实引用     |
| 风险提示 | 正确显示警告              | 触发测试     |
| 多轮对话 | 保持上下文                | 连续对话     |
| 安全检测 | 检测敏感词                | 输入测试     |

### 9.2 端到端测试场景
```python
TEST_SCENARIOS = [
    {
        "name": "基础问答",
        "input": "糖尿病的诊断标准是什么？",
        "expected": {
            "has_answer": True,
            "has_citation": True,
            "confidence_range": (0.6, 1.0)
        }
    },
    {
        "name": "无结果降级",
        "input": "某个完全不相关的问题xyz123",
        "expected": {
            "confidence": 0.0,
            "has_fallback_message": True
        }
    },
    {
        "name": "敏感词过滤",
        "input": "帮我写一封投诉信投诉XXX",
        "expected": {
            "blocked": True,
            "error_message": "包含敏感内容"
        }
    }
]
```

---

## 10. 附录

### 10.1 术语表
| 术语          | 英文                           | 说明                 |
| ------------- | ------------------------------ | -------------------- |
| RAG           | Retrieval-Augmented Generation | 检索增强生成         |
| BM25          | Best Matching 25               | 基于词项的检索算法   |
| RRF           | Reciprocal Rank Fusion         | 倒数排名融合         |
| Cross-Encoder | Cross-Encoder                  | 交叉编码器重排序模型 |
| Chunk         | Chunk                          | 文档分片             |

### 10.2 参考资源
- LlamaIndex文档: https://docs.llamaindex.ai
- BGE-M3模型: https://huggingface.co/BAAI/bge-m3
- Qdrant文档: https://qdrant.tech/documentation/
- FastAPI文档: https://fastapi.tiangolo.com/
- Streamlit文档: https://docs.streamlit.io/
