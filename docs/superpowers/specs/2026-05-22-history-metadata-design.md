# 问答历史附加信息存储与显示

## 背景问题

当前问答历史只存储了 `role` 和 `content`，医疗免责声明、引用来源等附加信息（confidence、citations、warnings）虽然存入了数据库 `Message.extra_data`，但前端加载历史时只读取了 `role` 和 `content`，导致历史页面无法显示置信度、警告和引用来源。

## 目标

1. 加载历史消息时保留完整元数据
2. 前端统一渲染置信度 badge、警告和引用来源
3. 不破坏现有数据库结构（`extra_data` JSON 列继续使用）

## 实现方案

### 1. API 层改造

**文件**: `app/api/routes/sessions.py`

修改 `/sessions/{session_id}/messages` 接口，从 `Message.extra_data` 反序列化 `confidence`/`citations`/`warnings`，返回结构化字段：

```python
# 返回结构
{
    "message_id": str,
    "role": str,          # "user" | "assistant"
    "content": str,
    "timestamp": datetime,
    "confidence": float | None,   # 从 metadata 反序列化
    "citations": list | None,
    "warnings": list | None,
}
```

### 2. Message Schema 扩展

**文件**: `app/models/schemas.py`

扩展 `Message` schema，增加显式字段：

```python
class Message(BaseModel):
    message_id: str
    role: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None   # 反序列化自 metadata
    citations: list | None = None
    warnings: list | None = None
```

### 3. 前端历史页面改造

**文件**: `streamlit_app/pages/history.py`

加载历史消息时保留完整 metadata，并在 UI 中显示置信度、警告和引用来源。

### 4. 前端 query.py 兼容性

**文件**: `streamlit_app/pages/query.py`

确保加载历史消息时保留 metadata，与 history.py 保持一致。

## 数据流

```
数据库 (Message.extra_data JSON)
    ↓
API 反序列化 (confidence/citations/warnings)
    ↓
MessageSchema 返回
    ↓
前端渲染 (confidence badge + warnings + citations)
```

## 关键文件

| 文件 | 改动 |
|------|------|
| `app/models/schemas.py` | Message schema 增加显式字段 |
| `app/api/routes/sessions.py` | 反序列化 metadata 到显式字段 |
| `streamlit_app/pages/history.py` | 加载历史时保留并渲染 metadata |
| `streamlit_app/pages/query.py` | 加载历史时保留 metadata |