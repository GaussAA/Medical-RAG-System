# Medical RAG System - Detailed Design Documentation

## Documentation Index

本目录包含 Medical RAG System 的模块化详细设计文档，按职责清晰划分。

## 📚 Documentation Structure

| Module                  | File                                                         | Description                                    |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **Architecture**        | [01-architecture-overview.md](./01-architecture-overview.md) | 系统架构、数据流、三层存储设计                 |
| **RAG Pipeline**        | [02-rag-pipeline.md](./02-rag-pipeline.md)                   | 查询引擎、安全检查、检索、重排、生成、置信评估 |
| **Document Processing** | [03-document-processing.md](./03-document-processing.md)     | 解析、语义分块、向量化、文档CRUD               |
| **Retrieval System**    | [04-retrieval-system.md](./04-retrieval-system.md)           | 混合检索、RRF融合、向量检索、BM25              |
| **Session Management**  | [05-session-management.md](./05-session-management.md)       | 会话生命周期、消息持久化、FIFO驱逐             |
| **Data Models**         | [06-data-models.md](./06-data-models.md)                     | PostgreSQL模型、API Schema、三层存储映射       |
| **Configuration**       | [07-configuration.md](./07-configuration.md)                 | YAML配置、环境变量、设备选择                   |
| **GPU Management**      | [08-gpu-memory-management.md](./08-gpu-memory-management.md) | 显存管理、模型迁移、懒加载策略                 |
| **Evaluation System**   | [11-evaluation-system.md](./11-evaluation-system.md)       | RAG评估、基准测试、指标体系                    |

## 🤖 AI Context Management Guide

When working with this codebase, follow this reading order based on your task:

### Quick Lookup
For specific questions about a module, directly reference the corresponding document above.

### Feature Development
```
1. Read 01-architecture-overview.md for system context
2. Read the specific module documentation
3. Reference CLAUDE.md for project-wide conventions
```

### Bug Investigation
```
1. Identify which module is affected
2. Read the relevant module doc for expected behavior
3. Trace through code following the documented flow
```

### Performance Issues
```
1. Read 08-gpu-memory-management.md if GPU-related
2. Read 04-retrieval-system.md for retrieval bottlenecks
3. Check 07-configuration.md for tuning parameters
```

## 🔗 Quick Links

- [Project Requirements](../requirement/medical-rag.md)
- [Data Flow Specification](../requirement/data-flow.md)
- [API Schemas](../../app/models/schemas.py)
- [Database Models](../../app/models/database.py)
- [Main Application](../../app/main.py)
