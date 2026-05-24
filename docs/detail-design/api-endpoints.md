# API Endpoints

Complete list of all API endpoints in the Medical RAG System.

## Base URL

```
http://localhost:8000/api/v1
```

## Query Endpoint

### `POST /api/v1/query`

Query the RAG knowledge base.

**Request Body**:
```json
{
  "question": "糖尿病的诊断标准是什么？",
  "session_id": "optional-session-id",
  "filters": {
    "doc_id": "optional-document-id",
    "content_type": "text|table|list"
  },
  "options": {}
}
```

**Response**:
```json
{
  "answer": "根据指南，糖尿病的诊断标准是...",
  "confidence": 0.85,
  "citations": [
    {
      "source_id": "1",
      "file_name": "糖尿病指南.md",
      "verified": true,
      "position": "direct"
    }
  ],
  "warnings": [
    {
      "type": "general",
      "message": "本回答由AI生成，仅供参考",
      "priority": "low"
    }
  ],
  "session_id": "session-uuid",
  "processing_time": 1.23,
  "metadata": {
    "retrieved_chunks": 5,
    "context_relevance": 0.9,
    "answer_completeness": 0.85
  }
}
```

## Document Endpoints

### Upload

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload single document |
| POST | `/api/v1/documents/upload/batch` | Batch upload (max 50 files) |
| GET | `/api/v1/documents/upload/batch/{batch_id}/status` | Get batch upload status |

### Document Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/documents` | List documents (filterable) |
| GET | `/api/v1/documents/{document_id}/status` | Get document status |
| GET | `/api/v1/documents/{document_id}/preview` | Get document preview |
| GET | `/api/v1/documents/{document_id}/chunks` | Get document chunks |
| PATCH | `/api/v1/documents/{document_id}` | Update document tags/status |
| DELETE | `/api/v1/documents/{document_id}` | Delete document |

### Chunk Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| PATCH | `/api/v1/documents/{document_id}/chunks/{chunk_id}` | Update chunk |
| DELETE | `/api/v1/documents/{document_id}/chunks/{chunk_id}` | Delete chunk |

### Batch Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/batch-delete` | Batch delete documents |
| PATCH | `/api/v1/documents/batch-update` | Batch update status/tags |

### Maintenance

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/documents/consistency-check` | Check cross-store consistency |
| POST | `/api/v1/documents/cleanup-orphans` | Clean orphaned index entries |

## Session Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/sessions` | Create session |
| GET | `/api/v1/sessions` | List all sessions |
| GET | `/api/v1/sessions/{session_id}/messages` | Get session messages |
| POST | `/api/v1/sessions/{session_id}/messages` | Add message to session |
| DELETE | `/api/v1/sessions/{session_id}` | Delete session |

## Health & Metrics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/metrics` | Prometheus metrics |

## Query Parameters for List Endpoints

### `GET /api/v1/documents`

| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | Filter by status (pending, processing, completed, failed) |
| tags | string | Filter by tags (comma-separated) |
| file_type | string | Filter by file type (md, markdown) |
| date_from | datetime | Filter by creation date (from) |
| date_to | datetime | Filter by creation date (to) |
| page | int | Page number (default: 1) |
| page_size | int | Items per page (default: 20, max: 100) |

### `GET /api/v1/documents/{document_id}/chunks`

| Parameter | Type | Description |
|-----------|------|-------------|
| page | int | Page number (default: 1) |
| page_size | int | Items per page (default: 50, max: 200) |

## Batch Upload Request

**`POST /api/v1/documents/upload/batch`**

Content-Type: `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| files | list[UploadFile] | Max 50 files, .md or .markdown only |

**Response**:
```json
{
  "batch_id": "uuid",
  "total": 10,
  "succeeded": 8,
  "failed": 1,
  "duplicate": 1,
  "items": [
    {
      "document_id": "uuid",
      "file_name": "guide.md",
      "status": "processing"
    }
  ],
  "message": "Batch upload started: 8 files being processed, 1 duplicates skipped, 1 failed"
}
```

## Batch Status Response

**`GET /api/v1/documents/upload/batch/{batch_id}/status`**

```json
{
  "batch_id": "uuid",
  "total": 10,
  "processing": 3,
  "completed": 5,
  "failed": 1,
  "duplicate": 1,
  "items": [...]
}
```

## Consistency Check

**`GET /api/v1/documents/consistency-check?repair=true`**

**Response**:
```json
{
  "total_documents": 100,
  "consistent_count": 95,
  "inconsistent_count": 5,
  "details": [
    {
      "doc_id": "uuid",
      "in_postgresql": true,
      "in_qdrant": true,
      "in_bm25": false,
      "pg_chunk_count": 10,
      "qdrant_chunk_count": 10,
      "bm25_chunk_count": 0,
      "status": "orphaned_in_index",
      "issues": ["BM25 index missing"]
    }
  ],
  "repair_actions": [
    {"action": "delete_from_qdrant", "doc_id": "uuid", "chunks": 10}
  ]
}
```

## Error Responses

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 409 | Conflict - Duplicate resource |
| 500 | Internal Server Error |