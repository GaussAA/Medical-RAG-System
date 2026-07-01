"""
Common Pydantic schemas - transitional hub for models used across slices.

In the final architecture, each vertical slice owns its models.py.
This file serves as the migration bridge; models will be gradually
extracted to their respective slices.

Migrated from: app/models/schemas.py
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


# ==================== Document Models ====================


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str | None = None


class ParsedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    source: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    content_type: str = "text"
    text_content: str = ""
    tables: list[TableData] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    source_file: str = ""
    section_title: str | None = None
    heading_tree: dict[int, str] | None = None
    content_type: str | None = None
    char_count: int = 0
    position: int = 0
    heading_level: int | None = None
    page_number: int | None = None


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    content: str = ""
    token_count: int = 0
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: list[float] | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    title: str
    file_name: str
    file_type: str
    status: str
    message: str


class DocumentStatus(BaseModel):
    id: str
    title: str
    status: str
    total_chunks: int | None = None
    error_message: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        allowed = {"pending", "processing", "completed", "failed", "already_exists"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v!r}")
        return v


class DocumentUpdateRequest(BaseModel):
    tags: list[str] | None = None
    status: str | None = None
    operation: str = "add"


class DocumentListResponse(BaseModel):
    documents: list[DocumentStatus]
    total: int
    page: int
    page_size: int


class DocumentPreviewResponse(BaseModel):
    document_id: str
    title: str
    preview_text: str
    preview_is_full: bool = False
    total_pages: int | None = None
    total_chunks: int | None = None


class ChunkResponse(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    position: int
    page_number: int | None = None
    section_title: str | None = None
    vector_id: str | None = None


class ChunkListResponse(BaseModel):
    chunks: list[ChunkResponse]
    total: int
    page: int
    page_size: int


class ChunkUpdateRequest(BaseModel):
    content: str | None = None
    section_title: str | None = None


class BatchDeleteRequest(BaseModel):
    ids: list[str]


class BatchUpdateRequest(BaseModel):
    ids: list[str]
    tags: list[str] | None = None
    status: str | None = None
    operation: str = "add"


class BatchOperationResponse(BaseModel):
    deleted: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    failed: list[dict[str, str]] = Field(default_factory=list)


class BatchUploadItem(BaseModel):
    document_id: str
    file_name: str
    status: str
    error_message: str | None = None


class BatchUploadResponse(BaseModel):
    batch_id: str
    total: int
    succeeded: int
    failed: int
    duplicate: int
    items: list[BatchUploadItem]
    message: str


class BatchUploadStatus(BaseModel):
    batch_id: str
    total: int
    processing: int = 0
    completed: int = 0
    failed: int = 0
    duplicate: int = 0
    items: list[BatchUploadItem] = Field(default_factory=list)


# ==================== Query Models ====================


class RetrievedNode(BaseModel):
    """Unified node type for both retrieved and reranked results."""

    node_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    rerank_score: float | None = None


RerankedNode = RetrievedNode


class SafetyResult(BaseModel):
    passed: bool
    flagged_types: list[str] = Field(default_factory=list)
    sanitized_text: str = ""
    risk_level: str = "low"


class CitationPosition(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    PARAPHRASED = "paraphrased"
    UNVERIFIED = "unverified"


class Citation(BaseModel):
    source_id: str
    document_id: str | None = None
    file_name: str
    page_number: int | None = None
    chunk_content: str = ""
    relevance_score: float = 0.0
    position: CitationPosition = CitationPosition.DIRECT
    verified: bool = False
    quote_in_answer: str | None = None
    verification_message: str | None = None


class RiskWarning(BaseModel):
    type: str
    message: str
    priority: str = "low"


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    filters: dict[str, Any] | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None

    @field_validator("question")
    @classmethod
    def _sanitize_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty")
        if len(v) > 2000:
            v = v[:2000]
        return v

    @field_validator("session_id", mode="before")
    @classmethod
    def _normalize_session_id(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            return v if v else None
        return None


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[RiskWarning] = Field(default_factory=list)
    session_id: str
    processing_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


# ==================== Conversation Models ====================


class Message(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    content: str
    timestamp: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    citations: list[dict[str, Any]] | None = None
    warnings: list[dict[str, Any]] | None = None


class ConversationSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_title: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    messages: list[Message] = Field(default_factory=list)
    context_documents: list[str] = Field(default_factory=list)
    is_active: bool = True
    db_confirmed: bool = Field(default=False)
    msg_count: int = 0


# ==================== Consistency Models ====================


class ConsistencyCheckItem(BaseModel):
    doc_id: str
    in_postgresql: bool
    in_qdrant: bool
    in_bm25: bool
    pg_chunk_count: int | None = None
    qdrant_chunk_count: int | None = None
    bm25_chunk_count: int | None = None
    status: str = "unknown"
    issues: list[str] = Field(default_factory=list)


class ConsistencyCheckResponse(BaseModel):
    total_documents: int
    consistent_count: int
    inconsistent_count: int
    details: list[ConsistencyCheckItem] = Field(default_factory=list)
    repair_actions: list[dict[str, Any]] = Field(default_factory=list)


class OrphanCleanupResponse(BaseModel):
    cleaned_from_qdrant: int = 0
    cleaned_from_bm25: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)


class BM25RebuildResponse(BaseModel):
    success: bool
    documents_rebuilt: int = 0
    errors: list[str] = Field(default_factory=list)
