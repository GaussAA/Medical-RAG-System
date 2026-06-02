import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SafetyResult(BaseModel):
    passed: bool
    flagged_types: list[str] = Field(default_factory=list)
    sanitized_text: str = ""
    risk_level: str = "low"


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
    heading_tree: dict[int, str] | None = None  # {1: "H1", 2: "H2", ...}
    content_type: str | None = None  # "text" | "table" | "list"
    char_count: int = 0
    position: int = 0
    heading_level: int | None = None  # H1=1, H2=2, ... H6=6 for Markdown chunks
    page_number: int | None = None


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    content: str = ""
    token_count: int = 0
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: list[float] | None = None


class RetrievedNode(BaseModel):
    """Unified node type for both retrieved and reranked results."""

    node_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    rerank_score: float | None = None  # Optional score from reranker


# RerankedNode is now an alias for RetrievedNode (they have identical structure)
RerankedNode = RetrievedNode


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
    trace_id: str | None = None  # Optional trace ID for request correlation


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[RiskWarning] = Field(default_factory=list)
    session_id: str
    processing_time: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None  # Echo back the trace_id for correlation


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


class DocumentUpdateRequest(BaseModel):
    tags: list[str] | None = None
    status: str | None = None
    operation: str = "add"  # "add" or "remove"


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


class ConsistencyCheckItem(BaseModel):
    """Single document consistency check result."""

    doc_id: str
    in_postgresql: bool
    in_qdrant: bool
    in_bm25: bool
    pg_chunk_count: int | None = None
    qdrant_chunk_count: int | None = None
    bm25_chunk_count: int | None = None
    status: str = "unknown"  # "consistent", "orphaned_in_index", "orphaned_in_db", "partial"
    issues: list[str] = Field(default_factory=list)


class ConsistencyCheckResponse(BaseModel):
    """Response for consistency check endpoint."""

    total_documents: int
    consistent_count: int
    inconsistent_count: int
    details: list[ConsistencyCheckItem] = Field(default_factory=list)
    repair_actions: list[dict[str, Any]] = Field(default_factory=list)


class OrphanCleanupResponse(BaseModel):
    """Response for orphan cleanup endpoint."""

    cleaned_from_qdrant: int = 0
    cleaned_from_bm25: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)


class BM25RebuildResponse(BaseModel):
    """Response for BM25 rebuild endpoint."""

    success: bool
    documents_rebuilt: int = 0
    errors: list[str] = Field(default_factory=list)


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
    # Direct message count field (separate from computed len(messages))
    msg_count: int = 0


class BatchUploadItem(BaseModel):
    """Single file item in a batch upload response."""

    document_id: str
    file_name: str
    status: str  # "processing" | "completed" | "failed" | "duplicate"
    error_message: str | None = None


class BatchUploadResponse(BaseModel):
    """Response for batch upload endpoint."""

    batch_id: str
    total: int
    succeeded: int
    failed: int
    duplicate: int
    items: list[BatchUploadItem]
    message: str


class BatchUploadStatus(BaseModel):
    """Real-time status of a batch upload operation."""

    batch_id: str
    total: int
    processing: int = 0
    completed: int = 0
    failed: int = 0
    duplicate: int = 0
    items: list[BatchUploadItem] = Field(default_factory=list)
