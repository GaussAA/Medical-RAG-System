"""Document management API routes — upload, list, delete, batch, and consistency check."""

import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from loguru import logger

from src.common.database import get_session_factory
from src.common.models import (
    BatchDeleteRequest,
    BatchOperationResponse,
    BatchUpdateRequest,
    BatchUploadItem,
    BatchUploadResponse,
    BatchUploadStatus,
    BM25RebuildResponse,
    ChunkListResponse,
    ChunkResponse,
    ChunkUpdateRequest,
    ConsistencyCheckResponse,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentStatus,
    DocumentUpdateRequest,
    DocumentUploadResponse,
    OrphanCleanupResponse,
    RetrievedNode,
)
from src.conversation import ConsistencyChecker, ConsistencyCheckerPort
from src.documents import DocumentStore, RetrievalIndexer
from src.documents.background import process_batch_documents_background, process_document_background
from src.documents.models import Document

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MAX_BATCH_SIZE = 50
MAX_UPLOAD_SIZE_MB = 50
_MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = None,
) -> DocumentUploadResponse:
    document_service = request.app.state.container.document_service

    allowed_types = [".md", ".markdown"]
    original_filename = file.filename or "Untitled"
    file_ext = Path(original_filename).suffix.lower()

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}",
        )

    # Sanitize filename for safe storage

    safe_filename = re.sub(r'[<>:"/\\|?*]', "_", original_filename)

    # Check for duplicate in raw_documents
    raw_file_path = Path(f"data/raw_documents/{safe_filename}")
    if raw_file_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Document '{original_filename}' already exists",
        )

    doc_id = str(uuid.uuid4())
    doc_title = title or original_filename

    # Save original file to raw_documents
    raw_file_path.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    if len(content) > _MAX_UPLOAD_SIZE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB exceeds maximum {MAX_UPLOAD_SIZE_MB}MB",
        )
    with open(raw_file_path, "wb") as f:
        f.write(content)

    # Initialize document entry (writes to PostgreSQL)
    await document_service.init_document(doc_id, str(raw_file_path), doc_title)

    asyncio.create_task(process_document_background(doc_id, str(raw_file_path), title=doc_title))

    return DocumentUploadResponse(
        document_id=doc_id,
        title=title or original_filename,
        file_name=original_filename,
        file_type=file_ext[1:],
        status="processing",
        message="Document uploaded successfully, processing in background...",
    )


MAX_BATCH_SIZE = 50
MAX_CONCURRENT = 5


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_documents_batch(
    request: Request,
    files: list[UploadFile] = File(...),
) -> BatchUploadResponse:
    """Batch upload multiple documents for processing."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_SIZE} files per batch",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    allowed_types = [".md", ".markdown"]
    batch_id = str(uuid.uuid4())
    document_service = request.app.state.container.document_service

    items: list[BatchUploadItem] = []
    file_infos: list[dict] = []
    succeeded = 0
    failed = 0
    duplicate = 0

    # Pre-process all files: validate, save, initialize DB records
    for file in files:
        original_filename = file.filename or "Untitled"
        file_ext = Path(original_filename).suffix.lower()

        # Type validation
        if file_ext not in allowed_types:
            items.append(
                BatchUploadItem(
                    document_id="",
                    file_name=original_filename,
                    status="failed",
                    error_message=f"Unsupported file type: {file_ext}",
                )
            )
            failed += 1
            continue

        safe_filename = re.sub(r'[<>:"/\\|?*]', "_", original_filename)
        raw_file_path = Path(f"data/raw_documents/{safe_filename}")

        # Check duplicate by filename
        if raw_file_path.exists():
            items.append(
                BatchUploadItem(
                    document_id="",
                    file_name=original_filename,
                    status="duplicate",
                    error_message="File already exists",
                )
            )
            duplicate += 1
            continue

        doc_id = str(uuid.uuid4())

        # Save file to disk
        try:
            raw_file_path.parent.mkdir(parents=True, exist_ok=True)
            content = await file.read()
            with open(raw_file_path, "wb") as f:
                f.write(content)
        except Exception as e:
            items.append(
                BatchUploadItem(
                    document_id=doc_id,
                    file_name=original_filename,
                    status="failed",
                    error_message=f"Failed to save file: {str(e)}",
                )
            )
            failed += 1
            continue

        # Initialize document in PostgreSQL
        try:
            await document_service.init_document(doc_id, str(raw_file_path), original_filename)
        except Exception as e:
            items.append(
                BatchUploadItem(
                    document_id=doc_id,
                    file_name=original_filename,
                    status="failed",
                    error_message=f"Failed to initialize document: {str(e)}",
                )
            )
            failed += 1
            continue

        file_infos.append(
            {
                "doc_id": doc_id,
                "file_path": str(raw_file_path),
                "title": original_filename,
            }
        )
        items.append(
            BatchUploadItem(
                document_id=doc_id,
                file_name=original_filename,
                status="processing",
            )
        )
        succeeded += 1

    # Initialize batch status in app.state
    if not hasattr(request.app.state, "batch_upload_status"):
        request.app.state.batch_upload_status = {}
    request.app.state.batch_upload_status[batch_id] = BatchUploadStatus(
        batch_id=batch_id,
        total=len(files),
        processing=succeeded,
        completed=0,
        failed=failed,
        duplicate=duplicate,
        items=items,
    )

    # Start unified batch processing (single embedding model load for all docs)
    asyncio.create_task(
        process_batch_documents_background(
            batch_id,
            file_infos,
            request.app.state,
        )
    )

    message = f"Batch upload started: {succeeded} files being processed"
    if duplicate > 0:
        message += f", {duplicate} duplicates skipped"
    if failed > 0:
        message += f", {failed} failed"

    return BatchUploadResponse(
        batch_id=batch_id,
        total=len(files),
        succeeded=succeeded,
        failed=failed,
        duplicate=duplicate,
        items=items,
        message=message,
    )


@router.get("/upload/batch/{batch_id}/status", response_model=BatchUploadStatus)
async def get_batch_upload_status(
    request: Request,
    batch_id: str,
) -> BatchUploadStatus:
    """Get the real-time status of a batch upload operation."""
    if not hasattr(request.app.state, "batch_upload_status"):
        raise HTTPException(status_code=404, detail="Batch not found")

    status = request.app.state.batch_upload_status.get(batch_id)
    if not status:
        raise HTTPException(status_code=404, detail="Batch not found")

    return status


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    status: str | None = Query(
        None,
        description="Filter by status (pending, processing, completed, failed, archived)",
    ),
    tags: str | None = Query(None, description="Filter by tags (comma-separated)"),
    file_type: str | None = Query(None, description="Filter by file type (pdf, docx, md, txt)"),
    date_from: datetime | None = Query(None, description="Filter by creation date (from)"),
    date_to: datetime | None = Query(None, description="Filter by creation date (to)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> DocumentListResponse:
    """List documents with filtering and pagination."""

    # Parse tags from comma-separated string
    tag_list = [t.strip().lower() for t in tags.split(",")] if tags else None

    async with DocumentStore() as store:
        documents, total = await store.list_documents(
            status=status,
            tags=tag_list,
            file_type=file_type,
            date_from=date_from,
            date_to=date_to,
            page=page,
            page_size=page_size,
        )

    return DocumentListResponse(
        documents=[
            DocumentStatus(
                id=str(doc.id),
                title=doc.title,
                status=doc.status,
                total_chunks=doc.total_chunks,
                tags=list(doc.tags) if doc.tags else [],
                created_at=doc.created_at,
            )
            for doc in documents
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete("/{document_id}", response_model=dict[str, str])
async def delete_document(
    request: Request,
    document_id: str,
) -> dict[str, str]:
    document_service = request.app.state.container.document_service
    success = await document_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}


@router.get("/{document_id}/status", response_model=DocumentStatus)
async def get_document_status(
    request: Request,
    document_id: str,
) -> DocumentStatus:
    """从数据库直接读取文档状态"""

    factory = get_session_factory()
    async with factory() as session:
        doc = await session.get(Document, uuid.UUID(document_id))
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentStatus(
            id=str(doc.id),
            title=doc.title,
            status=doc.status,
            total_chunks=doc.total_chunks,
        )


@router.patch("/{document_id}")
async def update_document(
    request: Request,
    document_id: str,
    update_data: DocumentUpdateRequest,
) -> DocumentStatus:
    """Update document tags and/or status."""

    async with DocumentStore() as store:
        doc = await store.update_document(
            doc_id=document_id,
            status=update_data.status,
            tags=update_data.tags,
            operation=update_data.operation,
        )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentStatus(
        id=str(doc.id),
        title=doc.title,
        status=doc.status,
        total_chunks=doc.total_chunks,
        tags=list(doc.tags) if doc.tags else [],
        created_at=doc.created_at,
    )


@router.get("/{document_id}/preview", response_model=DocumentPreviewResponse)
async def get_document_preview(
    request: Request,
    document_id: str,
) -> DocumentPreviewResponse:
    """Get document preview (text preview or processing status)."""

    factory = get_session_factory()
    async with factory() as session:
        doc = await session.get(Document, uuid.UUID(document_id))
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # If still processing, return 202
        if doc.status == "processing":
            return DocumentPreviewResponse(
                document_id=str(doc.id),
                title=doc.title,
                preview_text="Document is still being processed...",
                preview_is_full=False,
                total_pages=None,
                total_chunks=None,
            )

        # Get preview from extra_data or generate from content
        preview_text = ""
        preview_is_full = False

        if doc.extra_data and "preview" in doc.extra_data:
            preview_text = doc.extra_data["preview"]
            preview_is_full = doc.extra_data.get("preview_is_full", False)
        else:
            # Try to load from processed file (first 500 chars)
            try:
                processed_file = Path(f"data/processed/{Path(doc.file_path).stem}.txt")
                if processed_file.exists():
                    content = processed_file.read_text(encoding="utf-8")
                    if len(content) <= 500:
                        preview_text = content
                        preview_is_full = True
                    else:
                        preview_text = content[:500] + "..."
                        preview_is_full = False
            except Exception as e:
                logger.warning(f"Failed to read preview for {doc.file_path}: {e}")
                preview_text = "Preview not available"

        return DocumentPreviewResponse(
            document_id=str(doc.id),
            title=doc.title,
            preview_text=preview_text,
            preview_is_full=preview_is_full,
            total_pages=doc.total_pages,
            total_chunks=doc.total_chunks,
        )


@router.get("/{document_id}/chunks", response_model=ChunkListResponse)
async def get_document_chunks(
    request: Request,
    document_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
) -> ChunkListResponse:
    """Get all chunks for a document with pagination."""

    async with DocumentStore() as store:
        chunks, total = await store.get_chunks(document_id, page=page, page_size=page_size)

    return ChunkListResponse(
        chunks=[
            ChunkResponse(
                chunk_id=str(chunk.id),
                doc_id=str(chunk.doc_id),
                content=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                position=chunk.position or 0,
                page_number=None,  # Deprecated for Markdown
                section_title=chunk.section_title,
                vector_id=chunk.vector_id,
            )
            for chunk in chunks
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch("/{document_id}/chunks/{chunk_id}")
async def update_chunk(
    request: Request,
    document_id: str,
    chunk_id: str,
    update_data: ChunkUpdateRequest,
) -> dict[str, str]:
    """Update a chunk's content and/or metadata."""

    # Update chunk in PostgreSQL
    async with DocumentStore() as store:
        chunk = await store.update_chunk(
            doc_id=document_id,
            chunk_id=chunk_id,
            content=update_data.content,
            section_title=update_data.section_title,
        )
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # If content changed, need to re-index
    if update_data.content:
        indexer = RetrievalIndexer()
        section_parts = document_id.split("_", 1)

        # Create a new RetrievedNode for the updated chunk
        updated_node = RetrievedNode(
            node_id=chunk_id,
            content=update_data.content,
            score=1.0,
            metadata={
                "doc_id": document_id,
                "chunk_id": chunk_id,
                "source_file": section_parts[1] if len(section_parts) > 1 else "",
                "section_title": update_data.section_title or "",
                "heading_tree": {},
                "content_type": "text",
                "char_count": len(update_data.content),
                "position": 0,
            },
        )

        # Re-index: delete old, add new
        await indexer.delete_documents(document_id, 1)
        await indexer.add_documents([updated_node])

        logger.info(f"Chunk {chunk_id} re-indexed after content update")

    return {"chunk_id": chunk_id, "status": "re-indexed"}


@router.delete("/{document_id}/chunks/{chunk_id}")
async def delete_chunk(
    request: Request,
    document_id: str,
    chunk_id: str,
) -> dict[str, str]:
    """Delete a single chunk from a document."""

    async with DocumentStore() as store:
        success = await store.delete_chunk(document_id, chunk_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Delete from vector index

    indexer = RetrievalIndexer()
    await indexer.delete_documents(document_id, 1)

    return {"message": "Chunk deleted successfully"}


@router.post("/batch-delete", response_model=BatchOperationResponse)
async def batch_delete_documents(
    request: Request,
    batch_data: BatchDeleteRequest,
) -> BatchOperationResponse:
    """Delete multiple documents by ID."""

    if len(batch_data.ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 documents per batch operation")

    if not batch_data.ids:
        raise HTTPException(status_code=400, detail="ids array cannot be empty")

    deleted = []
    failed = []

    async with DocumentStore() as store:
        for doc_id in batch_data.ids:
            try:
                # Delete from vector/BM25

                indexer = RetrievalIndexer()
                await indexer.delete_documents(doc_id, 100)  # Approximate max chunks

                # Delete from PostgreSQL
                success = await store.delete_document(doc_id)
                if success:
                    deleted.append(doc_id)
                else:
                    failed.append({"id": doc_id, "error": "not found"})
            except Exception as e:
                failed.append({"id": doc_id, "error": str(e)})

    return BatchOperationResponse(deleted=deleted, failed=failed)


@router.patch("/batch-update", response_model=BatchOperationResponse)
async def batch_update_documents(
    request: Request,
    batch_data: BatchUpdateRequest,
) -> BatchOperationResponse:
    """Update status and/or tags for multiple documents."""

    if len(batch_data.ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 documents per batch operation")

    if not batch_data.ids:
        raise HTTPException(status_code=400, detail="ids array cannot be empty")

    updated = []
    failed = []

    async with DocumentStore() as store:
        for doc_id in batch_data.ids:
            try:
                doc = await store.update_document(
                    doc_id=doc_id,
                    status=batch_data.status,
                    tags=batch_data.tags,
                    operation=batch_data.operation,
                )
                if doc:
                    updated.append(doc_id)
                else:
                    failed.append({"id": doc_id, "error": "not found"})
            except Exception as e:
                failed.append({"id": doc_id, "error": str(e)})

    return BatchOperationResponse(updated=updated, failed=failed)


@router.get("/consistency-check", response_model=ConsistencyCheckResponse)
async def check_consistency(
    request: Request,
    repair: bool = Query(False, description="Automatically repair inconsistencies if found"),
) -> ConsistencyCheckResponse:
    """
    Check consistency across all three stores (PostgreSQL, Qdrant, BM25).

    Returns a detailed report of which documents exist in which stores
    and highlights inconsistencies. Optionally repairs them automatically.
    """
    # ponytail: local import; ConsistencyChecker satisfies the port structurally

    checker: ConsistencyCheckerPort = ConsistencyChecker()
    result = await checker.check_all_consistency(repair=repair)
    return result


@router.post("/cleanup-orphans", response_model=OrphanCleanupResponse)
async def cleanup_orphaned_data(
    request: Request,
) -> OrphanCleanupResponse:
    """
    Remove orphaned entries from Qdrant and BM25 that don't have
    corresponding documents in PostgreSQL.

    This is a maintenance endpoint to clean up after failed deletions
    or other inconsistencies.
    """
    # ponytail: local import; ConsistencyChecker satisfies the port structurally

    checker: ConsistencyCheckerPort = ConsistencyChecker()
    result = await checker.cleanup_orphans()
    return result


@router.post("/rebuild-bm25", response_model=BM25RebuildResponse)
async def rebuild_bm25_index(
    request: Request,
) -> BM25RebuildResponse:
    """
    Rebuild BM25 index from Qdrant payload data.

    Use this when the BM25 index is corrupted or lost but Qdrant
    still contains all document content in payloads. This will scan
    all points in Qdrant and reconstruct the BM25 keyword index.

    Note: This only rebuilds the BM25 index in memory. For persistence,
    ensure bm25_persist_path is configured in config.yaml.
    """

    indexer = RetrievalIndexer()
    result = await indexer.rebuild_bm25_from_qdrant()
    return BM25RebuildResponse(
        success=result["success"],
        documents_rebuilt=result["documents_rebuilt"],
        errors=result.get("errors", []),
    )
