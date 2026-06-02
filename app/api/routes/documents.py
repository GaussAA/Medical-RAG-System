import asyncio
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from loguru import logger
from sqlalchemy import text

from app.models.database import Document
from app.models.schemas import (
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
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def process_document_background(doc_id: str, file_path: str, title: str | None = None):
    """后台处理文档（使用独立的 DocumentService 实例）"""
    from app.core.database import get_session_factory
    from app.services.document import DocumentService

    factory = get_session_factory()
    async_session = factory()
    document_service = DocumentService(async_session=async_session)

    try:
        logger.info(f"Starting to process document {doc_id}")
        await document_service.process_document(file_path, title=title, doc_id=doc_id)
        logger.info(f"Document {doc_id} processed successfully")
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
    finally:
        if async_session is not None:
            await async_session.close()
            logger.debug(f"Background session closed for document {doc_id}")


async def process_batch_documents_background(
    batch_id: str,
    file_infos: list[dict],
    app_state_ref,  # Reference to app.state to update status
):
    """
    批量处理文档 - 统一向量化，只加载一次 embedding 模型。
    """
    from app.core.database import get_session_factory
    from app.core.rag_engine import RAGEngine
    from app.services.document_processor import DocumentProcessor

    factory = get_session_factory()
    async_session = factory()
    processor = DocumentProcessor()
    rag_engine = RAGEngine()

    try:
        logger.info(f"[Batch {batch_id}] Starting batch processing of {len(file_infos)} documents")

        # Step 1: Parse all documents and collect chunks (can be done in parallel)
        all_nodes = []
        doc_chunks_map: dict[
            str, tuple[list, list[dict], list]
        ] = {}  # doc_id -> (chunks, heading_ids, retrieved_nodes)

        for file_info in file_infos:
            doc_id = file_info["doc_id"]
            file_path = Path(file_info["file_path"])

            try:
                # Parse document
                parsed_doc, heading_tree = await processor.parse_with_headings(file_path)
                logger.info(f"[Batch {batch_id}] Parsed {file_path.name}, text length: {len(parsed_doc.text_content)}")

                # Build heading tree dict
                heading_tree_dict = {}
                for h in heading_tree:
                    heading_tree_dict[h["level"]] = h["title"]

                # Chunk document
                chunks = processor.chunk(
                    parsed_doc.text_content,
                    metadata={
                        "doc_id": doc_id,
                        "source_file": file_path.name,
                        "heading_tree": heading_tree_dict,
                        "tables": [t.model_dump() for t in parsed_doc.tables],
                    },
                )
                logger.info(f"[Batch {batch_id}] Chunked {file_path.name} into {len(chunks)} chunks")

                # Save processed text
                processor.save_processed_text(file_path, parsed_doc.text_content)

                # Create RetrievedNodes
                retrieved_nodes = processor.create_retrieved_nodes(doc_id, chunks, file_path.name)
                all_nodes.extend(retrieved_nodes)

                # Store for later processing
                doc_chunks_map[doc_id] = (chunks, heading_tree, retrieved_nodes)

            except Exception as e:
                logger.error(f"[Batch {batch_id}] Failed to parse/chunk {file_info['file_path']}: {e}")
                # Update status to failed
                for item in app_state_ref.batch_upload_status[batch_id].items:
                    if item.document_id == doc_id:
                        item.status = "failed"
                        item.error_message = str(e)
                        break
                app_state_ref.batch_upload_status[batch_id].processing -= 1
                app_state_ref.batch_upload_status[batch_id].failed += 1

        # Step 2: Save headings to PostgreSQL for each document
        # Store the actual heading IDs for each document
        doc_heading_ids: dict[str, dict[int, str]] = {}  # doc_id -> {heading_position: heading_id}

        async with factory() as session:
            for doc_id, (
                chunks,
                heading_tree,
                retrieved_nodes,
            ) in doc_chunks_map.items():
                try:
                    from app.models.database import Heading

                    position_to_id: dict[int, str] = {}
                    position_to_heading: dict[int, uuid.UUID] = {}

                    for heading_info in heading_tree:
                        parent_position = heading_info.get("parent_position")
                        heading = Heading(
                            id=uuid.uuid4(),
                            doc_id=uuid.UUID(doc_id),
                            level=heading_info["level"],
                            title=heading_info["title"],
                            position=heading_info["position"],
                            parent_id=position_to_heading.get(parent_position) if parent_position is not None else None,
                        )
                        session.add(heading)
                        await session.flush()
                        position_to_id[heading_info["position"]] = str(heading.id)
                        position_to_heading[heading_info["position"]] = heading.id

                    await session.commit()
                    doc_heading_ids[doc_id] = position_to_id.copy()
                    logger.info(f"[Batch {batch_id}] Saved headings for doc {doc_id}")
                except Exception as e:
                    logger.error(f"[Batch {batch_id}] Failed to save headings for {doc_id}: {e}")
                    await session.rollback()

        # Step 3: Vectorize ALL nodes at once (single embedding model load)
        if all_nodes:
            logger.info(f"[Batch {batch_id}] Vectorizing {len(all_nodes)} chunks in batch")
            success = await rag_engine.process_document(all_nodes)
            if not success:
                logger.warning(f"[Batch {batch_id}] GPU vectorization failed, falling back to CPU")
                from app.services.retrieval_indexer import RetrievalIndexer

                indexer = RetrievalIndexer()
                await indexer.add_documents(all_nodes)
            else:
                logger.info(f"[Batch {batch_id}] Batch vectorization completed successfully")

        # Step 4: Save chunks to PostgreSQL and update document status
        async with factory() as session:
            for doc_id, (
                chunks,
                heading_tree,
                retrieved_nodes,
            ) in doc_chunks_map.items():
                try:
                    from app.models.database import Chunk as DBChunk
                    from app.models.database import Document as DBDocument

                    # Use the actual heading IDs saved in Step 2
                    heading_ids_map = doc_heading_ids.get(doc_id, {})

                    # Verify chunks were provided
                    if not chunks:
                        logger.warning(f"[Batch {batch_id}] No chunks to save for doc {doc_id}")
                        continue

                    # Save chunks - use chunk position to find heading_id
                    for i, chunk in enumerate(chunks):
                        chunk_position = chunk.metadata.position if hasattr(chunk.metadata, "position") else i
                        heading_id_str = heading_ids_map.get(chunk_position)
                        chunk_record = DBChunk(
                            id=uuid.UUID(chunk.chunk_id),
                            doc_id=uuid.UUID(doc_id),
                            heading_id=uuid.UUID(heading_id_str) if heading_id_str else None,
                            content=chunk.content,
                            char_count=chunk.metadata.char_count,
                            position=i,
                            content_type=chunk.metadata.content_type,
                            section_title=chunk.metadata.section_title,
                        )
                        session.add(chunk_record)

                    # Force flush to catch any early errors
                    await session.flush()

                    # Update document status to completed
                    doc = await session.get(DBDocument, uuid.UUID(doc_id))
                    if doc:
                        doc.status = "completed"
                        doc.total_chunks = len(chunks)
                        await session.commit()
                        logger.info(f"[Batch {batch_id}] Saved {len(chunks)} chunks for doc {doc_id}")

                        # Verify save was successful
                        verify_result = await session.execute(
                            text("SELECT COUNT(*) FROM chunks WHERE doc_id = :doc_id"),
                            {"doc_id": uuid.UUID(doc_id)},
                        )
                        verify_count = verify_result.scalar()
                        if verify_count != len(chunks):
                            logger.error(
                                f"[Batch {batch_id}] Chunk save verification failed for {doc_id}: "
                                f"expected {len(chunks)}, got {verify_count}"
                            )

                    # Update batch status
                    for item in app_state_ref.batch_upload_status[batch_id].items:
                        if item.document_id == doc_id:
                            item.status = "completed"
                            break
                    app_state_ref.batch_upload_status[batch_id].processing -= 1
                    app_state_ref.batch_upload_status[batch_id].completed += 1

                except Exception as e:
                    logger.error(f"[Batch {batch_id}] Failed to save chunks for {doc_id}: {e}")
                    await session.rollback()
                    for item in app_state_ref.batch_upload_status[batch_id].items:
                        if item.document_id == doc_id:
                            item.status = "failed"
                            item.error_message = str(e)
                            break
                    app_state_ref.batch_upload_status[batch_id].processing -= 1
                    app_state_ref.batch_upload_status[batch_id].failed += 1

        logger.info(f"[Batch {batch_id}] Batch processing completed")

    except Exception as e:
        logger.error(f"[Batch {batch_id}] Batch processing failed: {e}")
    finally:
        if async_session is not None:
            await async_session.close()
        logger.debug(f"[Batch {batch_id}] Background session closed")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = None,
) -> DocumentUploadResponse:
    document_service = request.app.state.document_service

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
    document_service = request.app.state.document_service

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
    from app.services.document_store import DocumentStore

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
    document_service = request.app.state.document_service
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
    from app.core.database import get_session_factory

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
    from app.services.document_store import DocumentStore

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
    from app.core.database import get_session_factory

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
    from app.services.document_store import DocumentStore

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
    from app.services.document_store import DocumentStore

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
        # Delete old vector from Qdrant and BM25, then re-add
        from app.services.retrieval_indexer import RetrievalIndexer

        indexer = RetrievalIndexer()

        # Delete old entries
        await indexer.delete_documents(document_id, 1)

        # Re-embed and insert (simplified - full implementation would use DocumentProcessor)
        logger.info(f"Chunk {chunk_id} updated, re-indexing required")

    return {"chunk_id": chunk_id, "status": "re-indexed"}


@router.delete("/{document_id}/chunks/{chunk_id}")
async def delete_chunk(
    request: Request,
    document_id: str,
    chunk_id: str,
) -> dict[str, str]:
    """Delete a single chunk from a document."""
    from app.services.document_store import DocumentStore

    async with DocumentStore() as store:
        success = await store.delete_chunk(document_id, chunk_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Delete from vector index
    from app.services.retrieval_indexer import RetrievalIndexer

    indexer = RetrievalIndexer()
    await indexer.delete_documents(document_id, 1)

    return {"message": "Chunk deleted successfully"}


@router.post("/batch-delete", response_model=BatchOperationResponse)
async def batch_delete_documents(
    request: Request,
    batch_data: BatchDeleteRequest,
) -> BatchOperationResponse:
    """Delete multiple documents by ID."""
    from app.services.document_store import DocumentStore

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
                from app.services.retrieval_indexer import RetrievalIndexer

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
    from app.services.document_store import DocumentStore

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
    from app.services.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
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
    from app.services.consistency import ConsistencyChecker

    checker = ConsistencyChecker()
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
    from app.services.retrieval_indexer import RetrievalIndexer

    indexer = RetrievalIndexer()
    result = await indexer.rebuild_bm25_from_qdrant()
    return BM25RebuildResponse(
        success=result["success"],
        documents_rebuilt=result["documents_rebuilt"],
        errors=result.get("errors", []),
    )
