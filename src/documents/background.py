"""Background document processing tasks.

Extracted from api.py to eliminate deferred (function-body) imports.
All imports here are top-level — no cyclic dependency with agent layer
because ``src.agent`` never imports from ``src.documents``.
"""

import asyncio
import uuid
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from src.agent.rag_agent import RAGAgent
from src.common.database import get_session_factory
from src.documents import DocumentProcessor, DocumentService, RetrievalIndexer
from src.documents.models import Chunk as DBChunk
from src.documents.models import Document as DBDocument
from src.documents.models import Heading


async def process_document_background(doc_id: str, file_path: str, title: str | None = None):
    """后台处理文档（使用独立的 DocumentService 实例）"""

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

    factory = get_session_factory()
    async_session = factory()
    processor = DocumentProcessor()
    rag_engine = RAGAgent()

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
        doc_heading_ids: dict[str, dict[int, str]] = {}  # doc_id -> {heading_position: heading_id}

        async with factory() as session:
            for doc_id, (chunks, heading_tree, retrieved_nodes) in doc_chunks_map.items():
                try:
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
                indexer = RetrievalIndexer()
                await indexer.add_documents(all_nodes)
            else:
                logger.info(f"[Batch {batch_id}] Batch vectorization completed successfully")

        # Step 4: Save chunks to PostgreSQL and update document status
        async with factory() as session:
            for doc_id, (chunks, heading_tree, retrieved_nodes) in doc_chunks_map.items():
                try:
                    # Use the actual heading IDs saved in Step 2
                    heading_ids_map = doc_heading_ids.get(doc_id, {})

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
                            text("SELECT COUNT(*) FROM documents.chunks WHERE doc_id = :doc_id"),
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
