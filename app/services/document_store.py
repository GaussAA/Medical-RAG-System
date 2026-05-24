import uuid
from datetime import datetime, UTC
from pathlib import Path
from loguru import logger
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_factory
from app.models.database import Chunk as DBChunk, Document, Heading


class DocumentStore:
    """Handles PostgreSQL CRUD operations for documents and chunks."""

    def __init__(self, async_session: AsyncSession | None = None):
        self.async_session = async_session
        self._owns_session = False  # True if we created the session ourselves

    async def _ensure_session(self) -> AsyncSession:
        """Ensure a valid database session."""
        if self.async_session is None:
            factory = get_session_factory()
            self.async_session = factory()
            self._owns_session = True
        return self.async_session

    async def close_session(self) -> None:
        """Close session if we own it."""
        if self._owns_session and self.async_session is not None:
            await self.async_session.close()
            self.async_session = None
            self._owns_session = False

    async def create_document(
        self,
        doc_id: str,
        file_path: str,
        title: str,
        file_md5: str | None = None,
    ) -> Document:
        """Create a new document record in PostgreSQL."""
        session = await self._ensure_session()
        doc = Document(
            id=uuid.UUID(doc_id),
            title=title,
            file_path=file_path,
            file_name=Path(file_path).name,
            file_type=Path(file_path).suffix[1:],
            file_md5=file_md5,
            status="processing",
        )
        session.add(doc)
        await session.commit()
        return doc

    async def get_document(self, doc_id: str) -> Document | None:
        """Get document by ID."""
        session = await self._ensure_session()
        return await session.get(Document, uuid.UUID(doc_id))

    async def find_by_md5(self, file_md5: str) -> Document | None:
        """Find existing document by MD5 hash."""
        session = await self._ensure_session()
        try:
            result = await session.execute(select(Document).where(Document.file_md5 == file_md5))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.warning(f"Failed to find document by MD5: {e}")
            return None

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        total_chunks: int | None = None,
        total_pages: int | None = None,
    ) -> None:
        """Update document status and metadata."""
        session = await self._ensure_session()
        doc = await session.get(Document, uuid.UUID(doc_id))
        if doc:
            doc.status = status
            if total_chunks is not None:
                doc.total_chunks = total_chunks
            if total_pages is not None:
                doc.total_pages = total_pages
            await session.commit()

    async def save_headings(self, doc_id: str, heading_tree: list[dict]) -> dict[int, str]:
        """
        Save heading tree to PostgreSQL and return mapping of position to heading_id.

        Returns:
            dict mapping position -> heading_id for chunk association
        """
        session = await self._ensure_session()
        position_to_id = {}
        position_to_heading = {}  # Maps position to Heading record for FK resolution

        try:
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
            logger.info(f"Persisted {len(heading_tree)} headings to PostgreSQL")
            return position_to_id
        except Exception as e:
            logger.error(f"Failed to persist headings to PostgreSQL: {e}")
            await session.rollback()
            raise

    async def save_chunks(self, doc_id: str, chunks: list, heading_ids: dict[int, str] | None = None) -> None:
        """Persist chunks to PostgreSQL with optional heading associations."""
        session = await self._ensure_session()
        try:
            for i, chunk in enumerate(chunks):
                chunk_record = DBChunk(
                    id=uuid.UUID(chunk.chunk_id),
                    doc_id=uuid.UUID(doc_id),
                    heading_id=uuid.UUID(heading_ids.get(i)) if heading_ids and heading_ids.get(i) else None,
                    content=chunk.content,
                    char_count=chunk.metadata.char_count,
                    position=i,
                    content_type=chunk.metadata.content_type,
                    section_title=chunk.metadata.section_title,
                )
                session.add(chunk_record)
            await session.commit()
            logger.info(f"Persisted {len(chunks)} chunks to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to persist chunks to PostgreSQL: {e}")
            await session.rollback()
            raise

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document and its chunks from PostgreSQL."""
        session = await self._ensure_session()
        try:
            # Delete chunks first
            await session.execute(delete(DBChunk).where(DBChunk.doc_id == uuid.UUID(doc_id)))
            # Delete document
            await session.execute(delete(Document).where(Document.id == uuid.UUID(doc_id)))
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete document from PostgreSQL: {e}")
            await session.rollback()
            return False

    async def list_documents(
        self,
        status: str | None = None,
        tags: list[str] | None = None,
        file_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        """List documents with filtering and pagination. Returns (documents, total_count)."""
        session = await self._ensure_session()

        query = select(Document)
        count_query = select(func.count(Document.id))

        # Apply filters
        if status:
            query = query.where(Document.status == status)
            count_query = count_query.where(Document.status == status)
        if file_type:
            query = query.where(Document.file_type == file_type)
            count_query = count_query.where(Document.file_type == file_type)
        if date_from:
            query = query.where(Document.created_at >= date_from)
            count_query = count_query.where(Document.created_at >= date_from)
        if date_to:
            query = query.where(Document.created_at <= date_to)
            count_query = count_query.where(Document.created_at <= date_to)

        # Tag filtering with array containment using && operator
        if tags:
            # Use && operator for array containment: tags && ARRAY['tag1', 'tag2']
            tag_list = [t.lower() for t in tags]
            query = query.where(Document.tags.op("&&")(tag_list))
            count_query = count_query.where(Document.tags.op("&&")(tag_list))

        # Get total count
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination and ordering
        query = query.order_by(Document.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await session.execute(query)
        return list(result.scalars()), total

    async def get_chunks(
        self,
        doc_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[DBChunk], int]:
        """Get chunks for a document with pagination. Returns (chunks, total_count)."""
        session = await self._ensure_session()

        # Verify document exists
        doc = await session.get(Document, uuid.UUID(doc_id))
        if not doc:
            return [], 0

        # Count total chunks
        count_query = select(func.count(DBChunk.id)).where(DBChunk.doc_id == uuid.UUID(doc_id))
        total_result = await session.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated chunks
        query = (
            select(DBChunk)
            .where(DBChunk.doc_id == uuid.UUID(doc_id))
            .order_by(DBChunk.position)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await session.execute(query)
        return list(result.scalars()), total

    async def update_chunk(
        self,
        doc_id: str,
        chunk_id: str,
        content: str | None = None,
        section_title: str | None = None,
    ) -> DBChunk | None:
        """Update a chunk's content and/or metadata."""
        session = await self._ensure_session()
        chunk = await session.get(DBChunk, uuid.UUID(chunk_id))
        if not chunk or str(chunk.doc_id) != doc_id:
            return None

        if content is not None:
            chunk.content = content
        if section_title is not None:
            chunk.section_title = section_title

        # Update document's updated_at
        doc = await session.get(Document, uuid.UUID(doc_id))
        if doc:
            doc.updated_at = datetime.now(UTC)

        await session.commit()
        return chunk

    async def delete_chunk(self, doc_id: str, chunk_id: str) -> bool:
        """Delete a single chunk and reorder remaining chunks."""
        session = await self._ensure_session()
        chunk = await session.get(DBChunk, uuid.UUID(chunk_id))
        if not chunk or str(chunk.doc_id) != doc_id:
            return False

        # Delete the chunk
        await session.execute(delete(DBChunk).where(DBChunk.id == uuid.UUID(chunk_id)))

        # Reorder remaining chunks
        remaining_chunks = (
            select(DBChunk).where(DBChunk.doc_id == uuid.UUID(doc_id)).order_by(DBChunk.position)
        )
        result = await session.execute(remaining_chunks)
        chunks_to_reorder = list(result.scalars())

        for i, c in enumerate(chunks_to_reorder):
            c.position = i

        # Update document total_chunks and updated_at
        doc = await session.get(Document, uuid.UUID(doc_id))
        if doc:
            doc.total_chunks = len(chunks_to_reorder)
            doc.updated_at = datetime.now(UTC)

        await session.commit()
        return True

    async def update_document_tags(
        self,
        doc_id: str,
        tags: list[str],
        operation: str = "add",
    ) -> Document | None:
        """Add or remove tags from a document."""
        session = await self._ensure_session()
        doc = await session.get(Document, uuid.UUID(doc_id))
        if not doc:
            return None

        current_tags = set(doc.tags or [])
        if operation == "add":
            current_tags.update(t.lower() for t in tags)
        else:
            current_tags.difference_update(t.lower() for t in tags)

        doc.tags = list(current_tags)
        doc.updated_at = datetime.now(UTC)
        await session.commit()
        return doc

    async def update_document(
        self,
        doc_id: str,
        status: str | None = None,
        tags: list[str] | None = None,
        operation: str = "add",
    ) -> Document | None:
        """Update document status and/or tags."""
        session = await self._ensure_session()
        doc = await session.get(Document, uuid.UUID(doc_id))
        if not doc:
            return None

        if status:
            doc.status = status
        if tags:
            current_tags = set(doc.tags or [])
            if operation == "add":
                current_tags.update(t.lower() for t in tags)
            else:
                current_tags.difference_update(t.lower() for t in tags)
            doc.tags = list(current_tags)

        doc.updated_at = datetime.now(UTC)
        await session.commit()
        return doc
