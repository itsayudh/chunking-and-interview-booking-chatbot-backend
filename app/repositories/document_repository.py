"""Database access for documents."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.enums import ChunkingStrategy, DocumentStatus, FileType


class DocumentRepository:
    """CRUD operations for the documents table. Each method commits its own transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        filename: str,
        file_type: FileType,
        file_size_bytes: int,
        chunking_strategy: ChunkingStrategy,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Document:
        """Insert a new document row with status 'processing'."""
        document = Document(
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            status=DocumentStatus.PROCESSING,
        )
        self._session.add(document)
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def get(self, document_id: uuid.UUID) -> Document | None:
        """Return one document or None if it does not exist."""
        return await self._session.get(Document, document_id)

    async def list_all(self, *, limit: int = 50, offset: int = 0) -> Sequence[Document]:
        """Return documents, newest first."""
        statement = (
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def mark_completed(self, document: Document, chunk_count: int) -> Document:
        """Mark ingestion as finished and store the number of chunks."""
        document.chunk_count = chunk_count
        document.status = DocumentStatus.COMPLETED
        document.error_message = None
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def mark_failed(self, document: Document, error_message: str) -> Document:
        """Mark ingestion as failed and store the reason."""
        document.status = DocumentStatus.FAILED
        document.error_message = error_message
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def delete(self, document: Document) -> None:
        """Delete a document row."""
        await self._session.delete(document)
        await self._session.commit()