"""Document ingestion workflow: validate, extract, chunk, embed, store."""

import asyncio
import logging
import uuid
from collections.abc import Sequence
from pathlib import PureWindowsPath

from app.core.exceptions import (
    AppError,
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidRequestError,
)
from app.models.document import Document
from app.models.enums import ChunkingStrategy, FileType
from app.rag.chunking.base import BaseChunker, Chunk
from app.rag.chunking.factory import create_chunker
from app.rag.embeddings import EmbeddingService
from app.rag.loaders import detect_file_type, extract_text
from app.rag.vector_store import VectorStore
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

_MAX_FILENAME_LENGTH = 255
_MAX_ERROR_LENGTH = 500


class IngestionService:
    """Coordinates the repository, chunkers, embedder and vector store."""

    def __init__(
        self,
        *,
        repository: DocumentRepository,
        embedder: EmbeddingService,
        vector_store: VectorStore,
        default_chunk_size: int,
        default_chunk_overlap: int,
        max_upload_bytes: int,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_chunk_size = default_chunk_size
        self._default_chunk_overlap = default_chunk_overlap
        self._max_upload_bytes = max_upload_bytes

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted upload size in bytes."""
        return self._max_upload_bytes

    async def ingest(
        self,
        *,
        filename: str,
        content: bytes,
        strategy: ChunkingStrategy,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Document:
        """Ingest one uploaded file and return its stored metadata.

        Raises:
            UnsupportedFileTypeError, FileTooLargeError, InvalidRequestError,
            DocumentParsingError, EmptyDocumentError: the upload is unusable (nothing saved).
            EmbeddingError, VectorStoreError: processing failed after the row was created;
                the document is marked 'failed' and partial vectors are removed.
        """
        safe_name = PureWindowsPath(filename).name[-_MAX_FILENAME_LENGTH:]
        file_type = detect_file_type(safe_name)

        if len(content) > self._max_upload_bytes:
            limit_mb = self._max_upload_bytes // (1024 * 1024)
            raise FileTooLargeError(f"File exceeds the {limit_mb} MB upload limit.")

        size = self._default_chunk_size if chunk_size is None else chunk_size
        overlap = self._default_chunk_overlap if chunk_overlap is None else chunk_overlap
        chunker = self._build_chunker(strategy, size, overlap)

        # pypdf is blocking, so run extraction and chunking off the event loop.
        chunks = await asyncio.to_thread(self._extract_and_chunk, content, file_type, chunker)

        document = await self._repository.create(
            filename=safe_name,
            file_type=file_type,
            file_size_bytes=len(content),
            chunking_strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap,
        )

        try:
            vectors = await self._embedder.embed_documents([chunk.text for chunk in chunks])
            await self._vector_store.upsert_chunks(
                document_id=document.id,
                filename=safe_name,
                strategy=strategy,
                chunks=chunks,
                vectors=vectors,
            )
        except Exception as exc:
            await self._record_failure(document, exc)
            raise

        return await self._repository.mark_completed(document, chunk_count=len(chunks))

    async def get_document(self, document_id: uuid.UUID) -> Document:
        """Return a document or raise DocumentNotFoundError."""
        document = await self._repository.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} was not found.")
        return document

    async def list_documents(self, *, limit: int, offset: int) -> Sequence[Document]:
        """Return documents, newest first."""
        return await self._repository.list_all(limit=limit, offset=offset)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete a document's vectors first, then its metadata row."""
        document = await self.get_document(document_id)
        await self._vector_store.delete_document(document.id)
        await self._repository.delete(document)

    @staticmethod
    def _build_chunker(strategy: ChunkingStrategy, size: int, overlap: int) -> BaseChunker:
        try:
            return create_chunker(strategy, chunk_size=size, chunk_overlap=overlap)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc

    @staticmethod
    def _extract_and_chunk(
        content: bytes, file_type: FileType, chunker: BaseChunker
    ) -> list[Chunk]:
        return chunker.split(extract_text(content, file_type))

    async def _record_failure(self, document: Document, exc: Exception) -> None:
        """Remove partial vectors and mark the document as failed."""
        logger.exception("Ingestion failed for document %s", document.id)
        try:
            await self._vector_store.delete_document(document.id)
        except Exception:
            logger.exception("Could not clean up vectors for document %s", document.id)
        message = exc.message if isinstance(exc, AppError) else "Unexpected error during ingestion."
        await self._repository.mark_failed(document, message[:_MAX_ERROR_LENGTH])