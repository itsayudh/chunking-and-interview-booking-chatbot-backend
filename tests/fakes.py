"""In-memory test doubles shared by unit and integration tests."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from app.models.document import Document
from app.models.enums import ChunkingStrategy, DocumentStatus, FileType

DIMENSION = 4


class FakeDocumentRepository:
    """Same interface as DocumentRepository, backed by a dict."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}

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
        document = Document(
            id=uuid.uuid4(),
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_count=0,
            status=DocumentStatus.PROCESSING,
            error_message=None,
            created_at=datetime.now(UTC),
        )
        self.documents[document.id] = document
        return document

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return self.documents.get(document_id)

    async def list_all(self, *, limit: int = 50, offset: int = 0) -> Sequence[Document]:
        ordered = sorted(self.documents.values(), key=lambda d: d.created_at, reverse=True)
        return ordered[offset : offset + limit]

    async def mark_completed(self, document: Document, chunk_count: int) -> Document:
        document.chunk_count = chunk_count
        document.status = DocumentStatus.COMPLETED
        document.error_message = None
        return document

    async def mark_failed(self, document: Document, error_message: str) -> Document:
        document.status = DocumentStatus.FAILED
        document.error_message = error_message
        return document

    async def delete(self, document: Document) -> None:
        self.documents.pop(document.id, None)


class FakeEmbedder:
    """Deterministic embedder that needs no network. Can be told to fail."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.calls = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        vector = [0.0] * DIMENSION
        vector[len(text) % DIMENSION] = 1.0
        return vector