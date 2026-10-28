"""Unit tests for IngestionService (fake DB and embedder, in-memory Qdrant)."""

import uuid

import pytest

from app.core.exceptions import (
    DocumentNotFoundError,
    EmbeddingError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidRequestError,
    UnsupportedFileTypeError,
)
from app.models.enums import ChunkingStrategy, DocumentStatus
from app.rag.vector_store import VectorStore
from app.services.ingestion_service import IngestionService
from tests.fakes import FakeDocumentRepository, FakeEmbedder

TEXT = ("Refunds are processed within five business days. " * 8).encode()


async def test_successful_ingestion(
    ingestion_service: IngestionService,
    repository: FakeDocumentRepository,
    vector_store: VectorStore,
) -> None:
    document = await ingestion_service.ingest(
        filename="policy.txt", content=TEXT, strategy=ChunkingStrategy.RECURSIVE
    )

    assert document.status is DocumentStatus.COMPLETED
    assert document.chunk_count > 1
    assert document.chunk_size == 200
    assert document.chunk_overlap == 20
    assert await vector_store.count_document_chunks(document.id) == document.chunk_count
    assert len(repository.documents) == 1


async def test_strategy_and_sizes_are_stored(ingestion_service: IngestionService) -> None:
    document = await ingestion_service.ingest(
        filename="policy.txt",
        content=TEXT,
        strategy=ChunkingStrategy.FIXED,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert document.chunking_strategy is ChunkingStrategy.FIXED
    assert (document.chunk_size, document.chunk_overlap) == (100, 10)


async def test_directory_part_of_filename_is_removed(
    ingestion_service: IngestionService,
) -> None:
    document = await ingestion_service.ingest(
        filename="C:\\Users\\me\\docs\\policy.txt", content=TEXT, strategy=ChunkingStrategy.FIXED
    )
    assert document.filename == "policy.txt"


@pytest.mark.parametrize(
    ("filename", "content", "error"),
    [
        ("image.png", b"data", UnsupportedFileTypeError),
        ("big.txt", b"x" * 1001, FileTooLargeError),
        ("empty.txt", b"   \n  ", EmptyDocumentError),
    ],
)
async def test_bad_uploads_are_rejected_without_saving(
    ingestion_service: IngestionService,
    repository: FakeDocumentRepository,
    filename: str,
    content: bytes,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        await ingestion_service.ingest(
            filename=filename, content=content, strategy=ChunkingStrategy.RECURSIVE
        )
    assert repository.documents == {}


async def test_invalid_chunk_settings_are_rejected_without_saving(
    ingestion_service: IngestionService, repository: FakeDocumentRepository
) -> None:
    with pytest.raises(InvalidRequestError):
        await ingestion_service.ingest(
            filename="a.txt",
            content=TEXT,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=50,
            chunk_overlap=60,
        )
    assert repository.documents == {}


async def test_embedding_failure_marks_document_failed_and_cleans_up(
    ingestion_service: IngestionService,
    repository: FakeDocumentRepository,
    embedder: FakeEmbedder,
    vector_store: VectorStore,
) -> None:
    embedder.fail_with = EmbeddingError("quota exceeded")

    with pytest.raises(EmbeddingError):
        await ingestion_service.ingest(
            filename="a.txt", content=TEXT, strategy=ChunkingStrategy.RECURSIVE
        )

    (document,) = repository.documents.values()
    assert document.status is DocumentStatus.FAILED
    assert document.error_message == "quota exceeded"
    assert await vector_store.count_document_chunks(document.id) == 0


async def test_delete_removes_row_and_vectors(
    ingestion_service: IngestionService,
    repository: FakeDocumentRepository,
    vector_store: VectorStore,
) -> None:
    document = await ingestion_service.ingest(
        filename="a.txt", content=TEXT, strategy=ChunkingStrategy.RECURSIVE
    )

    await ingestion_service.delete_document(document.id)

    assert repository.documents == {}
    assert await vector_store.count_document_chunks(document.id) == 0


async def test_unknown_document_raises_not_found(ingestion_service: IngestionService) -> None:
    with pytest.raises(DocumentNotFoundError):
        await ingestion_service.get_document(uuid.uuid4())
    with pytest.raises(DocumentNotFoundError):
        await ingestion_service.delete_document(uuid.uuid4())