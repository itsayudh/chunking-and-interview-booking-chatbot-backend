"""Shared pytest fixtures."""

from collections.abc import AsyncIterator
from typing import cast

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore
from app.repositories.document_repository import DocumentRepository
from app.services.ingestion_service import IngestionService
from tests.fakes import DIMENSION, FakeDocumentRepository, FakeEmbedder


@pytest_asyncio.fixture
async def vector_store() -> AsyncIterator[VectorStore]:
    client = AsyncQdrantClient(location=":memory:")
    store = VectorStore(client, collection="test_chunks", dimension=DIMENSION)
    await store.ensure_collection()
    yield store
    await client.close()


@pytest.fixture
def repository() -> FakeDocumentRepository:
    return FakeDocumentRepository()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def ingestion_service(
    repository: FakeDocumentRepository,
    embedder: FakeEmbedder,
    vector_store: VectorStore,
) -> IngestionService:
    return IngestionService(
        repository=cast(DocumentRepository, repository),
        embedder=cast(EmbeddingService, embedder),
        vector_store=vector_store,
        default_chunk_size=200,
        default_chunk_overlap=20,
        max_upload_bytes=1000,
    )