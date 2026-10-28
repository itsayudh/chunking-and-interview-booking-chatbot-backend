"""Unit tests for VectorStore, using Qdrant's in-process memory mode (no network)."""

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models

from app.core.exceptions import VectorStoreError
from app.models.enums import ChunkingStrategy
from app.rag.chunking.base import Chunk
from app.rag.vector_store import VectorStore

DIMENSION = 4
AXIS_X = [1.0, 0.0, 0.0, 0.0]
AXIS_Y = [0.0, 1.0, 0.0, 0.0]
AXIS_Z = [0.0, 0.0, 1.0, 0.0]
NEAR_X = [0.9, 0.1, 0.0, 0.0]


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncQdrantClient]:
    qdrant = AsyncQdrantClient(location=":memory:")
    yield qdrant
    await qdrant.close()


@pytest_asyncio.fixture
async def store(client: AsyncQdrantClient) -> VectorStore:
    vector_store = VectorStore(client, collection="test_chunks", dimension=DIMENSION)
    await vector_store.ensure_collection()
    return vector_store


def chunks(*texts: str) -> list[Chunk]:
    return [Chunk(index=i, text=text) for i, text in enumerate(texts)]


async def add_document(
    store: VectorStore, name: str, texts: list[str], vectors: list[list[float]]
) -> uuid.UUID:
    document_id = uuid.uuid4()
    await store.upsert_chunks(
        document_id=document_id,
        filename=name,
        strategy=ChunkingStrategy.RECURSIVE,
        chunks=chunks(*texts),
        vectors=vectors,
    )
    return document_id


async def test_ensure_collection_is_idempotent(client: AsyncQdrantClient) -> None:
    vector_store = VectorStore(client, collection="idem", dimension=DIMENSION)
    await vector_store.ensure_collection()
    await vector_store.ensure_collection()
    assert await client.collection_exists("idem")


async def test_dimension_mismatch_is_detected(client: AsyncQdrantClient) -> None:
    await client.create_collection(
        collection_name="wrong_size",
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    vector_store = VectorStore(client, collection="wrong_size", dimension=DIMENSION)
    with pytest.raises(VectorStoreError, match="vector size"):
        await vector_store.ensure_collection()


async def test_upsert_and_count(store: VectorStore) -> None:
    doc_id = await add_document(store, "a.txt", ["one", "two"], [AXIS_X, AXIS_Y])
    assert await store.count_document_chunks(doc_id) == 2


async def test_upsert_is_idempotent(store: VectorStore) -> None:
    doc_id = uuid.uuid4()
    for _ in range(2):
        await store.upsert_chunks(
            document_id=doc_id,
            filename="a.txt",
            strategy=ChunkingStrategy.FIXED,
            chunks=chunks("one", "two"),
            vectors=[AXIS_X, AXIS_Y],
        )
    assert await store.count_document_chunks(doc_id) == 2


async def test_search_returns_nearest_first_with_payload(store: VectorStore) -> None:
    doc_id = await add_document(
        store, "a.txt", ["about x", "about y", "about z"], [AXIS_X, AXIS_Y, AXIS_Z]
    )

    hits = await store.search(NEAR_X, top_k=2)

    assert [hit.text for hit in hits] == ["about x", "about y"]
    assert hits[0].score > hits[1].score
    assert hits[0].document_id == doc_id
    assert hits[0].filename == "a.txt"
    assert hits[0].chunk_index == 0


async def test_search_can_be_limited_to_documents(store: VectorStore) -> None:
    await add_document(store, "a.txt", ["a-x"], [AXIS_X])
    doc_b = await add_document(store, "b.txt", ["b-x"], [NEAR_X])

    hits = await store.search(AXIS_X, top_k=5, document_ids=[doc_b])

    assert [hit.filename for hit in hits] == ["b.txt"]


async def test_score_threshold_filters_weak_matches(store: VectorStore) -> None:
    await add_document(store, "a.txt", ["x", "z"], [AXIS_X, AXIS_Z])

    hits = await store.search(AXIS_X, top_k=5, score_threshold=0.5)

    assert [hit.text for hit in hits] == ["x"]


async def test_delete_removes_only_that_document(store: VectorStore) -> None:
    doc_a = await add_document(store, "a.txt", ["a1", "a2"], [AXIS_X, AXIS_Y])
    doc_b = await add_document(store, "b.txt", ["b1"], [AXIS_Z])

    await store.delete_document(doc_a)

    assert await store.count_document_chunks(doc_a) == 0
    assert await store.count_document_chunks(doc_b) == 1


async def test_chunk_vector_count_mismatch_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="chunks"):
        await store.upsert_chunks(
            document_id=uuid.uuid4(),
            filename="a.txt",
            strategy=ChunkingStrategy.FIXED,
            chunks=chunks("one", "two"),
            vectors=[AXIS_X],
        )


async def test_wrong_vector_dimension_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="dimension"):
        await store.upsert_chunks(
            document_id=uuid.uuid4(),
            filename="a.txt",
            strategy=ChunkingStrategy.FIXED,
            chunks=chunks("one"),
            vectors=[[1.0, 0.0]],
        )


async def test_invalid_top_k_raises(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        await store.search(AXIS_X, top_k=0)