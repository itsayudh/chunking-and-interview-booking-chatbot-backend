"""Qdrant vector store wrapper: collection setup, upsert, search and delete."""

import logging
import uuid
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.exceptions import VectorStoreError
from app.models.enums import ChunkingStrategy
from app.rag.chunking.base import Chunk

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 50  # 3072-dim vectors are large, so keep requests moderate
_DOCUMENT_ID_FIELD = "document_id"


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk with its similarity score."""

    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str
    score: float


def _point_id(document_id: uuid.UUID, chunk_index: int) -> str:
    """Stable ID for a chunk, so re-ingesting a document overwrites instead of duplicating."""
    return str(uuid.uuid5(document_id, str(chunk_index)))


def _document_filter(document_id: uuid.UUID) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key=_DOCUMENT_ID_FIELD,
                match=models.MatchValue(value=str(document_id)),
            )
        ]
    )


class VectorStore:
    """Stores and searches chunk vectors in one Qdrant collection (cosine distance)."""

    def __init__(self, client: AsyncQdrantClient, collection: str, dimension: int) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    async def _guard[T](self, action: str, awaitable: Awaitable[T]) -> T:
        """Await a Qdrant call and convert client errors into VectorStoreError."""
        try:
            return await awaitable
        except (UnexpectedResponse, ResponseHandlingException) as exc:
            logger.exception("Qdrant failed while trying to %s", action)
            raise VectorStoreError(f"Vector database error while trying to {action}.") from exc

    async def ensure_collection(self) -> None:
        """Create the collection and its payload index if missing; verify the vector size."""
        exists = await self._guard(
            "check the collection", self._client.collection_exists(self._collection)
        )
        if not exists:
            await self._guard(
                "create the collection",
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._dimension, distance=models.Distance.COSINE
                    ),
                ),
            )
            await self._guard(
                "create the payload index",
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=_DOCUMENT_ID_FIELD,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self._collection)
            return

        info = await self._guard(
            "read the collection", self._client.get_collection(self._collection)
        )
        vectors = info.config.params.vectors
        size = vectors.size if isinstance(vectors, models.VectorParams) else None
        if size != self._dimension:
            raise VectorStoreError(
                f"Collection '{self._collection}' has vector size {size}, expected "
                f"{self._dimension}. Use a new collection name or recreate the collection."
            )

    async def upsert_chunks(
        self,
        *,
        document_id: uuid.UUID,
        filename: str,
        strategy: ChunkingStrategy,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Store chunks with their vectors. Returns the number of points written."""
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks but {len(vectors)} vectors")
        for vector in vectors:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"Vector has dimension {len(vector)}, expected {self._dimension}"
                )

        points = [
            models.PointStruct(
                id=_point_id(document_id, chunk.index),
                vector=list(vector),
                payload={
                    _DOCUMENT_ID_FIELD: str(document_id),
                    "filename": filename,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "strategy": strategy.value,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[start : start + _UPSERT_BATCH_SIZE]
            await self._guard(
                "store chunk vectors",
                self._client.upsert(collection_name=self._collection, points=batch, wait=True),
            )
        return len(points)

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """Return the most similar chunks, optionally limited to certain documents."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_filter: models.Filter | None = None
        if document_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=_DOCUMENT_ID_FIELD,
                        match=models.MatchAny(any=[str(doc_id) for doc_id in document_ids]),
                    )
                ]
            )

        response = await self._guard(
            "search the collection",
            self._client.query_points(
                collection_name=self._collection,
                query=list(query_vector),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            ),
        )

        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    document_id=uuid.UUID(str(payload[_DOCUMENT_ID_FIELD])),
                    filename=str(payload["filename"]),
                    chunk_index=int(payload["chunk_index"]),
                    text=str(payload["text"]),
                    score=point.score,
                )
            )
        return hits

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk belonging to a document."""
        await self._guard(
            "delete document vectors",
            self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=_document_filter(document_id)),
                wait=True,
            ),
        )

    async def count_document_chunks(self, document_id: uuid.UUID) -> int:
        """Return how many chunks are stored for a document."""
        result = await self._guard(
            "count document chunks",
            self._client.count(
                collection_name=self._collection,
                count_filter=_document_filter(document_id),
                exact=True,
            ),
        )
        return result.count