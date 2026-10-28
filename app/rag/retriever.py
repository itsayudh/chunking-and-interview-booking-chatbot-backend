"""Finds the document chunks most relevant to a question."""

import uuid
from collections.abc import Sequence

from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import SearchHit, VectorStore


class Retriever:
    """Embeds a query and returns the best chunks above a minimum similarity score."""

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: VectorStore,
        *,
        top_k: int,
        min_score: float,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._min_score = min_score

    async def retrieve(
        self, query: str, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        """Return up to top_k chunks whose score is at least min_score, best first."""
        query_vector = await self._embedder.embed_query(query)
        return await self._vector_store.search(
            query_vector,
            top_k=self._top_k,
            document_ids=document_ids,
            score_threshold=self._min_score,
        )