"""Embedding service backed by the Gemini API (gemini-embedding-001)."""

import asyncio
import logging
from collections.abc import Sequence

from google import genai
from google.genai import errors, types

from app.core.exceptions import EmbeddingError, EmbeddingQuotaError
from app.rag.retry import describe_quota_error, retry_delay

logger = logging.getLogger(__name__)

_API_MAX_BATCH = 100  # hard limit enforced by the Gemini batch endpoint


class EmbeddingService:
    """Turns text into vectors, with batching, retries and output validation."""

    def __init__(
        self,
        client: genai.Client,
        model: str,
        dimension: int,
        batch_size: int = 50,
        max_retries: int = 5,
    ) -> None:
        if not 1 <= batch_size <= _API_MAX_BATCH:
            raise ValueError(f"batch_size must be between 1 and {_API_MAX_BATCH}")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._client = client
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_retries = max_retries

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunks for storage. Returns one vector per input, in order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch, "RETRIEVAL_DOCUMENT"))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        query = text.strip()
        if not query:
            raise ValueError("Query text must not be empty")
        vectors = await self._embed_batch([query], "RETRIEVAL_QUERY")
        return vectors[0]

    async def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        """Embed one batch, retrying temporary failures and failing fast on exhausted quotas."""
        config = types.EmbedContentConfig(task_type=task_type)
        attempt = 0
        while True:
            try:
                result = await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=[text for text in batch],
                    config=config,
                )
                break
            except errors.APIError as exc:
                delay = retry_delay(exc, attempt)
                if delay is None or attempt >= self._max_retries:
                    raise self._to_embedding_error(exc) from exc
                logger.warning(
                    "Embedding call failed with %s; retry %d/%d in %.1fs",
                    exc.code,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

        return self._validated_vectors(result, expected_count=len(batch))

    @staticmethod
    def _to_embedding_error(exc: errors.APIError) -> EmbeddingError:
        if exc.code == 429:
            message, retry_after = describe_quota_error("Embedding API", exc)
            return EmbeddingQuotaError(message, retry_after_seconds=retry_after)
        return EmbeddingError(f"Embedding API request failed (HTTP {exc.code}).")

    def _validated_vectors(
        self, result: types.EmbedContentResponse, expected_count: int
    ) -> list[list[float]]:
        embeddings = result.embeddings or []
        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} embeddings but received {len(embeddings)}."
            )
        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = list(embedding.values or [])
            if len(values) != self._dimension:
                raise EmbeddingError(
                    f"Expected vectors of dimension {self._dimension} but received {len(values)}."
                )
            vectors.append(values)
        return vectors