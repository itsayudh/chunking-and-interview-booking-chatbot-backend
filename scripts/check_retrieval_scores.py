"""Print raw retrieval scores (no cutoff) to help choose RETRIEVAL_MIN_SCORE."""

import asyncio

from google import genai
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

QUESTIONS: list[tuple[str, str]] = [
    ("relevant", "How long does shipping take?"),
    ("relevant", "Can I get a refund for a digital product?"),
    ("relevant", "Do international orders cost extra?"),
    ("borderline", "How can I contact support?"),
    ("borderline", "What payment methods do you accept?"),
    ("borderline", "Where is your office located?"),
    ("unrelated", "What is the capital of France?"),
    ("unrelated", "How do I bake a chocolate cake?"),
    ("unrelated", "Who won the football world cup in 2018?"),
    ("unrelated", "hello"),
    ("unrelated", "thanks, that was helpful"),
]


async def main() -> None:
    settings = get_settings()
    gemini = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        timeout=30,
    )
    embedder = EmbeddingService(
        client=gemini,
        model=settings.gemini_embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )
    store = VectorStore(qdrant, settings.qdrant_collection, settings.embedding_dimension)

    print(f"Current cutoff: {settings.retrieval_min_score}\n")
    print(f"{'type':<11}{'best':>7}  {'source':<14}question")
    try:
        for label, question in QUESTIONS:
            vector = await embedder.embed_query(question)
            hits = await store.search(vector, top_k=1)
            if hits:
                print(f"{label:<11}{hits[0].score:>7.3f}  {hits[0].filename:<14}{question}")
            else:
                print(f"{label:<11}{'none':>7}  {'':<14}{question}")
    finally:
        await qdrant.close()
        await gemini.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())