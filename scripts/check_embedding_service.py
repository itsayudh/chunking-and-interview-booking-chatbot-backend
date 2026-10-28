"""Real-API check of EmbeddingService: ranking of three documents against one query."""

import asyncio

from google import genai

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingService

QUERY = "How do I get a refund for my order?"
DOCUMENTS = [
    "Customers can request a refund within 30 days by contacting support with the order number.",
    "The mitochondria is the organelle that produces most of the cell's energy.",
    "Our office is closed on public holidays and reopens the next working day.",
]


async def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    service = EmbeddingService(
        client=client,
        model=settings.gemini_embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )
    try:
        doc_vectors = await service.embed_documents(DOCUMENTS)
        query_vector = await service.embed_query(QUERY)
    finally:
        await client.aio.aclose()

    # Vectors are normalized, so the dot product equals cosine similarity.
    scores = [sum(q * d for q, d in zip(query_vector, vec, strict=True)) for vec in doc_vectors]
    print("Query:", QUERY)
    for score, text in sorted(zip(scores, DOCUMENTS, strict=True), reverse=True):
        print(f"  {score:.4f}  {text[:70]}")
    best = max(range(len(scores)), key=scores.__getitem__)
    print("Best match is the refund text:", best == 0)


if __name__ == "__main__":
    asyncio.run(main())