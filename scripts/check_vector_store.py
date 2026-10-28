"""End-to-end check: chunk -> embed (Gemini) -> store/search/delete (Qdrant Cloud)."""

import asyncio
import uuid

from google import genai
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.models.enums import ChunkingStrategy
from app.rag.chunking.factory import create_chunker
from app.rag.embeddings import EmbeddingService
from app.rag.vector_store import VectorStore

SCRATCH_COLLECTION = "scratch_check_collection"

REFUND_TEXT = (
    "Customers can request a refund within 30 days of purchase. "
    "To start a refund, contact support and provide the order number. "
    "Refunds are returned to the original payment method within five business days. "
    "Digital products are refundable only if they have not been downloaded."
)
BIOLOGY_TEXT = (
    "The mitochondria is the organelle that produces most of the cell's energy. "
    "It converts nutrients into a molecule called ATP. "
    "Cells with high energy needs, such as muscle cells, contain many mitochondria. "
    "Mitochondria also carry a small amount of their own DNA."
)


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
    store = VectorStore(qdrant, SCRATCH_COLLECTION, settings.embedding_dimension)
    chunker = create_chunker(ChunkingStrategy.RECURSIVE, chunk_size=200, chunk_overlap=30)

    try:
        await store.ensure_collection()
        print("Scratch collection ready")

        ids: dict[str, uuid.UUID] = {}
        for name, text in [("refunds.txt", REFUND_TEXT), ("biology.txt", BIOLOGY_TEXT)]:
            chunks = chunker.split(text)
            vectors = await embedder.embed_documents([c.text for c in chunks])
            ids[name] = uuid.uuid4()
            written = await store.upsert_chunks(
                document_id=ids[name],
                filename=name,
                strategy=ChunkingStrategy.RECURSIVE,
                chunks=chunks,
                vectors=vectors,
            )
            print(f"Stored {written} chunks for {name}")

        query = "How long do I have to ask for my money back?"
        query_vector = await embedder.embed_query(query)

        print(f"\nQuery: {query}")
        for hit in await store.search(query_vector, top_k=3):
            print(f"  {hit.score:.4f}  [{hit.filename} #{hit.chunk_index}]  {hit.text[:60]}")

        only_biology = await store.search(query_vector, top_k=3, document_ids=[ids["biology.txt"]])
        print("\nFiltered to biology.txt only:", {hit.filename for hit in only_biology})

        await store.delete_document(ids["refunds.txt"])
        remaining = await store.count_document_chunks(ids["refunds.txt"])
        print("Chunks left for refunds.txt after delete:", remaining)
    finally:
        await qdrant.delete_collection(SCRATCH_COLLECTION)
        print("\nScratch collection deleted")
        await qdrant.close()
        await gemini.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())