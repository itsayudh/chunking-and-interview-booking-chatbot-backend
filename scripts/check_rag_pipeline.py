"""Real end-to-end check of the RAG pipeline against the ingested documents."""

import asyncio

from google import genai
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings
from app.memory.redis_memory import ChatMessage
from app.rag.embeddings import EmbeddingService
from app.rag.llm import LLMService
from app.rag.pipeline import RagPipeline
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore

CASES: list[tuple[str, str, list[ChatMessage]]] = [
    ("Single question", "How long do I have to ask for a refund?", []),
    (
        "Follow-up (needs the history)",
        "And what about digital products?",
        [
            ChatMessage(role="user", content="How long do I have to ask for a refund?"),
            ChatMessage(role="assistant", content="Refunds can be requested within 30 days."),
        ],
    ),
    ("Unrelated question (should refuse)", "Who won the football world cup in 2018?", []),
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
    llm = LLMService(gemini, settings.gemini_llm_model, settings.llm_max_retries)
    pipeline = RagPipeline(
        llm=llm,
        rewriter=QueryRewriter(llm),
        retriever=Retriever(
            embedder,
            store,
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        ),
        temperature=settings.llm_temperature,
    )

    print(f"Minimum score in use: {settings.retrieval_min_score}")
    try:
        for label, question, history in CASES:
            print(f"\n=== {label} ===")
            print("Question:", question)

            result = await pipeline.answer(question, history)
            print("Standalone question:", result.standalone_question)

            query_vector = await embedder.embed_query(result.standalone_question)
            raw = await store.search(query_vector, top_k=3)
            print("Raw top scores (no cutoff):", [round(hit.score, 3) for hit in raw])

            print("Answered:", result.answered)
            print("Answer:", result.answer)
            for source in result.sources:
                print(
                    f"  source: {source.filename} part {source.chunk_index + 1} "
                    f"({source.score:.3f})"
                )
    finally:
        await qdrant.close()
        await gemini.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())