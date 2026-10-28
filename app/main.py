"""FastAPI application entry point."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.api.v1.endpoints import health
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import setup_logging
from app.db.session import create_engine, create_session_factory
from app.memory.redis_memory import RedisMemory
from app.rag.embeddings import EmbeddingService
from app.rag.llm import LLMService
from app.rag.pipeline import RagPipeline
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.booking_extractor import BookingExtractor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients and services at startup and close them at shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting %s in %s mode", settings.app_name, settings.environment)

    engine = create_engine(settings)
    gemini = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        timeout=30,
    )
    redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)

    embedder = EmbeddingService(
        client=gemini,
        model=settings.gemini_embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_retries=settings.embedding_max_retries,
    )
    vector_store = VectorStore(qdrant, settings.qdrant_collection, settings.embedding_dimension)
    llm = LLMService(gemini, settings.gemini_llm_model, settings.llm_max_retries)
    retriever = Retriever(
        embedder,
        vector_store,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
    )

    app.state.db_engine = engine
    app.state.session_factory = create_session_factory(engine)
    app.state.embedding_service = embedder
    app.state.vector_store = vector_store
    app.state.redis_memory = RedisMemory(
        redis,
        max_messages=settings.chat_history_max_messages,
        ttl_seconds=settings.chat_ttl_seconds,
    )
    app.state.rag_pipeline = RagPipeline(
        llm=llm,
        rewriter=QueryRewriter(llm),
        retriever=retriever,
        temperature=settings.llm_temperature,
    )
    app.state.booking_extractor = BookingExtractor(llm)

    try:
        await vector_store.ensure_collection()  # fail fast if Qdrant is unreachable
        await redis.ping()  # fail fast if Redis is unreachable
        yield
    finally:
        await redis.aclose()
        await qdrant.close()
        await gemini.aio.aclose()
        await engine.dispose()
        logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Document ingestion and conversational RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Turn expected application errors into clean JSON responses."""
        headers: dict[str, str] = {}
        if exc.retry_after_seconds is not None and exc.retry_after_seconds > 0:
            headers["Retry-After"] = str(exc.retry_after_seconds)
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.message}, headers=headers
        )

    app.include_router(health.router)
    app.include_router(api_router)
    return app


app: FastAPI = create_app()