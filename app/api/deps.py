"""FastAPI dependency providers."""

from collections.abc import AsyncIterator
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository
from app.services.booking_service import BookingService
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a database session scoped to one request."""
    factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with factory() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_document_repository(session: SessionDep) -> DocumentRepository:
    """Provide a DocumentRepository bound to the request session."""
    return DocumentRepository(session)


def get_booking_repository(session: SessionDep) -> BookingRepository:
    """Provide a BookingRepository bound to the request session."""
    return BookingRepository(session)


def get_ingestion_service(
    request: Request,
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
    settings: SettingsDep,
) -> IngestionService:
    """Provide an IngestionService wired to the shared embedder and vector store."""
    return IngestionService(
        repository=repository,
        embedder=request.app.state.embedding_service,
        vector_store=request.app.state.vector_store,
        default_chunk_size=settings.default_chunk_size,
        default_chunk_overlap=settings.default_chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_booking_service(
    request: Request,
    repository: Annotated[BookingRepository, Depends(get_booking_repository)],
    settings: SettingsDep,
) -> BookingService:
    """Provide a BookingService bound to the request's database session."""
    return BookingService(
        memory=request.app.state.redis_memory,
        repository=repository,
        timezone=ZoneInfo(settings.booking_timezone),
    )


def get_chat_service(
    request: Request,
    booking_service: Annotated[BookingService, Depends(get_booking_service)],
    settings: SettingsDep,
) -> ChatService:
    """Provide a ChatService wired to the shared pipeline, extractor and memory."""
    return ChatService(
        memory=request.app.state.redis_memory,
        pipeline=request.app.state.rag_pipeline,
        extractor=request.app.state.booking_extractor,
        booking_service=booking_service,
        timezone_name=settings.booking_timezone,
    )


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]