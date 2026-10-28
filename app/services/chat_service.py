"""Chat orchestration: route each message to the booking flow or the RAG pipeline."""

import logging
import uuid
from dataclasses import dataclass

from app.core.exceptions import SessionNotFoundError
from app.memory.redis_memory import BookingDraft, ChatMessage, RedisMemory
from app.models.booking import InterviewBooking
from app.models.enums import ChatIntent
from app.rag.pipeline import RagPipeline
from app.rag.vector_store import SearchHit
from app.schemas.booking import BookingExtraction, TurnIntent
from app.services.booking_extractor import BookingExtractor
from app.services.booking_service import BookingService, BookingStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Everything the API needs to answer one chat message."""

    session_id: str
    reply: str
    intent: ChatIntent
    answered: bool
    sources: list[SearchHit]
    booking: InterviewBooking | None = None
    booking_status: BookingStatus | None = None


class ChatService:
    """Decides what each message is, handles it, and remembers the conversation."""

    def __init__(
        self,
        *,
        memory: RedisMemory,
        pipeline: RagPipeline,
        extractor: BookingExtractor,
        booking_service: BookingService,
        timezone_name: str,
    ) -> None:
        self._memory = memory
        self._pipeline = pipeline
        self._extractor = extractor
        self._booking = booking_service
        self._timezone_name = timezone_name

    async def chat(self, *, message: str, session_id: str | None) -> ChatResult:
        """Handle one user message and return the reply.

        The turn is stored in Redis only after it succeeds, so a failed request leaves the
        conversation unchanged.
        """
        resolved_id = self._resolve_session_id(session_id)
        history = await self._memory.get_history(resolved_id)
        draft = await self._memory.get_booking_draft(resolved_id)

        extraction = await self._extractor.extract(
            message=message,
            history=history,
            draft=draft,
            now=self._booking.now(),
            timezone_name=self._timezone_name,
        )

        if extraction.intent is TurnIntent.QUESTION:
            result = await self._answer_question(resolved_id, message, history, draft)
        else:
            result = await self._handle_booking(resolved_id, extraction)

        await self._memory.append_messages(
            resolved_id,
            [
                ChatMessage(role="user", content=message),
                ChatMessage(role="assistant", content=result.reply),
            ],
        )
        return result

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return a stored conversation, or raise SessionNotFoundError."""
        resolved_id = RedisMemory.validate_session_id(session_id)
        if not await self._memory.session_exists(resolved_id):
            raise SessionNotFoundError(f"No conversation found for session '{resolved_id}'.")
        return await self._memory.get_history(resolved_id)

    async def clear_session(self, session_id: str) -> None:
        """Delete a conversation and any booking in progress (safe to repeat)."""
        await self._memory.clear_session(RedisMemory.validate_session_id(session_id))

    @staticmethod
    def _resolve_session_id(session_id: str | None) -> str:
        if session_id is None or not session_id.strip():
            return uuid.uuid4().hex
        return RedisMemory.validate_session_id(session_id.strip())

    async def _answer_question(
        self,
        session_id: str,
        message: str,
        history: list[ChatMessage],
        draft: BookingDraft | None,
    ) -> ChatResult:
        rag = await self._pipeline.answer(message, history)
        reply = rag.answer
        if draft is not None:  # the user asked something mid-booking: keep the booking alive
            missing = draft.missing_fields()
            detail = (
                f"Still needed: {', '.join(missing)}."
                if missing
                else "Say 'keep it' to finish."
            )
            reply += (
                f"\n\n(Your interview booking is still in progress. {detail} "
                "Say 'cancel' to drop it.)"
            )
        return ChatResult(
            session_id=session_id,
            reply=reply,
            intent=ChatIntent.QUESTION,
            answered=rag.answered,
            sources=rag.sources,
        )

    async def _handle_booking(
        self, session_id: str, extraction: BookingExtraction
    ) -> ChatResult:
        outcome = await self._booking.process(session_id, extraction)
        return ChatResult(
            session_id=session_id,
            reply=outcome.reply,
            intent=ChatIntent.BOOKING,
            answered=True,
            sources=[],
            booking=outcome.booking,
            booking_status=outcome.status,
        )