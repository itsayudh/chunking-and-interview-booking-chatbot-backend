"""Test doubles and builders for the chat tests."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from zoneinfo import ZoneInfo

from fakeredis import FakeAsyncRedis

from app.memory.redis_memory import ChatMessage, RedisMemory
from app.rag.llm import LLMService
from app.rag.pipeline import RagAnswer, RagPipeline
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import BookingExtraction, TurnIntent
from app.services.booking_extractor import BookingExtractor
from app.services.booking_service import BookingService
from app.services.chat_service import ChatService
from tests.fakes_booking import FakeBookingRepository
from tests.fakes_rag import FakeLLM, make_hit

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)


def booking_extraction(
    intent: TurnIntent = TurnIntent.BOOKING,
    *,
    name: str | None = None,
    email: str | None = None,
    date: str | None = None,
    time: str | None = None,
) -> BookingExtraction:
    return BookingExtraction(intent=intent, name=name, email=email, date=date, time=time)


QUESTION = booking_extraction(TurnIntent.QUESTION)
CANCEL = booking_extraction(TurnIntent.CANCEL_BOOKING)


class FakePipeline:
    """Stands in for RagPipeline: records calls and replays scripted answers or errors."""

    def __init__(self, answers: Sequence[RagAnswer | Exception] = ()) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def answer(self, question: str, history: Sequence[ChatMessage]) -> RagAnswer:
        self.calls.append((question, list(history)))
        if self._answers:
            outcome = self._answers.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return RagAnswer(
            answer=f"answer to: {question}",
            answered=True,
            standalone_question=question,
            sources=[make_hit()],
        )


def build_chat_service(
    *,
    redis_client: FakeAsyncRedis,
    llm: FakeLLM,
    pipeline: FakePipeline,
    repository: FakeBookingRepository,
) -> ChatService:
    """A real ChatService wired to fakes. The fake LLM supplies the intent extractions."""
    memory = RedisMemory(redis_client, max_messages=20, ttl_seconds=3600)
    booking_service = BookingService(
        memory=memory,
        repository=cast(BookingRepository, repository),
        timezone=ZoneInfo("UTC"),
        clock=lambda: NOW,
    )
    return ChatService(
        memory=memory,
        pipeline=cast(RagPipeline, pipeline),
        extractor=BookingExtractor(cast(LLMService, llm)),
        booking_service=booking_service,
        timezone_name="UTC",
    )