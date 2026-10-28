"""Real end-to-end check of the booking flow (Gemini + Redis + PostgreSQL)."""

import asyncio
import uuid
from zoneinfo import ZoneInfo

from google import genai
from redis.asyncio import Redis

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.memory.redis_memory import ChatMessage, RedisMemory
from app.models.booking import InterviewBooking
from app.rag.llm import LLMService
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import TurnIntent
from app.services.booking_extractor import BookingExtractor
from app.services.booking_service import BookingService

SCENARIOS: list[tuple[str, list[str]]] = [
    (
        "Multi-turn booking, a bad email that gets corrected",
        [
            "Hi, I'd like to book an interview",
            "My name is Asha Karki and my email is asha.karki@gmail",
            "Sorry, the email is asha.karki@example.com. Tomorrow at 2:30 pm works.",
        ],
    ),
    (
        "A past date is rejected, then the user cancels",
        [
            "Please book an interview for Ben Lee, ben.lee@example.com, "
            "on 2020-01-15 at 10:00",
            "Actually, cancel the booking.",
        ],
    ),
]


async def main() -> None:
    settings = get_settings()
    gemini = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
    engine = create_engine(settings)
    factory = create_session_factory(engine)

    memory = RedisMemory(
        redis,
        max_messages=settings.chat_history_max_messages,
        ttl_seconds=settings.chat_ttl_seconds,
    )
    llm = LLMService(gemini, settings.gemini_llm_model, settings.llm_max_retries)
    extractor = BookingExtractor(llm)
    created: list[InterviewBooking] = []
    sessions: list[str] = []

    print(f"Booking timezone: {settings.booking_timezone}")
    try:
        async with factory() as db_session:
            repository = BookingRepository(db_session)
            service = BookingService(
                memory=memory,
                repository=repository,
                timezone=ZoneInfo(settings.booking_timezone),
            )

            for title, messages in SCENARIOS:
                print(f"\n=== {title} ===")
                session_id = f"check-{uuid.uuid4().hex[:8]}"
                sessions.append(session_id)
                history: list[ChatMessage] = []

                for message in messages:
                    print(f"\nUser: {message}")
                    extraction = await extractor.extract(
                        message=message,
                        history=history,
                        draft=await memory.get_booking_draft(session_id),
                        now=service.now(),
                        timezone_name=settings.booking_timezone,
                    )
                    print("  model understood:", extraction.model_dump(mode="json"))

                    if extraction.intent is TurnIntent.QUESTION:
                        print("  intent is 'question' (this would go to the RAG pipeline)")
                        continue

                    outcome = await service.process(session_id, extraction)
                    print(f"  status: {outcome.status.value}")
                    print(f"  reply:  {outcome.reply}")
                    history += [
                        ChatMessage(role="user", content=message),
                        ChatMessage(role="assistant", content=outcome.reply),
                    ]
                    if outcome.booking is not None:
                        created.append(outcome.booking)

            print("\nBookings created during this check:", len(created))
            for booking in created:
                await repository.delete(booking)
            print("Test bookings deleted")
    finally:
        for session_id in sessions:
            await memory.clear_session(session_id)
        await redis.aclose()
        await gemini.aio.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())