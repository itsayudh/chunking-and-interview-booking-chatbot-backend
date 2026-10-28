"""Unit tests for BookingService (fake repository, fakeredis, fixed clock)."""

import datetime as dt
from collections.abc import AsyncIterator
from typing import cast
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.memory.redis_memory import BookingDraft, RedisMemory
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import BookingExtraction, TurnIntent
from app.services.booking_service import BookingService, BookingStatus
from tests.fakes_booking import FakeBookingRepository

NOW = dt.datetime(2026, 11, 4, 12, 0, tzinfo=dt.UTC)  # a Wednesday, noon UTC


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def memory(redis_client: FakeAsyncRedis) -> RedisMemory:
    return RedisMemory(redis_client, max_messages=20, ttl_seconds=3600)


@pytest.fixture
def repository() -> FakeBookingRepository:
    return FakeBookingRepository()


def make_service(
    memory: RedisMemory,
    repository: FakeBookingRepository,
    zone: str = "UTC",
    now: dt.datetime = NOW,
) -> BookingService:
    return BookingService(
        memory=memory,
        repository=cast(BookingRepository, repository),
        timezone=ZoneInfo(zone),
        clock=lambda: now,
    )


def extraction(
    intent: TurnIntent = TurnIntent.BOOKING,
    *,
    name: str | None = None,
    email: str | None = None,
    date: str | None = None,
    time: str | None = None,
) -> BookingExtraction:
    return BookingExtraction(intent=intent, name=name, email=email, date=date, time=time)


async def test_starting_a_booking_asks_for_everything(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    outcome = await make_service(memory, repository).process("s1", extraction())

    assert outcome.status is BookingStatus.NEEDS_INFO
    for label in ("full name", "email address", "date", "time"):
        assert label in outcome.reply
    assert await memory.get_booking_draft("s1") == BookingDraft()  # marks "booking in progress"


async def test_multi_turn_booking_is_confirmed_and_saved(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    service = make_service(memory, repository)

    first = await service.process("s1", extraction(name="Asha  Karki", email="asha@example.com"))
    assert first.status is BookingStatus.NEEDS_INFO
    assert "date" in first.reply
    assert repository.bookings == []

    second = await service.process("s1", extraction(date="2026-11-05", time="14:30"))

    assert second.status is BookingStatus.CONFIRMED
    assert "Asha" in second.reply
    assert "05 November 2026" in second.reply
    assert "14:30" in second.reply
    (booking,) = repository.bookings
    assert booking.name == "Asha Karki"
    assert booking.session_id == "s1"
    assert booking.interview_date == dt.date(2026, 11, 5)
    assert booking.interview_time == dt.time(14, 30)
    assert await memory.get_booking_draft("s1") is None


async def test_everything_in_one_message(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    outcome = await make_service(memory, repository).process(
        "s1",
        extraction(name="Ben Lee", email="ben@example.com", date="2026-11-10", time="9:00"),
    )
    assert outcome.status is BookingStatus.CONFIRMED
    assert repository.bookings[0].interview_time == dt.time(9, 0)


async def test_invalid_email_is_not_stored(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    outcome = await make_service(memory, repository).process(
        "s1", extraction(name="Asha", email="asha@gmail")
    )

    assert outcome.status is BookingStatus.NEEDS_INFO
    assert "doesn't look valid" in outcome.reply
    assert "email address" in outcome.reply
    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert draft.name == "Asha"
    assert draft.email is None


async def test_past_date_is_rejected(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    outcome = await make_service(memory, repository).process("s1", extraction(date="2026-11-03"))

    assert "in the past" in outcome.reply
    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert draft.date is None


async def test_time_already_passed_today_is_dropped(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    outcome = await make_service(memory, repository).process(
        "s1", extraction(date="2026-11-04", time="09:00")
    )

    assert "already passed" in outcome.reply
    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert (draft.date, draft.time) == ("2026-11-04", None)


async def test_moving_the_date_to_today_clears_a_stored_time_that_has_passed(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    service = make_service(memory, repository)
    await service.process("s1", extraction(date="2026-11-05", time="09:00"))

    outcome = await service.process("s1", extraction(date="2026-11-04"))

    assert "already passed" in outcome.reply
    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert (draft.date, draft.time) == ("2026-11-04", None)


async def test_corrections_overwrite_earlier_values(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    service = make_service(memory, repository)
    await service.process("s1", extraction(email="old@example.com"))
    await service.process("s1", extraction(email="new@example.com"))

    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert draft.email == "new@example.com"


async def test_blank_values_are_ignored(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    await make_service(memory, repository).process("s1", extraction(name="   ", email=""))
    assert await memory.get_booking_draft("s1") == BookingDraft()


async def test_cancel_clears_the_draft(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    service = make_service(memory, repository)
    await service.process("s1", extraction(name="Asha"))

    outcome = await service.process("s1", extraction(TurnIntent.CANCEL_BOOKING))

    assert outcome.status is BookingStatus.CANCELLED
    assert await memory.get_booking_draft("s1") is None


async def test_cancel_without_booking(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    outcome = await make_service(memory, repository).process(
        "s1", extraction(TurnIntent.CANCEL_BOOKING)
    )
    assert outcome.status is BookingStatus.NOT_IN_PROGRESS


async def test_question_intent_is_not_handled_here(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    with pytest.raises(ValueError):
        await make_service(memory, repository).process("s1", extraction(TurnIntent.QUESTION))


async def test_database_failure_keeps_details_and_retry_succeeds(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    service = make_service(memory, repository)
    repository.fail = True

    with pytest.raises(RuntimeError):
        await service.process(
            "s1",
            extraction(name="Asha", email="asha@example.com", date="2026-11-05", time="10:00"),
        )
    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert draft.is_complete  # nothing the user said was lost

    # A message with an invalid correction must not save anything.
    blocked = await service.process("s1", extraction(email="not-an-email"))
    assert blocked.status is BookingStatus.NEEDS_INFO
    assert "keep it" in blocked.reply

    repository.fail = False
    retried = await service.process("s1", extraction())  # "keep it"
    assert retried.status is BookingStatus.CONFIRMED
    assert len(repository.bookings) == 1
    assert await memory.get_booking_draft("s1") is None


async def test_today_is_decided_by_the_booking_timezone(
    memory: RedisMemory, repository: FakeBookingRepository
) -> None:
    # 20:00 UTC on 4 Nov is already 05:00 on 5 Nov in Tokyo.
    late = dt.datetime(2026, 11, 4, 20, 0, tzinfo=dt.UTC)
    service = make_service(memory, repository, zone="Asia/Tokyo", now=late)

    stale = await service.process("s1", extraction(date="2026-11-04"))
    assert "in the past" in stale.reply

    ok = await service.process("s1", extraction(date="2026-11-05", time="06:00"))
    assert ok.status is BookingStatus.NEEDS_INFO
    draft = await memory.get_booking_draft("s1")
    assert draft is not None
    assert (draft.date, draft.time) == ("2026-11-05", "06:00")