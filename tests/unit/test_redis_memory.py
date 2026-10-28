"""Unit tests for RedisMemory, using fakeredis (an in-memory Redis with real semantics)."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.core.exceptions import InvalidRequestError
from app.memory.redis_memory import BookingDraft, ChatMessage, RedisMemory

MAX_MESSAGES = 6
TTL = 100


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def memory(redis_client: FakeAsyncRedis) -> RedisMemory:
    return RedisMemory(redis_client, max_messages=MAX_MESSAGES, ttl_seconds=TTL)


def turn(n: int) -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content=f"question {n}"),
        ChatMessage(role="assistant", content=f"answer {n}"),
    ]


async def test_history_keeps_order(memory: RedisMemory) -> None:
    await memory.append_messages("s1", turn(1))
    await memory.append_messages("s1", turn(2))

    history = await memory.get_history("s1")

    assert [m.content for m in history] == ["question 1", "answer 1", "question 2", "answer 2"]
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]


async def test_empty_session_has_empty_history(memory: RedisMemory) -> None:
    assert await memory.get_history("nobody") == []
    assert not await memory.session_exists("nobody")


async def test_sliding_window_keeps_only_newest(memory: RedisMemory) -> None:
    for n in range(1, 6):  # 10 messages, window is 6
        await memory.append_messages("s1", turn(n))

    history = await memory.get_history("s1")

    assert len(history) == MAX_MESSAGES
    assert history[0].content == "question 3"
    assert history[-1].content == "answer 5"


async def test_limit_returns_newest_subset(memory: RedisMemory) -> None:
    for n in range(1, 4):
        await memory.append_messages("s1", turn(n))

    history = await memory.get_history("s1", limit=2)

    assert [m.content for m in history] == ["question 3", "answer 3"]


async def test_invalid_limit_raises(memory: RedisMemory) -> None:
    with pytest.raises(ValueError):
        await memory.get_history("s1", limit=0)


async def test_sessions_are_isolated(memory: RedisMemory) -> None:
    await memory.append_messages("alice", turn(1))
    await memory.append_messages("bob", turn(2))

    assert (await memory.get_history("alice"))[0].content == "question 1"
    assert (await memory.get_history("bob"))[0].content == "question 2"


async def test_ttl_is_set_and_refreshed(memory: RedisMemory, redis_client: FakeAsyncRedis) -> None:
    await memory.append_messages("s1", turn(1))
    assert 0 < await redis_client.ttl("chat:s1") <= TTL

    await redis_client.expire("chat:s1", 5)
    await memory.append_messages("s1", turn(2))
    assert await redis_client.ttl("chat:s1") > 5


async def test_empty_append_does_nothing(memory: RedisMemory) -> None:
    await memory.append_messages("s1", [])
    assert not await memory.session_exists("s1")


async def test_clear_history(memory: RedisMemory) -> None:
    await memory.append_messages("s1", turn(1))
    await memory.clear_history("s1")
    assert await memory.get_history("s1") == []


async def test_corrupt_message_is_skipped(
    memory: RedisMemory, redis_client: FakeAsyncRedis
) -> None:
    await memory.append_messages("s1", turn(1))
    await redis_client.rpush("chat:s1", "{not valid json")
    await memory.append_messages("s1", turn(2))

    history = await memory.get_history("s1")

    assert [m.content for m in history] == ["question 1", "answer 1", "question 2", "answer 2"]


async def test_booking_draft_round_trip(memory: RedisMemory, redis_client: FakeAsyncRedis) -> None:
    assert await memory.get_booking_draft("s1") is None

    await memory.save_booking_draft("s1", BookingDraft(name="Asha", email="asha@example.com"))
    draft = await memory.get_booking_draft("s1")

    assert draft is not None
    assert draft.name == "Asha"
    assert draft.missing_fields() == ["date", "time"]
    assert not draft.is_complete
    assert 0 < await redis_client.ttl("booking_state:s1") <= TTL


async def test_clear_booking_draft(memory: RedisMemory) -> None:
    await memory.save_booking_draft("s1", BookingDraft(name="Asha"))
    await memory.clear_booking_draft("s1")
    assert await memory.get_booking_draft("s1") is None


async def test_corrupt_draft_is_discarded(
    memory: RedisMemory, redis_client: FakeAsyncRedis
) -> None:
    await redis_client.set("booking_state:s1", "garbage")
    assert await memory.get_booking_draft("s1") is None


async def test_clear_session_removes_history_and_draft(memory: RedisMemory) -> None:
    await memory.append_messages("s1", turn(1))
    await memory.save_booking_draft("s1", BookingDraft(name="Asha"))

    assert await memory.clear_session("s1") is True

    assert await memory.get_history("s1") == []
    assert await memory.get_booking_draft("s1") is None
    assert await memory.clear_session("s1") is False


def test_draft_completeness_and_merge() -> None:
    draft = BookingDraft(name="Asha")
    merged = draft.merged_with(
        {"email": " asha@example.com ", "date": "", "time": None, "unknown": "x"}
    )

    assert merged.email == "asha@example.com"
    assert merged.date is None  # empty values never erase or add data
    assert merged.name == "Asha"
    assert merged.missing_fields() == ["date", "time"]

    full = merged.merged_with({"date": "2026-11-05", "time": "14:30", "name": "Asha K"})
    assert full.is_complete
    assert full.name == "Asha K"


@pytest.mark.parametrize("bad_id", ["", "a:b", "a b", "../x", "x" * 65, "chat:other", "naïve"])
async def test_unsafe_session_ids_are_rejected(memory: RedisMemory, bad_id: str) -> None:
    with pytest.raises(InvalidRequestError):
        await memory.get_history(bad_id)
    with pytest.raises(InvalidRequestError):
        await memory.save_booking_draft(bad_id, BookingDraft())


@pytest.mark.parametrize(
    "good_id",
    ["abc", "A-b_9", "x" * 64, "123e4567-e89b-12d3-a456-426614174000"],
)
async def test_safe_session_ids_are_accepted(memory: RedisMemory, good_id: str) -> None:
    assert await memory.get_history(good_id) == []