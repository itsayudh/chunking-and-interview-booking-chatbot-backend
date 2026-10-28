"""Unit tests for ChatService: a real service wired to fakes."""

import re
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.core.exceptions import (
    EmbeddingError,
    InvalidRequestError,
    LLMError,
    SessionNotFoundError,
)
from app.models.enums import ChatIntent
from app.rag.pipeline import RagAnswer
from app.services.booking_service import BookingStatus
from tests.fakes_booking import FakeBookingRepository
from tests.fakes_chat import (
    CANCEL,
    QUESTION,
    FakePipeline,
    booking_extraction,
    build_chat_service,
)
from tests.fakes_rag import FakeLLM, make_hit


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
def repository() -> FakeBookingRepository:
    return FakeBookingRepository()


async def test_question_is_answered_and_the_turn_is_stored(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    service = build_chat_service(
        redis_client=redis_client, llm=FakeLLM([QUESTION]), pipeline=pipeline, repository=repository
    )

    result = await service.chat(message="How long do refunds take?", session_id="s1")

    assert result.intent is ChatIntent.QUESTION
    assert result.reply == "answer to: How long do refunds take?"
    assert result.answered is True
    assert len(result.sources) == 1
    assert result.booking is None
    assert pipeline.calls == [("How long do refunds take?", [])]

    history = await service.get_history("s1")
    assert [(m.role, m.content) for m in history] == [
        ("user", "How long do refunds take?"),
        ("assistant", "answer to: How long do refunds take?"),
    ]


async def test_unanswerable_question_is_reported(
    redis_client: FakeAsyncRedis, repository: FakeBookingRepository
) -> None:
    refusal = RagAnswer(
        answer="I couldn't find the answer to that in the uploaded documents.",
        answered=False,
        standalone_question="Who won?",
        sources=[],
    )
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION]),
        pipeline=FakePipeline([refusal]),
        repository=repository,
    )

    result = await service.chat(message="Who won?", session_id="s1")

    assert result.answered is False
    assert result.sources == []


async def test_session_ids_are_generated_or_validated(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    llm = FakeLLM([QUESTION, QUESTION, QUESTION])
    service = build_chat_service(
        redis_client=redis_client, llm=llm, pipeline=pipeline, repository=repository
    )

    generated = await service.chat(message="q1", session_id=None)
    blank = await service.chat(message="q2", session_id="   ")
    given = await service.chat(message="q3", session_id="my-session_1")

    assert re.fullmatch(r"[0-9a-f]{32}", generated.session_id)
    assert re.fullmatch(r"[0-9a-f]{32}", blank.session_id)
    assert generated.session_id != blank.session_id
    assert given.session_id == "my-session_1"

    llm.calls.clear()
    pipeline.calls.clear()
    with pytest.raises(InvalidRequestError):
        await service.chat(message="hi", session_id="bad id!")
    assert llm.calls == []  # rejected before any model call
    assert pipeline.calls == []


async def test_follow_up_receives_the_conversation_history(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION, QUESTION]),
        pipeline=pipeline,
        repository=repository,
    )

    await service.chat(message="What is the refund policy?", session_id="s1")
    await service.chat(message="And for digital products?", session_id="s1")

    question, history = pipeline.calls[1]
    assert question == "And for digital products?"
    assert [m.content for m in history] == [
        "What is the refund policy?",
        "answer to: What is the refund policy?",
    ]


async def test_booking_conversation_across_turns(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    llm = FakeLLM(
        [
            booking_extraction(),
            booking_extraction(
                name="Asha Karki", email="asha@example.com", date="2026-11-05", time="14:30"
            ),
        ]
    )
    service = build_chat_service(
        redis_client=redis_client, llm=llm, pipeline=pipeline, repository=repository
    )

    first = await service.chat(message="I'd like to book an interview", session_id="s1")
    assert first.intent is ChatIntent.BOOKING
    assert first.booking_status is BookingStatus.NEEDS_INFO
    assert first.sources == []

    second = await service.chat(
        message="Asha Karki, asha@example.com, tomorrow 2:30pm", session_id="s1"
    )
    assert second.booking_status is BookingStatus.CONFIRMED
    assert second.booking is not None
    assert second.booking.name == "Asha Karki"
    assert len(repository.bookings) == 1
    assert pipeline.calls == []  # booking messages never reach the RAG pipeline
    assert len(await service.get_history("s1")) == 4


async def test_question_in_the_middle_of_a_booking_keeps_the_booking(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    llm = FakeLLM(
        [
            booking_extraction(name="Asha Karki"),
            QUESTION,
            booking_extraction(email="asha@example.com", date="2026-11-05", time="10:00"),
        ]
    )
    service = build_chat_service(
        redis_client=redis_client, llm=llm, pipeline=pipeline, repository=repository
    )

    await service.chat(message="Book an interview for Asha Karki", session_id="s1")
    middle = await service.chat(message="How long does shipping take?", session_id="s1")

    assert middle.intent is ChatIntent.QUESTION
    assert middle.reply.startswith("answer to: How long does shipping take?")
    assert "still in progress" in middle.reply
    assert "email" in middle.reply
    assert repository.bookings == []

    done = await service.chat(message="asha@example.com, 5 Nov at 10", session_id="s1")
    assert done.booking_status is BookingStatus.CONFIRMED
    assert repository.bookings[0].name == "Asha Karki"  # the name given earlier was kept


async def test_cancelling_a_booking(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([booking_extraction(name="Asha"), CANCEL]),
        pipeline=pipeline,
        repository=repository,
    )

    await service.chat(message="Book an interview, I'm Asha", session_id="s1")
    result = await service.chat(message="Never mind, cancel it", session_id="s1")

    assert result.intent is ChatIntent.BOOKING
    assert result.booking_status is BookingStatus.CANCELLED


async def test_failed_turn_is_not_stored(
    redis_client: FakeAsyncRedis, repository: FakeBookingRepository
) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION]),
        pipeline=FakePipeline([EmbeddingError("quota exceeded")]),
        repository=repository,
    )

    with pytest.raises(EmbeddingError):
        await service.chat(message="Any question", session_id="s1")

    with pytest.raises(SessionNotFoundError):
        await service.get_history("s1")


async def test_intent_detection_failure_propagates(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([LLMError("down")]),
        pipeline=pipeline,
        repository=repository,
    )

    with pytest.raises(LLMError):
        await service.chat(message="hello", session_id="s1")
    assert pipeline.calls == []


async def test_history_and_clear_session(
    redis_client: FakeAsyncRedis, pipeline: FakePipeline, repository: FakeBookingRepository
) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION]),
        pipeline=pipeline,
        repository=repository,
    )
    await service.chat(message="A question", session_id="s1")

    assert len(await service.get_history("s1")) == 2

    await service.clear_session("s1")
    await service.clear_session("s1")  # safe to repeat
    with pytest.raises(SessionNotFoundError):
        await service.get_history("s1")
    with pytest.raises(InvalidRequestError):
        await service.get_history("bad id!")


def test_make_hit_is_available_for_sources() -> None:
    assert make_hit().filename == "policy.txt"