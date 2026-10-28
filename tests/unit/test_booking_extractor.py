"""Unit tests for BookingExtractor (the LLM is faked)."""

from datetime import UTC, datetime
from typing import cast

import pytest

from app.memory.redis_memory import BookingDraft, ChatMessage
from app.rag.llm import LLMService
from app.schemas.booking import BookingExtraction, TurnIntent
from app.services.booking_extractor import BookingExtractor
from tests.fakes_rag import FakeLLM

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)  # a Wednesday
RESULT = BookingExtraction(
    intent=TurnIntent.BOOKING, name=None, email=None, date="2026-11-05", time="14:30"
)
HISTORY = [
    ChatMessage(role="user", content="I want to book an interview"),
    ChatMessage(role="assistant", content="Please send me your details."),
]


def extractor_of(llm: FakeLLM, max_history: int = 6) -> BookingExtractor:
    return BookingExtractor(cast(LLMService, llm), max_history_messages=max_history)


async def test_prompt_contains_date_draft_history_and_message() -> None:
    llm = FakeLLM([RESULT])

    result = await extractor_of(llm).extract(
        message="  tomorrow at 2:30 pm  ",
        history=HISTORY,
        draft=BookingDraft(name="Asha"),
        now=NOW,
        timezone_name="UTC",
    )

    assert result is RESULT
    call = llm.calls[0]
    system = str(call["system"])
    assert "2026-11-04" in system
    assert "Wednesday" in system
    assert "UTC" in system
    prompt = call["messages"][0].content  # type: ignore[index]
    assert "name: Asha" in prompt
    assert "User: I want to book an interview" in prompt
    assert "Final user message: tomorrow at 2:30 pm" in prompt
    assert call["schema"] is BookingExtraction


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        (None, "no booking in progress"),
        (BookingDraft(), "booking started, no details yet"),
    ],
)
async def test_draft_summary_variants(draft: BookingDraft | None, expected: str) -> None:
    llm = FakeLLM([RESULT])
    await extractor_of(llm).extract(
        message="hi", history=[], draft=draft, now=NOW, timezone_name="UTC"
    )
    assert expected in llm.calls[0]["messages"][0].content  # type: ignore[index]


async def test_only_recent_history_is_sent() -> None:
    history = [ChatMessage(role="user", content=f"message {n}") for n in range(1, 6)]
    llm = FakeLLM([RESULT])

    await extractor_of(llm, max_history=2).extract(
        message="hi", history=history, draft=None, now=NOW, timezone_name="UTC"
    )

    prompt = llm.calls[0]["messages"][0].content  # type: ignore[index]
    assert "message 5" in prompt
    assert "message 4" in prompt
    assert "message 3" not in prompt


async def test_blank_message_is_rejected() -> None:
    with pytest.raises(ValueError):
        await extractor_of(FakeLLM()).extract(
            message="   ", history=[], draft=None, now=NOW, timezone_name="UTC"
        )