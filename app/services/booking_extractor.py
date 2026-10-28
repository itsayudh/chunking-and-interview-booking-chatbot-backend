"""Uses the LLM to understand what a chat message says about a booking."""

from collections.abc import Sequence
from datetime import datetime

from app.memory.redis_memory import BookingDraft, ChatMessage
from app.rag.llm import LLMMessage, LLMService
from app.rag.prompts import (
    build_booking_analysis_prompt,
    build_booking_analysis_system_prompt,
)
from app.schemas.booking import BookingExtraction


class BookingExtractor:
    """Extracts intent and booking details from a message. It never saves anything."""

    def __init__(self, llm: LLMService, max_history_messages: int = 6) -> None:
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be greater than 0")
        self._llm = llm
        self._max_history_messages = max_history_messages

    async def extract(
        self,
        *,
        message: str,
        history: Sequence[ChatMessage],
        draft: BookingDraft | None,
        now: datetime,
        timezone_name: str,
    ) -> BookingExtraction:
        """Return the intent and any details stated in ``message``."""
        text = message.strip()
        if not text:
            raise ValueError("Message must not be empty")

        system = build_booking_analysis_system_prompt(
            today=now.date().isoformat(), weekday=now.strftime("%A"), timezone=timezone_name
        )
        prompt = build_booking_analysis_prompt(
            list(history)[-self._max_history_messages :], self._summarize(draft), text
        )
        return await self._llm.generate_structured(
            system_instruction=system,
            messages=[LLMMessage(role="user", content=prompt)],
            schema=BookingExtraction,
        )

    @staticmethod
    def _summarize(draft: BookingDraft | None) -> str:
        if draft is None:
            return "no booking in progress"
        provided = [f"{field}: {value}" for field, value in draft.model_dump().items() if value]
        return "; ".join(provided) if provided else "booking started, no details yet"