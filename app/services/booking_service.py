"""Interview booking workflow: validate details, keep the draft, save the booking."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

from app.memory.redis_memory import BookingDraft, RedisMemory
from app.models.booking import InterviewBooking
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import BookingExtraction, TurnIntent
from app.services.booking_validation import (
    BookingValidationError,
    ensure_slot_in_future,
    validate_date,
    validate_email,
    validate_name,
    validate_time,
)

logger = logging.getLogger(__name__)

_FIELD_LABELS: dict[str, str] = {
    "name": "your full name",
    "email": "your email address",
    "date": "the interview date",
    "time": "the interview time",
}


class BookingStatus(StrEnum):
    """Result of processing one booking-related message."""

    NEEDS_INFO = "needs_info"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    NOT_IN_PROGRESS = "not_in_progress"


@dataclass(frozen=True, slots=True)
class BookingOutcome:
    """What happened, and the reply to show the user."""

    status: BookingStatus
    reply: str
    booking: InterviewBooking | None = None
    draft: BookingDraft | None = None


def _join(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _normalise(field: str, raw: str, today: date) -> str:
    """Validate one value and return the normalised string stored in the draft."""
    if field == "name":
        return validate_name(raw)
    if field == "email":
        return validate_email(raw)
    if field == "date":
        return validate_date(raw, today).isoformat()
    return validate_time(raw).strftime("%H:%M")


class BookingService:
    """Applies the booking rules. The model understands messages; this class decides."""

    def __init__(
        self,
        *,
        memory: RedisMemory,
        repository: BookingRepository,
        timezone: ZoneInfo,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._memory = memory
        self._repository = repository
        self._timezone = timezone
        self._clock = clock or (lambda: datetime.now(UTC))

    def now(self) -> datetime:
        """Current time in the booking timezone."""
        return self._clock().astimezone(self._timezone)

    async def process(self, session_id: str, extraction: BookingExtraction) -> BookingOutcome:
        """Handle a message whose intent is 'booking' or 'cancel_booking'."""
        if extraction.intent is TurnIntent.QUESTION:
            raise ValueError("process() handles booking and cancel intents only")
        if extraction.intent is TurnIntent.CANCEL_BOOKING:
            return await self._cancel(session_id)
        return await self._advance(session_id, extraction)

    async def _cancel(self, session_id: str) -> BookingOutcome:
        existing = await self._memory.get_booking_draft(session_id)
        if existing is None:
            return BookingOutcome(
                BookingStatus.NOT_IN_PROGRESS, "There is no booking in progress to cancel."
            )
        await self._memory.clear_booking_draft(session_id)
        return BookingOutcome(
            BookingStatus.CANCELLED,
            "Okay, I've cancelled that booking. Tell me if you'd like to book again.",
        )

    async def _advance(self, session_id: str, extraction: BookingExtraction) -> BookingOutcome:
        now = self.now()
        previous = await self._memory.get_booking_draft(session_id)
        draft = previous or BookingDraft()

        updates, problems = self._validate_updates(extraction, now.date())
        draft = draft.merged_with(updates)

        slot_problem = self._slot_problem(draft, now)
        if slot_problem is not None:
            problems.append(slot_problem)
            draft = draft.model_copy(update={"time": None})

        # Save before the database write, so a failed write can be retried without re-asking.
        await self._memory.save_booking_draft(session_id, draft)

        if draft.is_complete and not problems:
            booking = await self._save_booking(session_id, draft)
            await self._memory.clear_booking_draft(session_id)
            return BookingOutcome(
                BookingStatus.CONFIRMED, self._confirmation_reply(booking), booking=booking
            )

        reply = self._ask_reply(problems, draft, in_progress=previous is not None)
        return BookingOutcome(BookingStatus.NEEDS_INFO, reply, draft=draft)

    @staticmethod
    def _validate_updates(
        extraction: BookingExtraction, today: date
    ) -> tuple[dict[str, str], list[str]]:
        """Validate each value the model found. Invalid values are reported, never stored."""
        updates: dict[str, str] = {}
        problems: list[str] = []
        found = (
            ("name", extraction.name),
            ("email", extraction.email),
            ("date", extraction.date),
            ("time", extraction.time),
        )
        for field, raw in found:
            if raw is None or not raw.strip():
                continue
            try:
                updates[field] = _normalise(field, raw, today)
            except BookingValidationError as exc:
                problems.append(exc.message)
        return updates, problems

    @staticmethod
    def _slot_problem(draft: BookingDraft, now: datetime) -> str | None:
        """Return a message if the date and time together are not in the future."""
        if not (draft.date and draft.time):
            return None
        try:
            ensure_slot_in_future(
                date.fromisoformat(draft.date), time.fromisoformat(draft.time), now
            )
        except BookingValidationError as exc:
            return exc.message
        return None

    async def _save_booking(self, session_id: str, draft: BookingDraft) -> InterviewBooking:
        # The draft only ever holds values that already passed validation.
        assert draft.name and draft.email and draft.date and draft.time
        return await self._repository.create(
            session_id=session_id,
            name=draft.name,
            email=draft.email,
            interview_date=date.fromisoformat(draft.date),
            interview_time=time.fromisoformat(draft.time),
        )

    @staticmethod
    def _ask_reply(problems: list[str], draft: BookingDraft, *, in_progress: bool) -> str:
        missing = draft.missing_fields()
        if not missing:  # complete, but this message contained an invalid correction
            return (
                f"{' '.join(problems)} Please send the corrected detail, "
                "or say 'keep it' to continue with what I have."
            )
        need = _join([_FIELD_LABELS[field] for field in missing])
        if problems:
            return f"{' '.join(problems)} Please send me {need}."
        if len(missing) == len(_FIELD_LABELS) and not in_progress:
            return f"I'd be happy to book an interview. Please send me {need}."
        return f"Thanks, I've noted that. I still need {need}."

    @staticmethod
    def _confirmation_reply(booking: InterviewBooking) -> str:
        first_name = booking.name.split()[0]
        when = f"{booking.interview_date:%A, %d %B %Y} at {booking.interview_time:%H:%M}"
        return (
            f"You're all set, {first_name}! I've booked your interview for {when} "
            f"and saved it under {booking.email}."
        )