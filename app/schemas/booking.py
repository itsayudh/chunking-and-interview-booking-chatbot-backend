"""API and LLM schemas for interview bookings."""

import datetime as dt
import uuid
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class TurnIntent(StrEnum):
    """What the user's latest message is about."""

    BOOKING = "booking"
    CANCEL_BOOKING = "cancel_booking"
    QUESTION = "question"


class BookingExtraction(BaseModel):
    """What the model understood from the user's latest message (the LLM output schema)."""

    intent: TurnIntent = Field(
        description=(
            "booking, cancel_booking, or question "
            "(anything else, including questions about documents)"
        )
    )
    name: str | None = Field(description="The user's full name if stated, otherwise null")
    email: str | None = Field(description="The email address exactly as written, otherwise null")
    date: str | None = Field(description="Interview date as YYYY-MM-DD, otherwise null")
    time: str | None = Field(description="Interview time as 24-hour HH:MM, otherwise null")


class BookingResponse(BaseModel):
    """A stored interview booking."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: str
    name: str
    email: str
    interview_date: dt.date
    interview_time: dt.time
    created_at: dt.datetime