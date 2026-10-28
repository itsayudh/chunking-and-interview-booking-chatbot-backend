"""API schemas for the chat endpoints."""

import datetime as dt
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from app.models.enums import ChatIntent
from app.schemas.booking import BookingResponse


class ChatRequest(BaseModel):
    """One user message."""

    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Conversation id. Leave empty to start a new conversation.",
    )
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class SourceReference(BaseModel):
    """A document chunk the answer was based on (chunk_index starts at 0)."""

    document_id: uuid.UUID
    filename: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    """The assistant's reply to one message."""

    session_id: str
    answer: str
    intent: ChatIntent
    answered: bool = Field(
        description="False when the documents did not contain the answer to a question"
    )
    sources: list[SourceReference]
    booking: BookingResponse | None = None
    booking_status: str | None = Field(
        default=None, description="needs_info, confirmed, cancelled or not_in_progress"
    )


class ChatMessageResponse(BaseModel):
    """One stored message."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: dt.datetime


class ChatHistoryResponse(BaseModel):
    """The stored conversation of a session."""

    session_id: str
    messages: list[ChatMessageResponse]