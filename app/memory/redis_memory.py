"""Redis-backed conversation memory and booking draft storage."""

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, Field, ValidationError
from redis.asyncio import Redis

from app.core.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_BOOKING_FIELDS: tuple[str, ...] = ("name", "email", "date", "time")


class ChatMessage(BaseModel):
    """One message in a conversation."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BookingDraft(BaseModel):
    """Interview booking details collected so far. Values are raw, unvalidated strings."""

    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None

    def missing_fields(self) -> list[str]:
        """Names of the fields that have not been provided yet, in a fixed order."""
        return [field for field in _BOOKING_FIELDS if not getattr(self, field)]

    @property
    def is_complete(self) -> bool:
        """True when all four fields have a value."""
        return not self.missing_fields()

    def merged_with(self, updates: Mapping[str, str | None]) -> Self:
        """Return a copy with non-empty updates applied; empty values never erase data."""
        cleaned = {
            key: value.strip()
            for key, value in updates.items()
            if key in _BOOKING_FIELDS and value and value.strip()
        }
        return self.model_copy(update=cleaned)


class RedisMemory:
    """Stores chat history and booking drafts per session.

    The Redis client must be created with ``decode_responses=True``.
    """

    def __init__(self, client: Redis, *, max_messages: int, ttl_seconds: int) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than 0")
        self._client = client
        self._max_messages = max_messages
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def validate_session_id(session_id: str) -> str:
        """Return the session id if it is safe to use inside a Redis key."""
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise InvalidRequestError(
                "session_id must be 1-64 characters: letters, digits, '-' or '_'."
            )
        return session_id

    def _chat_key(self, session_id: str) -> str:
        return f"chat:{self.validate_session_id(session_id)}"

    def _draft_key(self, session_id: str) -> str:
        return f"booking_state:{self.validate_session_id(session_id)}"

    @staticmethod
    async def _run(command: Any) -> Any:
        """Await a redis-py command (its type hints allow sync or async return values)."""
        return await command

    # ----- chat history -----

    async def append_messages(self, session_id: str, messages: Sequence[ChatMessage]) -> None:
        """Add messages, keep only the newest ones, and refresh the expiry (one transaction)."""
        if not messages:
            return
        key = self._chat_key(session_id)
        payloads = [message.model_dump_json() for message in messages]
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *payloads)
            pipe.ltrim(key, -self._max_messages, -1)
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

    async def get_history(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Return the newest messages, oldest first. Corrupt entries are skipped."""
        count = self._max_messages if limit is None else min(limit, self._max_messages)
        if count <= 0:
            raise ValueError("limit must be greater than 0")
        key = self._chat_key(session_id)
        raw = cast(list[str], await self._run(self._client.lrange(key, -count, -1)))

        messages: list[ChatMessage] = []
        for item in raw:
            try:
                messages.append(ChatMessage.model_validate_json(item))
            except ValidationError:
                logger.warning("Skipping corrupt chat message in %s", key)
        return messages

    async def clear_history(self, session_id: str) -> None:
        """Delete a session's chat history."""
        await self._run(self._client.delete(self._chat_key(session_id)))

    # ----- booking draft -----

    async def get_booking_draft(self, session_id: str) -> BookingDraft | None:
        """Return the in-progress booking, or None if there is none."""
        key = self._draft_key(session_id)
        raw = cast(str | None, await self._run(self._client.get(key)))
        if raw is None:
            return None
        try:
            return BookingDraft.model_validate_json(raw)
        except ValidationError:
            logger.warning("Discarding corrupt booking draft in %s", key)
            return None

    async def save_booking_draft(self, session_id: str, draft: BookingDraft) -> None:
        """Store the in-progress booking and refresh its expiry."""
        await self._run(
            self._client.set(
                self._draft_key(session_id), draft.model_dump_json(), ex=self._ttl_seconds
            )
        )

    async def clear_booking_draft(self, session_id: str) -> None:
        """Delete the in-progress booking (after it is saved or cancelled)."""
        await self._run(self._client.delete(self._draft_key(session_id)))

    # ----- whole session -----

    async def clear_session(self, session_id: str) -> bool:
        """Delete history and draft. Returns True if anything existed."""
        deleted = cast(
            int,
            await self._run(
                self._client.delete(self._chat_key(session_id), self._draft_key(session_id))
            ),
        )
        return deleted > 0

    async def session_exists(self, session_id: str) -> bool:
        """True if the session has stored history."""
        exists = cast(int, await self._run(self._client.exists(self._chat_key(session_id))))
        return exists > 0