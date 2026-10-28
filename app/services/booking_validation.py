"""Validation and normalisation of booking details collected through the chat."""

import re
from datetime import date, datetime, time

from pydantic import EmailStr, TypeAdapter, ValidationError

_EMAIL_ADAPTER: TypeAdapter[str] = TypeAdapter(EmailStr)
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_PATTERN = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_MAX_NAME_LENGTH = 255
_MAX_EMAIL_LENGTH = 320


class BookingValidationError(ValueError):
    """A booking detail is invalid. ``message`` is safe to show to the user."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


def validate_name(value: str) -> str:
    """Collapse whitespace and require a plausible name (at least one letter)."""
    name = " ".join(value.split())
    if not name or len(name) > _MAX_NAME_LENGTH or not any(char.isalpha() for char in name):
        raise BookingValidationError("name", "I didn't catch a valid name.")
    return name


def validate_email(value: str) -> str:
    """Check the address syntax (no deliverability check) and return it normalised."""
    candidate = value.strip()
    if len(candidate) > _MAX_EMAIL_LENGTH:
        raise BookingValidationError("email", "That email address doesn't look valid.")
    try:
        return _EMAIL_ADAPTER.validate_python(candidate)
    except ValidationError as exc:
        raise BookingValidationError("email", "That email address doesn't look valid.") from exc


def validate_date(value: str, today: date) -> date:
    """Parse a strict YYYY-MM-DD date that is today or later."""
    text = value.strip()
    if not _ISO_DATE_PATTERN.fullmatch(text):
        raise BookingValidationError("date", "I couldn't read that date.")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise BookingValidationError("date", "I couldn't read that date.") from exc
    if parsed < today:
        raise BookingValidationError("date", "That date is in the past.")
    return parsed


def validate_time(value: str) -> time:
    """Parse a 24-hour HH:MM time."""
    match = _TIME_PATTERN.fullmatch(value.strip())
    if match is None:
        raise BookingValidationError("time", "I couldn't read that time.")
    return time(int(match.group(1)), int(match.group(2)))


def ensure_slot_in_future(day: date, at: time, now: datetime) -> None:
    """Require the date and time together to be later than ``now`` (timezone-aware)."""
    slot = datetime.combine(day, at, tzinfo=now.tzinfo)
    if slot <= now:
        raise BookingValidationError("time", "That time has already passed today.")