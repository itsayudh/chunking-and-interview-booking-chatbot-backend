"""Unit tests for booking detail validation."""

from datetime import UTC, date, datetime, time

import pytest

from app.services.booking_validation import (
    BookingValidationError,
    ensure_slot_in_future,
    validate_date,
    validate_email,
    validate_name,
    validate_time,
)

TODAY = date(2026, 11, 4)


def test_name_is_cleaned() -> None:
    assert validate_name("  Asha   Karki ") == "Asha Karki"


@pytest.mark.parametrize("bad", ["", "   ", "12345", "x" * 256])
def test_bad_names_are_rejected(bad: str) -> None:
    with pytest.raises(BookingValidationError) as info:
        validate_name(bad)
    assert info.value.field == "name"


def test_valid_email_is_accepted() -> None:
    assert validate_email("  Asha.Karki@Example.COM ").lower() == "asha.karki@example.com"


@pytest.mark.parametrize(
    "bad", ["asha", "asha@", "@example.com", "asha@gmail", "a b@example.com", ""]
)
def test_bad_emails_are_rejected(bad: str) -> None:
    with pytest.raises(BookingValidationError) as info:
        validate_email(bad)
    assert info.value.field == "email"


def test_today_and_future_dates_are_accepted() -> None:
    assert validate_date("2026-11-04", TODAY) == TODAY
    assert validate_date("2026-12-25", TODAY) == date(2026, 12, 25)


@pytest.mark.parametrize(
    "bad", ["2026-11-03", "11/05/2026", "20261105", "2026-W45-4", "tomorrow", "2026-02-30", ""]
)
def test_bad_dates_are_rejected(bad: str) -> None:
    with pytest.raises(BookingValidationError) as info:
        validate_date(bad, TODAY)
    assert info.value.field == "date"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14:30", time(14, 30)),
        ("00:00", time(0, 0)),
        ("23:59", time(23, 59)),
        ("9:30", time(9, 30)),
    ],
)
def test_valid_times(text: str, expected: time) -> None:
    assert validate_time(text) == expected


@pytest.mark.parametrize("bad", ["24:00", "14:60", "2pm", "14:30:00", "1430", ""])
def test_bad_times_are_rejected(bad: str) -> None:
    with pytest.raises(BookingValidationError) as info:
        validate_time(bad)
    assert info.value.field == "time"


def test_slot_must_be_in_the_future() -> None:
    now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)

    ensure_slot_in_future(date(2026, 11, 4), time(13, 0), now)
    ensure_slot_in_future(date(2026, 11, 5), time(0, 0), now)

    for past in (time(12, 0), time(11, 59)):
        with pytest.raises(BookingValidationError):
            ensure_slot_in_future(date(2026, 11, 4), past, now)