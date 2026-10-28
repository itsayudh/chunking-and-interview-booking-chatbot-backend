"""Unit tests for the shared retry and quota helpers."""

import pytest

from app.rag.retry import (
    describe_quota_error,
    format_duration,
    is_retryable,
    parse_quota_error,
    retry_delay,
)
from tests.fakes_quota import DAILY, PER_MINUTE, quota_error


def test_daily_quota_is_recognised() -> None:
    info = parse_quota_error(quota_error(quota_id=DAILY, retry_delay_text="15254s"))
    assert info.is_daily is True
    assert info.quota_id == DAILY
    assert info.retry_after_seconds == 15254.0


def test_per_minute_quota_is_not_daily() -> None:
    info = parse_quota_error(quota_error(quota_id=PER_MINUTE, retry_delay_text="27s"))
    assert info.is_daily is False
    assert info.retry_after_seconds == 27.0


def test_missing_details_give_empty_info() -> None:
    info = parse_quota_error(quota_error())
    assert (info.quota_id, info.is_daily, info.retry_after_seconds) == (None, False, None)


def test_decimal_and_malformed_retry_delays() -> None:
    assert parse_quota_error(quota_error(retry_delay_text="4.5s")).retry_after_seconds == 4.5
    assert parse_quota_error(quota_error(retry_delay_text="soon")).retry_after_seconds is None


@pytest.mark.parametrize(
    ("code", "expected"),
    [(429, True), (500, True), (503, True), (400, False), (404, False)],
)
def test_is_retryable(code: int, expected: bool) -> None:
    assert is_retryable(quota_error(code=code)) is expected


def test_daily_limit_is_never_retried() -> None:
    assert retry_delay(quota_error(quota_id=DAILY, retry_delay_text="15254s"), 0) is None


def test_very_long_per_minute_delay_is_not_retried() -> None:
    assert retry_delay(quota_error(quota_id=PER_MINUTE, retry_delay_text="300s"), 0) is None


def test_suggested_delay_is_respected() -> None:
    delay = retry_delay(quota_error(quota_id=PER_MINUTE, retry_delay_text="27s"), 0)
    assert delay is not None
    assert delay >= 28.0


def test_rate_limit_without_details_uses_backoff() -> None:
    delay = retry_delay(quota_error(), 0)
    assert delay is not None
    assert 2.0 <= delay <= 3.0


def test_server_errors_use_backoff_and_client_errors_are_not_retried() -> None:
    server = retry_delay(quota_error(code=503), 1)
    assert server is not None
    assert 4.0 <= server <= 5.0
    assert retry_delay(quota_error(code=400), 0) is None


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1, "1 second"),
        (45, "45 seconds"),
        (60, "1 minute"),
        (120, "2 minutes"),
        (3600, "1 hour"),
        (7200, "2 hours"),
        (15254, "4 hours 14 minutes"),
    ],
)
def test_format_duration(seconds: int, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_describe_daily_quota() -> None:
    message, retry = describe_quota_error(
        "Language model", quota_error(quota_id=DAILY, retry_delay_text="15254s")
    )
    assert retry == 15254
    assert message.startswith("Language model quota exceeded.")
    assert "daily" in message
    assert "4 hours 14 minutes" in message


def test_describe_per_minute_and_unknown_quota() -> None:
    message, retry = describe_quota_error(
        "Embedding API", quota_error(quota_id=PER_MINUTE, retry_delay_text="27s")
    )
    assert retry == 27
    assert "Try again in about 27 seconds." in message

    message, retry = describe_quota_error("Embedding API", quota_error())
    assert retry is None
    assert "Please try again shortly." in message