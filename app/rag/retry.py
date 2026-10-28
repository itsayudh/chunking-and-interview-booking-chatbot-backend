"""Shared retry and quota helpers for Gemini API calls."""

import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from google.genai import errors

logger = logging.getLogger(__name__)

_BASE_DELAY_SECONDS = 2.0
_MAX_DELAY_SECONDS = 60.0
_RETRY_DELAY_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)s$")


@dataclass(frozen=True, slots=True)
class QuotaInfo:
    """What Google's 429 response says about the quota that was exceeded."""

    quota_id: str | None
    is_daily: bool
    retry_after_seconds: float | None


def is_retryable(exc: errors.APIError) -> bool:
    """Rate limits (429) and server errors (5xx) can be temporary; other errors are not."""
    code = exc.code
    return code == 429 or (isinstance(code, int) and code >= 500)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: about 2s, 4s, 8s... capped at 60s."""
    delay: float = min(_MAX_DELAY_SECONDS, _BASE_DELAY_SECONDS * 2.0**attempt)
    return delay + random.uniform(0.0, 1.0)


def _detail_entries(exc: errors.APIError) -> list[dict[str, Any]]:
    payload: Any = exc.details
    error = payload.get("error") if isinstance(payload, dict) else None
    entries = error.get("details") if isinstance(error, dict) else None
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def parse_quota_error(exc: errors.APIError) -> QuotaInfo:
    """Extract the quota id and suggested retry delay from a 429 response, if present."""
    quota_ids: list[str] = []
    retry_after: float | None = None

    for entry in _detail_entries(exc):
        entry_type = str(entry.get("@type", ""))
        if entry_type.endswith("QuotaFailure"):
            violations = entry.get("violations")
            if isinstance(violations, list):
                quota_ids.extend(
                    str(violation["quotaId"])
                    for violation in violations
                    if isinstance(violation, dict) and violation.get("quotaId")
                )
        elif entry_type.endswith("RetryInfo"):
            match = _RETRY_DELAY_PATTERN.fullmatch(str(entry.get("retryDelay", "")))
            if match:
                retry_after = float(match.group(1))

    is_daily = any("perday" in quota_id.lower() for quota_id in quota_ids)
    return QuotaInfo(
        quota_id=quota_ids[0] if quota_ids else None,
        is_daily=is_daily,
        retry_after_seconds=retry_after,
    )


def retry_delay(exc: errors.APIError, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when retrying cannot help."""
    if not is_retryable(exc):
        return None
    if exc.code != 429:
        return backoff_delay(attempt)

    info = parse_quota_error(exc)
    if info.is_daily:
        return None  # a daily limit will not reset within a retry window
    suggested = info.retry_after_seconds
    if suggested is not None and suggested > _MAX_DELAY_SECONDS:
        return None
    return max(backoff_delay(attempt), (suggested or 0.0) + 1.0)


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}" + ("" if count == 1 else "s")


def format_duration(seconds: float) -> str:
    """Human-readable duration, for example '4 hours 14 minutes' or '27 seconds'."""
    total = max(1, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        text = _plural(hours, "hour")
        return f"{text} {_plural(minutes, 'minute')}" if minutes else text
    if minutes:
        return _plural(minutes, "minute")
    return _plural(secs, "second")


def describe_quota_error(label: str, exc: errors.APIError) -> tuple[str, int | None]:
    """Build a user-facing message and a Retry-After value (seconds) for a 429."""
    info = parse_quota_error(exc)
    retry = None if info.retry_after_seconds is None else math.ceil(info.retry_after_seconds)
    logger.warning(
        "%s quota error: quota_id=%s daily=%s retry_after=%s",
        label,
        info.quota_id,
        info.is_daily,
        retry,
    )

    if info.is_daily:
        hint = "The daily free-tier limit is used up"
        hint += f"; try again in about {format_duration(retry)}." if retry else "."
    elif retry:
        hint = f"Try again in about {format_duration(retry)}."
    else:
        hint = "Please try again shortly."
    return f"{label} quota exceeded. {hint}", retry