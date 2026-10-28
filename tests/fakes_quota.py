"""Builders for Gemini-style quota errors, shaped like the real 429 response."""

from typing import Any

from google.genai import errors

DAILY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def quota_error(
    *, quota_id: str | None = None, retry_delay_text: str | None = None, code: int = 429
) -> errors.APIError:
    details: list[dict[str, Any]] = []
    if quota_id:
        details.append(
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": quota_id}],
            }
        )
    if retry_delay_text:
        details.append(
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay_text}
        )
    payload = {
        "error": {
            "code": code,
            "message": "quota",
            "status": "RESOURCE_EXHAUSTED",
            "details": details,
        }
    }
    if code >= 500:
        return errors.ServerError(code, payload)
    return errors.ClientError(code, payload)