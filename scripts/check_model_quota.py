"""Try one tiny request on each candidate model and report which still have quota."""

import asyncio

from google import genai
from google.genai import errors

from app.core.config import get_settings

CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]


async def probe(client: genai.Client, model: str) -> str:
    try:
        response = await client.aio.models.generate_content(
            model=model, contents="Reply with one word: ready"
        )
    except errors.APIError as exc:
        message = exc.message or ""
        detail = next((line.strip() for line in message.splitlines() if "limit:" in line), "")
        return f"FAILED  HTTP {exc.code}  {(detail or message)[:150]}"
    return f"OK      replied {(response.text or '').strip()[:30]!r}"


async def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    try:
        for model in CANDIDATES:
            print(f"{model:<28}{await probe(client, model)}")
    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())