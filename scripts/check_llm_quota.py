"""Show the raw response from Gemini for the configured LLM (no secrets are printed)."""

import asyncio

from google import genai
from google.genai import errors

from app.core.config import get_settings


async def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    print("Model:", settings.gemini_llm_model)
    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_llm_model, contents="Reply with one word: ready"
        )
        print("Call succeeded:", (response.text or "").strip())
    except errors.APIError as exc:
        print("HTTP code:", exc.code)
        print("Status:", exc.status)
        print("Message:", (exc.message or "")[:1500])
        print("Details:", str(exc.details)[:1500])
    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())