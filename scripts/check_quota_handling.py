"""Optional: show how an exhausted daily quota is reported (use a model whose limit is used up)."""

import asyncio
import sys
import time

from google import genai

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.rag.llm import LLMMessage, LLMService


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash"
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    service = LLMService(client, model, max_retries=settings.llm_max_retries)
    started = time.monotonic()
    try:
        text = await service.generate(
            system_instruction="Be brief.",
            messages=[LLMMessage(role="user", content="Reply with one word: ready")],
        )
        print(f"{model} replied {text!r}: its limit has reset, so nothing to demonstrate")
    except AppError as exc:
        print(f"Failed after {time.monotonic() - started:.1f}s with HTTP {exc.status_code}")
        print("Message:", exc.message)
        print("Retry-After:", exc.retry_after_seconds)
    finally:
        await client.aio.aclose()


if __name__ == "__main__":
    asyncio.run(main())