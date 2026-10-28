"""Text generation service backed by the Gemini API."""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from app.core.exceptions import LLMError, LLMQuotaError
from app.rag.retry import describe_quota_error, retry_delay

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMMessage:
    """One conversation turn sent to the model (Gemini calls the assistant role 'model')."""

    role: Literal["user", "model"]
    content: str


class LLMService:
    """Generates text or validated structured data, with retries on temporary failures."""

    def __init__(self, client: genai.Client, model: str, max_retries: int = 5) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._client = client
        self._model = model
        self._max_retries = max_retries

    async def generate(
        self,
        *,
        system_instruction: str,
        messages: Sequence[LLMMessage],
        temperature: float | None = None,
    ) -> str:
        """Return the model's text reply.

        Raises:
            LLMQuotaError: the quota is exhausted (HTTP 429).
            LLMError: the API failed or returned an empty reply.
        """
        response = await self._call(system_instruction, messages, temperature, None)
        return self._text_of(response)

    async def generate_structured[T: BaseModel](
        self,
        *,
        system_instruction: str,
        messages: Sequence[LLMMessage],
        schema: type[T],
        temperature: float | None = None,
    ) -> T:
        """Return the model's reply parsed into ``schema`` (Gemini JSON output mode).

        Raises:
            LLMQuotaError: the quota is exhausted (HTTP 429).
            LLMError: the API failed, or the reply did not match the schema.
        """
        response = await self._call(system_instruction, messages, temperature, schema)
        text = self._text_of(response)
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            logger.warning(
                "Structured output did not match %s (%d errors)", schema.__name__, exc.error_count()
            )
            raise LLMError("The model returned data in an unexpected format.") from exc

    async def _call(
        self,
        system_instruction: str,
        messages: Sequence[LLMMessage],
        temperature: float | None,
        schema: type[BaseModel] | None,
    ) -> types.GenerateContentResponse:
        """Call the API, retrying temporary failures and failing fast on exhausted quotas."""
        if not messages:
            raise ValueError("At least one message is required")

        contents: list[types.ContentUnion] = [
            types.Content(role=message.role, parts=[types.Part(text=message.content)])
            for message in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json" if schema is not None else None,
            response_schema=schema,
        )

        attempt = 0
        while True:
            try:
                return await self._client.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
            except errors.APIError as exc:
                delay = retry_delay(exc, attempt)
                if delay is None or attempt >= self._max_retries:
                    raise self._to_llm_error(exc) from exc
                logger.warning(
                    "LLM call failed with %s; retry %d/%d in %.1fs",
                    exc.code,
                    attempt + 1,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

    @staticmethod
    def _text_of(response: types.GenerateContentResponse) -> str:
        text = (response.text or "").strip()
        if not text:
            raise LLMError("The model returned an empty response (it may have been blocked).")
        return text

    @staticmethod
    def _to_llm_error(exc: errors.APIError) -> LLMError:
        if exc.code == 429:
            message, retry_after = describe_quota_error("Language model", exc)
            return LLMQuotaError(message, retry_after_seconds=retry_after)
        return LLMError(f"Language model request failed (HTTP {exc.code}).")