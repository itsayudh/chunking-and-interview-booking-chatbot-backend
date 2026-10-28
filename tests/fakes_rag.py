"""Test doubles for the RAG components."""

import uuid
from collections.abc import Sequence
from typing import cast

from pydantic import BaseModel

from app.rag.llm import LLMMessage
from app.rag.vector_store import SearchHit


class FakeLLM:
    """Replays scripted replies (text, models, or errors) and records every call."""

    def __init__(self, responses: Sequence[str | BaseModel | Exception] = ()) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def _next(self) -> str | BaseModel | Exception:
        return self._responses.pop(0) if self._responses else "fake answer"

    async def generate(
        self,
        *,
        system_instruction: str,
        messages: Sequence[LLMMessage],
        temperature: float | None = None,
    ) -> str:
        self.calls.append(
            {"system": system_instruction, "messages": list(messages), "temperature": temperature}
        )
        outcome = self._next()
        if isinstance(outcome, Exception):
            raise outcome
        if not isinstance(outcome, str):
            raise TypeError("FakeLLM.generate() was given a structured response")
        return outcome

    async def generate_structured[T: BaseModel](
        self,
        *,
        system_instruction: str,
        messages: Sequence[LLMMessage],
        schema: type[T],
        temperature: float | None = None,
    ) -> T:
        self.calls.append(
            {
                "system": system_instruction,
                "messages": list(messages),
                "temperature": temperature,
                "schema": schema,
            }
        )
        outcome = self._next()
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return schema.model_validate_json(outcome)
        return cast(T, outcome)


class FakeRetriever:
    """Returns the same hits for every query and records the queries."""

    def __init__(self, hits: Sequence[SearchHit] = ()) -> None:
        self.hits = list(hits)
        self.queries: list[str] = []

    async def retrieve(
        self, query: str, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[SearchHit]:
        self.queries.append(query)
        return list(self.hits)


def make_hit(
    text: str = "Refunds take five business days.",
    filename: str = "policy.txt",
    chunk_index: int = 0,
    score: float = 0.8,
) -> SearchHit:
    return SearchHit(
        document_id=uuid.uuid4(),
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        score=score,
    )