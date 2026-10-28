"""Unit tests for LLMService.generate_structured (the Gemini client is faked)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai
from google.genai import errors
from pydantic import BaseModel

from app.core.exceptions import LLMError
from app.rag.llm import LLMMessage, LLMService


class Answer(BaseModel):
    city: str
    population: int | None


class FakeModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcomes = list(outcomes)

    async def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_service(models: FakeModels) -> LLMService:
    client = cast(genai.Client, SimpleNamespace(aio=SimpleNamespace(models=models)))
    return LLMService(client=client, model="test-model", max_retries=3)


def reply(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


MESSAGES = [LLMMessage(role="user", content="Where is the Eiffel Tower?")]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("app.rag.llm.asyncio.sleep", instant)


async def test_structured_reply_is_parsed_and_json_mode_requested() -> None:
    models = FakeModels([reply('{"city": "Paris", "population": null}')])

    result = await make_service(models).generate_structured(
        system_instruction="s", messages=MESSAGES, schema=Answer
    )

    assert result == Answer(city="Paris", population=None)
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is Answer


async def test_plain_generate_does_not_request_json() -> None:
    models = FakeModels([reply("Paris")])
    await make_service(models).generate(system_instruction="s", messages=MESSAGES)
    assert models.calls[0]["config"].response_mime_type is None


@pytest.mark.parametrize("bad", ["not json at all", '{"population": 3}', '{"city": 5}'])
async def test_reply_not_matching_schema_raises(bad: str) -> None:
    with pytest.raises(LLMError, match="unexpected format"):
        await make_service(FakeModels([reply(bad)])).generate_structured(
            system_instruction="s", messages=MESSAGES, schema=Answer
        )


async def test_structured_call_is_retried_on_rate_limit() -> None:
    limited = errors.ClientError(429, {"error": {"message": "slow down", "status": "TEST"}})
    models = FakeModels([limited, reply('{"city": "Paris", "population": 2}')])

    result = await make_service(models).generate_structured(
        system_instruction="s", messages=MESSAGES, schema=Answer
    )

    assert result.population == 2
    assert len(models.calls) == 2