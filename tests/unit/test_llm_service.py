"""Unit tests for LLMService (the Gemini client is faked, so no network or quota)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai
from google.genai import errors

from app.core.exceptions import LLMError
from app.rag.llm import LLMMessage, LLMService


class FakeModels:
    def __init__(self, outcomes: list[Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcomes = list(outcomes or [])

    async def generate_content(self, *, model: str, contents: list[Any], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        outcome = self._outcomes.pop(0) if self._outcomes else SimpleNamespace(text="ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_service(models: FakeModels, max_retries: int = 3) -> LLMService:
    client = cast(genai.Client, SimpleNamespace(aio=SimpleNamespace(models=models)))
    return LLMService(client=client, model="test-model", max_retries=max_retries)


def api_error(code: int) -> errors.APIError:
    return errors.ClientError(code, {"error": {"message": "boom", "status": "TEST"}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("app.rag.llm.asyncio.sleep", instant)


async def test_generate_sends_system_prompt_and_roles() -> None:
    models = FakeModels([SimpleNamespace(text="  The answer.  ")])
    service = make_service(models)

    text = await service.generate(
        system_instruction="Be brief.",
        messages=[
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="model", content="Hi"),
            LLMMessage(role="user", content="Question"),
        ],
        temperature=0.3,
    )

    assert text == "The answer."
    call = models.calls[0]
    assert call["model"] == "test-model"
    assert call["config"].system_instruction == "Be brief."
    assert call["config"].temperature == 0.3
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    assert call["contents"][2].parts[0].text == "Question"


async def test_rate_limit_is_retried_then_succeeds() -> None:
    models = FakeModels([api_error(429), api_error(503)])
    text = await make_service(models).generate(
        system_instruction="s", messages=[LLMMessage(role="user", content="q")]
    )
    assert text == "ok"
    assert len(models.calls) == 3


async def test_gives_up_after_max_retries() -> None:
    models = FakeModels([api_error(429)] * 10)
    with pytest.raises(LLMError, match="quota"):
        await make_service(models, max_retries=2).generate(
            system_instruction="s", messages=[LLMMessage(role="user", content="q")]
        )
    assert len(models.calls) == 3


async def test_client_error_is_not_retried() -> None:
    models = FakeModels([api_error(400)])
    with pytest.raises(LLMError):
        await make_service(models).generate(
            system_instruction="s", messages=[LLMMessage(role="user", content="q")]
        )
    assert len(models.calls) == 1


@pytest.mark.parametrize("empty", [None, "", "   "])
async def test_empty_reply_raises(empty: str | None) -> None:
    service = make_service(FakeModels([SimpleNamespace(text=empty)]))
    with pytest.raises(LLMError, match="empty"):
        await service.generate(
            system_instruction="s", messages=[LLMMessage(role="user", content="q")]
        )


async def test_no_messages_raises() -> None:
    with pytest.raises(ValueError):
        await make_service(FakeModels()).generate(system_instruction="s", messages=[])


def test_negative_retries_raise() -> None:
    client = cast(genai.Client, SimpleNamespace())
    with pytest.raises(ValueError):
        LLMService(client=client, model="m", max_retries=-1)