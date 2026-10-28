"""How LLMService and EmbeddingService react to Gemini quota errors (clients are faked)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai

from app.core.exceptions import EmbeddingQuotaError, LLMQuotaError
from app.rag.embeddings import EmbeddingService
from app.rag.llm import LLMMessage, LLMService
from tests.fakes_quota import DAILY, PER_MINUTE, quota_error

MESSAGES = [LLMMessage(role="user", content="hello")]


class ScriptedModels:
    """Serves scripted outcomes for both generate_content and embed_content."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.calls = 0
        self._outcomes = list(outcomes)

    def _next(self) -> Any:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def generate_content(self, **_: Any) -> Any:
        return self._next()

    async def embed_content(self, **_: Any) -> Any:
        return self._next()


def client_of(models: ScriptedModels) -> genai.Client:
    return cast(genai.Client, SimpleNamespace(aio=SimpleNamespace(models=models)))


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr("app.rag.llm.asyncio.sleep", record)
    monkeypatch.setattr("app.rag.embeddings.asyncio.sleep", record)
    return waits


async def test_llm_daily_limit_fails_fast(slept: list[float]) -> None:
    models = ScriptedModels([quota_error(quota_id=DAILY, retry_delay_text="15254s")] * 5)
    service = LLMService(client_of(models), "m", max_retries=5)

    with pytest.raises(LLMQuotaError) as info:
        await service.generate(system_instruction="s", messages=MESSAGES)

    assert models.calls == 1
    assert slept == []
    assert info.value.status_code == 429
    assert info.value.retry_after_seconds == 15254
    assert "4 hours 14 minutes" in info.value.message


async def test_llm_waits_for_the_delay_google_suggests(slept: list[float]) -> None:
    models = ScriptedModels(
        [
            quota_error(quota_id=PER_MINUTE, retry_delay_text="27s"),
            SimpleNamespace(text="ok"),
        ]
    )
    service = LLMService(client_of(models), "m", max_retries=3)

    assert await service.generate(system_instruction="s", messages=MESSAGES) == "ok"
    assert slept == [28.0]


async def test_embedding_daily_limit_fails_fast(slept: list[float]) -> None:
    models = ScriptedModels([quota_error(quota_id=DAILY, retry_delay_text="3600s")] * 5)
    service = EmbeddingService(client_of(models), "m", dimension=4, max_retries=5)

    with pytest.raises(EmbeddingQuotaError) as info:
        await service.embed_query("a question")

    assert models.calls == 1
    assert slept == []
    assert info.value.retry_after_seconds == 3600