"""Unit tests for the embedding service (no network: the Gemini client is faked)."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from google import genai
from google.genai import errors

from app.core.exceptions import EmbeddingError
from app.rag.embeddings import EmbeddingService

DIMENSION = 4


class FakeModels:
    """Stands in for client.aio.models, recording calls and replaying scripted outcomes."""

    def __init__(self, outcomes: list[Any] | None = None, dimension: int = DIMENSION) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcomes = list(outcomes or [])
        self._dimension = dimension

    async def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "task_type": config.task_type})
        if self._outcomes:
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        embeddings = [SimpleNamespace(values=[0.5] * self._dimension) for _ in contents]
        return SimpleNamespace(embeddings=embeddings)


def make_service(
    models: FakeModels, batch_size: int = 50, max_retries: int = 3
) -> EmbeddingService:
    client = cast(genai.Client, SimpleNamespace(aio=SimpleNamespace(models=models)))
    return EmbeddingService(
        client=client,
        model="test-model",
        dimension=DIMENSION,
        batch_size=batch_size,
        max_retries=max_retries,
    )


def api_error(code: int) -> errors.APIError:
    return errors.ClientError(code, {"error": {"message": "boom", "status": "TEST"}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("app.rag.embeddings.asyncio.sleep", instant)


async def test_documents_are_split_into_batches() -> None:
    models = FakeModels()
    service = make_service(models, batch_size=50)

    vectors = await service.embed_documents([f"text {i}" for i in range(120)])

    assert len(vectors) == 120
    assert [len(call["contents"]) for call in models.calls] == [50, 50, 20]
    assert {call["task_type"] for call in models.calls} == {"RETRIEVAL_DOCUMENT"}


async def test_empty_input_makes_no_api_call() -> None:
    models = FakeModels()
    assert await make_service(models).embed_documents([]) == []
    assert models.calls == []


async def test_query_uses_query_task_type() -> None:
    models = FakeModels()
    vector = await make_service(models).embed_query("  what is the refund policy?  ")

    assert len(vector) == DIMENSION
    assert models.calls[0]["task_type"] == "RETRIEVAL_QUERY"
    assert models.calls[0]["contents"] == ["what is the refund policy?"]


async def test_blank_query_is_rejected() -> None:
    with pytest.raises(ValueError):
        await make_service(FakeModels()).embed_query("   ")


async def test_rate_limit_is_retried_then_succeeds() -> None:
    models = FakeModels(outcomes=[api_error(429), api_error(429)])
    service = make_service(models, max_retries=3)

    vectors = await service.embed_documents(["only text"])

    assert len(vectors) == 1
    assert len(models.calls) == 3


async def test_rate_limit_gives_up_after_max_retries() -> None:
    models = FakeModels(outcomes=[api_error(429)] * 10)
    service = make_service(models, max_retries=2)

    with pytest.raises(EmbeddingError, match="quota"):
        await service.embed_documents(["text"])
    assert len(models.calls) == 3  # first attempt + 2 retries


async def test_client_error_is_not_retried() -> None:
    models = FakeModels(outcomes=[api_error(400)])
    service = make_service(models, max_retries=3)

    with pytest.raises(EmbeddingError):
        await service.embed_documents(["text"])
    assert len(models.calls) == 1


async def test_wrong_number_of_vectors_is_rejected() -> None:
    bad = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1] * DIMENSION)])
    service = make_service(FakeModels(outcomes=[bad]))

    with pytest.raises(EmbeddingError, match="Expected 2"):
        await service.embed_documents(["a", "b"])


async def test_wrong_dimension_is_rejected() -> None:
    service = make_service(FakeModels(dimension=DIMENSION + 1))

    with pytest.raises(EmbeddingError, match="dimension"):
        await service.embed_documents(["a"])


@pytest.mark.parametrize("size", [0, 101])
def test_invalid_batch_size_raises(size: int) -> None:
    client = cast(genai.Client, SimpleNamespace())
    with pytest.raises(ValueError):
        EmbeddingService(client=client, model="m", dimension=DIMENSION, batch_size=size)