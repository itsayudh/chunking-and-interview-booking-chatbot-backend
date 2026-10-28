"""The API turns quota errors into HTTP 429 with a Retry-After header."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.api.deps import get_chat_service
from app.core.exceptions import EmbeddingError, LLMQuotaError
from app.main import create_app
from app.services.chat_service import ChatService
from tests.fakes_booking import FakeBookingRepository
from tests.fakes_chat import QUESTION, FakePipeline, build_chat_service
from tests.fakes_rag import FakeLLM


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@asynccontextmanager
async def api_client(service: ChatService) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()
    app.dependency_overrides[get_chat_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def service_failing_with(redis_client: FakeAsyncRedis, error: Exception) -> ChatService:
    return build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION]),
        pipeline=FakePipeline([error]),
        repository=FakeBookingRepository(),
    )


async def test_quota_error_returns_429_with_retry_after(redis_client: FakeAsyncRedis) -> None:
    error = LLMQuotaError(
        "Language model quota exceeded. Try again in about 2 minutes.",
        retry_after_seconds=120,
    )
    async with api_client(service_failing_with(redis_client, error)) as client:
        response = await client.post("/api/v1/chat", json={"message": "A question"})

    assert response.status_code == 429
    assert response.headers["retry-after"] == "120"
    assert "quota exceeded" in response.json()["detail"]


async def test_other_upstream_errors_have_no_retry_after(redis_client: FakeAsyncRedis) -> None:
    async with api_client(service_failing_with(redis_client, EmbeddingError("boom"))) as client:
        response = await client.post("/api/v1/chat", json={"message": "A question"})

    assert response.status_code == 502
    assert "retry-after" not in response.headers