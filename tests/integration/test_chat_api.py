"""HTTP-level tests for the chat endpoints (service dependencies are faked)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import pytest_asyncio
from fakeredis import FakeAsyncRedis

from app.api.deps import get_chat_service
from app.core.exceptions import EmbeddingError
from app.main import create_app
from app.services.chat_service import ChatService
from tests.fakes_booking import FakeBookingRepository
from tests.fakes_chat import QUESTION, FakePipeline, booking_extraction, build_chat_service
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


async def test_question_history_and_delete_flow(redis_client: FakeAsyncRedis) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION]),
        pipeline=FakePipeline(),
        repository=FakeBookingRepository(),
    )

    async with api_client(service) as client:
        response = await client.post("/api/v1/chat", json={"message": "How long do refunds take?"})

        assert response.status_code == 200
        body = response.json()
        assert len(body["session_id"]) == 32
        assert body["intent"] == "question"
        assert body["answered"] is True
        assert body["booking"] is None
        assert body["booking_status"] is None
        assert body["sources"][0]["filename"] == "policy.txt"
        session_id = body["session_id"]

        history = (await client.get(f"/api/v1/chat/{session_id}/history")).json()
        assert [m["role"] for m in history["messages"]] == ["user", "assistant"]

        assert (await client.delete(f"/api/v1/chat/{session_id}")).status_code == 204
        assert (await client.get(f"/api/v1/chat/{session_id}/history")).status_code == 404


async def test_booking_through_the_api(redis_client: FakeAsyncRedis) -> None:
    llm = FakeLLM(
        [
            booking_extraction(),
            booking_extraction(
                name="Asha Karki", email="asha@example.com", date="2026-11-05", time="14:30"
            ),
        ]
    )
    service = build_chat_service(
        redis_client=redis_client,
        llm=llm,
        pipeline=FakePipeline(),
        repository=FakeBookingRepository(),
    )

    async with api_client(service) as client:
        first = await client.post(
            "/api/v1/chat", json={"session_id": "book-1", "message": "I want to book an interview"}
        )
        assert first.json()["booking_status"] == "needs_info"

        second = await client.post(
            "/api/v1/chat", json={"session_id": "book-1", "message": "Asha Karki, details above"}
        )

    body = second.json()
    assert body["intent"] == "booking"
    assert body["booking_status"] == "confirmed"
    assert body["sources"] == []
    assert body["booking"]["name"] == "Asha Karki"
    assert body["booking"]["interview_date"] == "2026-11-05"
    assert body["booking"]["interview_time"] == "14:30:00"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": "hello", "session_id": "bad id!"},
        {"message": "hello", "session_id": "x" * 65},
    ],
)
async def test_invalid_requests_return_422(
    redis_client: FakeAsyncRedis, payload: dict[str, str]
) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM(),
        pipeline=FakePipeline(),
        repository=FakeBookingRepository(),
    )
    async with api_client(service) as client:
        response = await client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422


async def test_upstream_failure_returns_502_with_message(redis_client: FakeAsyncRedis) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM([QUESTION]),
        pipeline=FakePipeline([EmbeddingError("quota exceeded")]),
        repository=FakeBookingRepository(),
    )
    async with api_client(service) as client:
        response = await client.post("/api/v1/chat", json={"message": "A question"})

    assert response.status_code == 502
    assert response.json()["detail"] == "quota exceeded"


async def test_history_of_unknown_session_returns_404(redis_client: FakeAsyncRedis) -> None:
    service = build_chat_service(
        redis_client=redis_client,
        llm=FakeLLM(),
        pipeline=FakePipeline(),
        repository=FakeBookingRepository(),
    )
    async with api_client(service) as client:
        assert (await client.get("/api/v1/chat/nobody/history")).status_code == 404