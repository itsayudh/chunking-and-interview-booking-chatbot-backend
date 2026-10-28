"""HTTP-level test for the bookings list endpoint (the repository is faked)."""

from collections.abc import AsyncIterator
from datetime import date, time

import httpx
import pytest_asyncio

from app.api.deps import get_booking_repository
from app.main import create_app
from tests.fakes_booking import FakeBookingRepository


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    repository = FakeBookingRepository()
    await repository.create(
        session_id="s1",
        name="Asha Karki",
        email="asha@example.com",
        interview_date=date(2026, 11, 5),
        interview_time=time(14, 30),
    )
    app = create_app()
    app.dependency_overrides[get_booking_repository] = lambda: repository
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def test_list_bookings(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/bookings")

    assert response.status_code == 200
    (item,) = response.json()
    assert item["name"] == "Asha Karki"
    assert item["email"] == "asha@example.com"
    assert item["interview_date"] == "2026-11-05"
    assert item["interview_time"] == "14:30:00"
    assert item["session_id"] == "s1"


async def test_invalid_limit_is_rejected(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/v1/bookings?limit=0")).status_code == 422