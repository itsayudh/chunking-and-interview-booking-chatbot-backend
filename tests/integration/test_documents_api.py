"""HTTP-level tests for the document endpoints (service dependencies are faked)."""

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest_asyncio
from fastapi import FastAPI

from app.api.deps import get_ingestion_service
from app.main import create_app
from app.services.ingestion_service import IngestionService

TEXT = b"Refunds are processed within five business days. " * 8


@pytest_asyncio.fixture
async def client(ingestion_service: IngestionService) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app()
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion_service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client


async def upload(
    client: httpx.AsyncClient,
    name: str = "policy.txt",
    content: bytes = TEXT,
    data: dict[str, str] | None = None,
) -> httpx.Response:
    return await client.post(
        "/api/v1/documents/upload",
        files={"file": (name, content, "application/octet-stream")},
        data=data or {},
    )


async def test_upload_returns_completed_document(client: httpx.AsyncClient) -> None:
    response = await upload(client, data={"chunking_strategy": "fixed", "chunk_size": "100"})

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["chunking_strategy"] == "fixed"
    assert body["chunk_size"] == 100
    assert body["chunk_count"] > 1
    assert body["filename"] == "policy.txt"


async def test_unsupported_type_returns_415(client: httpx.AsyncClient) -> None:
    response = await upload(client, name="image.png", content=b"data")
    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


async def test_oversized_file_returns_413(client: httpx.AsyncClient) -> None:
    response = await upload(client, content=b"x" * 2000)
    assert response.status_code == 413


async def test_empty_file_returns_422(client: httpx.AsyncClient) -> None:
    response = await upload(client, content=b"   ")
    assert response.status_code == 422


async def test_invalid_chunk_settings_return_422(client: httpx.AsyncClient) -> None:
    response = await upload(
        client, data={"chunking_strategy": "fixed", "chunk_size": "50", "chunk_overlap": "60"}
    )
    assert response.status_code == 422


async def test_unknown_strategy_returns_422(client: httpx.AsyncClient) -> None:
    response = await upload(client, data={"chunking_strategy": "semantic"})
    assert response.status_code == 422


async def test_get_list_and_delete_flow(client: httpx.AsyncClient) -> None:
    created = (await upload(client)).json()
    document_id = created["id"]

    assert (await client.get(f"/api/v1/documents/{document_id}")).json()["id"] == document_id
    listing = (await client.get("/api/v1/documents")).json()
    assert [item["id"] for item in listing] == [document_id]

    assert (await client.delete(f"/api/v1/documents/{document_id}")).status_code == 204
    assert (await client.get(f"/api/v1/documents/{document_id}")).status_code == 404


async def test_unknown_document_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_malformed_id_returns_422(client: httpx.AsyncClient) -> None:
    assert (await client.get("/api/v1/documents/not-a-uuid")).status_code == 422