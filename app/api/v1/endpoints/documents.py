"""Document ingestion endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from app.api.deps import IngestionServiceDep
from app.models.enums import ChunkingStrategy
from app.schemas.document import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    service: IngestionServiceDep,
    file: Annotated[UploadFile, File(description="A .pdf or .txt file")],
    chunking_strategy: Annotated[
        ChunkingStrategy, Form(description="'fixed' or 'recursive'")
    ] = ChunkingStrategy.RECURSIVE,
    chunk_size: Annotated[
        int | None, Form(gt=0, description="Characters per chunk (default from settings)")
    ] = None,
    chunk_overlap: Annotated[
        int | None, Form(ge=0, description="Overlap in characters (default from settings)")
    ] = None,
) -> DocumentResponse:
    """Upload a document, chunk it, embed it and store it."""
    # Read at most limit + 1 bytes so oversized files are rejected without loading them fully.
    content = await file.read(service.max_upload_bytes + 1)
    document = await service.ingest(
        filename=file.filename or "",
        content=content,
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return DocumentResponse.model_validate(document)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    service: IngestionServiceDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentResponse]:
    """List documents, newest first."""
    documents = await service.list_documents(limit=limit, offset=offset)
    return [DocumentResponse.model_validate(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: uuid.UUID, service: IngestionServiceDep) -> DocumentResponse:
    """Return one document's metadata."""
    return DocumentResponse.model_validate(await service.get_document(document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, service: IngestionServiceDep) -> None:
    """Delete a document and all of its vectors."""
    await service.delete_document(document_id)