"""API schemas for documents."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import ChunkingStrategy, DocumentStatus, FileType


class DocumentResponse(BaseModel):
    """Metadata returned for an ingested document."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: FileType
    file_size_bytes: int
    chunking_strategy: ChunkingStrategy
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    status: DocumentStatus
    error_message: str | None
    created_at: datetime