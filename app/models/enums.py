"""Enumerations shared by ORM models and API schemas."""

from enum import StrEnum


class FileType(StrEnum):
    """Supported upload file types."""

    PDF = "pdf"
    TXT = "txt"


class ChunkingStrategy(StrEnum):
    """Selectable chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"


class DocumentStatus(StrEnum):
    """Lifecycle status of a document during ingestion."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ChatIntent(StrEnum):
    """What a chat turn turned out to be."""

    QUESTION = "question"
    BOOKING = "booking"