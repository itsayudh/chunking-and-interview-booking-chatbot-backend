"""Application-specific exceptions. HTTP mapping is done by the handler in app/main.py."""

from typing import ClassVar


class AppError(Exception):
    """Base class for expected, user-facing errors."""

    status_code: ClassVar[int] = 500

    def __init__(self, message: str, *, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after_seconds = retry_after_seconds


class UnsupportedFileTypeError(AppError):
    """The uploaded file type is not .pdf or .txt."""

    status_code = 415


class FileTooLargeError(AppError):
    """The uploaded file exceeds the configured size limit."""

    status_code = 413


class DocumentParsingError(AppError):
    """The file could not be read (corrupt, encrypted, or wrong encoding)."""

    status_code = 422


class EmptyDocumentError(AppError):
    """No text could be extracted (empty file or scanned PDF without a text layer)."""

    status_code = 422


class EmbeddingError(AppError):
    """The embedding provider failed, was rate limited, or returned unusable data."""

    status_code = 502


class EmbeddingQuotaError(EmbeddingError):
    """The embedding provider's quota is exhausted."""

    status_code = 429


class VectorStoreError(AppError):
    """The vector database failed or is misconfigured."""

    status_code = 502


class DocumentNotFoundError(AppError):
    """No document exists with the requested id."""

    status_code = 404


class InvalidRequestError(AppError):
    """The request is well-formed but contains invalid values."""

    status_code = 422


class LLMError(AppError):
    """The language model failed, was rate limited, or returned no usable text."""

    status_code = 502


class LLMQuotaError(LLMError):
    """The language model's quota is exhausted."""

    status_code = 429


class SessionNotFoundError(AppError):
    """No stored conversation exists for the requested session id."""

    status_code = 404