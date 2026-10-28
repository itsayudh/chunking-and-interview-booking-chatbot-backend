"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from typing import Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings. Required fields without defaults must be in .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # App
    app_name: str = "rag-interview-backend"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    max_upload_mb: int = Field(default=10, gt=0)

    # Gemini
    gemini_api_key: SecretStr
    gemini_llm_model: str
    gemini_embedding_model: str
    embedding_dimension: int = Field(gt=0)

    # Qdrant
    qdrant_url: str
    qdrant_api_key: SecretStr
    qdrant_collection: str = "documents_chunks"

    # PostgreSQL / Redis (URLs contain passwords, so they are secrets)
    database_url: SecretStr
    redis_url: SecretStr
    chat_history_max_messages: int = Field(default=20, gt=0)
    chat_ttl_seconds: int = Field(default=3600, gt=0)

    # RAG
    default_chunk_size: int = Field(default=800, gt=0)
    default_chunk_overlap: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=4, gt=0)
    retrieval_min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    embedding_batch_size: int = Field(default=50, gt=0, le=100)
    embedding_max_retries: int = Field(default=5, ge=0)
    llm_max_retries: int = Field(default=5, ge=0)
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    # Booking
    booking_timezone: str = "UTC"

    @field_validator("booking_timezone")
    @classmethod
    def _check_booking_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Unknown timezone '{value}'. Use an IANA name such as 'UTC'."
            ) from exc
        return value

    @model_validator(mode="after")
    def _check_chunk_settings(self) -> Self:
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError("DEFAULT_CHUNK_OVERLAP must be smaller than DEFAULT_CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance (created once per process)."""
    return Settings()