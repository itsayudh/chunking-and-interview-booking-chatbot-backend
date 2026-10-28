"""Schemas for the health endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str
    app_name: str
    environment: str
