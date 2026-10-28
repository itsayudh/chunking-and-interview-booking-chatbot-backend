"""Interview booking endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_booking_repository
from app.repositories.booking_repository import BookingRepository
from app.schemas.booking import BookingResponse

router = APIRouter(prefix="/bookings", tags=["bookings"])


@router.get("", response_model=list[BookingResponse])
async def list_bookings(
    repository: Annotated[BookingRepository, Depends(get_booking_repository)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[BookingResponse]:
    """List stored interview bookings, newest first."""
    bookings = await repository.list_all(limit=limit, offset=offset)
    return [BookingResponse.model_validate(booking) for booking in bookings]