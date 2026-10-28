"""Test double for the booking repository."""

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime, time

from app.models.booking import InterviewBooking


class FakeBookingRepository:
    """Same interface as BookingRepository, backed by a list. Can be told to fail."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.bookings: list[InterviewBooking] = []

    async def create(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
    ) -> InterviewBooking:
        if self.fail:
            raise RuntimeError("database down")
        booking = InterviewBooking(
            id=uuid.uuid4(),
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
            created_at=datetime.now(UTC),
        )
        self.bookings.append(booking)
        return booking

    async def list_all(self, *, limit: int = 50, offset: int = 0) -> Sequence[InterviewBooking]:
        ordered = sorted(self.bookings, key=lambda b: b.created_at, reverse=True)
        return ordered[offset : offset + limit]