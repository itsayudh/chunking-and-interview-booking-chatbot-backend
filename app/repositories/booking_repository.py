"""Database access for interview bookings."""

import uuid
from collections.abc import Sequence
from datetime import date, time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.booking import InterviewBooking


class BookingRepository:
    """CRUD operations for the interview_bookings table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        session_id: str,
        name: str,
        email: str,
        interview_date: date,
        interview_time: time,
    ) -> InterviewBooking:
        """Insert a confirmed booking."""
        booking = InterviewBooking(
            session_id=session_id,
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )
        self._session.add(booking)
        await self._session.commit()
        await self._session.refresh(booking)
        return booking

    async def get(self, booking_id: uuid.UUID) -> InterviewBooking | None:
        """Return one booking or None."""
        return await self._session.get(InterviewBooking, booking_id)

    async def list_all(self, *, limit: int = 50, offset: int = 0) -> Sequence[InterviewBooking]:
        """Return bookings, newest first."""
        statement = (
            select(InterviewBooking)
            .order_by(InterviewBooking.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(statement)
        return result.scalars().all()

    async def delete(self, booking: InterviewBooking) -> None:
        """Delete a booking row."""
        await self._session.delete(booking)
        await self._session.commit()