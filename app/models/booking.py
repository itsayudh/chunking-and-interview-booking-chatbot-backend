"""ORM model for interview bookings."""

import uuid
from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, String, Time, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class InterviewBooking(Base):
    """One confirmed interview booking collected through the chat."""

    __tablename__ = "interview_bookings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(320))
    interview_date: Mapped[date] = mapped_column(Date)
    interview_time: Mapped[time] = mapped_column(Time)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )