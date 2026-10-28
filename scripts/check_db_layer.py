"""Smoke test: create, read and delete rows through the repositories."""

import asyncio
from datetime import date, time

from app.core.config import get_settings
from app.db.session import create_engine, create_session_factory
from app.models.enums import ChunkingStrategy, FileType
from app.repositories.booking_repository import BookingRepository
from app.repositories.document_repository import DocumentRepository


async def main() -> None:
    engine = create_engine(get_settings())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            documents = DocumentRepository(session)
            bookings = BookingRepository(session)

            doc = await documents.create(
                filename="test.pdf",
                file_type=FileType.PDF,
                file_size_bytes=1234,
                chunking_strategy=ChunkingStrategy.RECURSIVE,
                chunk_size=800,
                chunk_overlap=100,
            )
            print("Created document:", doc.id, doc.status.value)

            doc = await documents.mark_completed(doc, chunk_count=7)
            print("Updated:", doc.status.value, "chunks =", doc.chunk_count)

            booking = await bookings.create(
                session_id="test-session",
                name="Test User",
                email="test@example.com",
                interview_date=date(2026, 11, 5),
                interview_time=time(14, 30),
            )
            print("Created booking:", booking.id, booking.interview_date, booking.interview_time)

            print("Documents in DB:", len(await documents.list_all()))
            print("Bookings in DB:", len(await bookings.list_all()))

            await documents.delete(doc)
            await bookings.delete(booking)
            print("Cleaned up test rows. Database layer OK")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())