"""Verify the PostgreSQL connection with a simple query."""

import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main() -> None:
    load_dotenv()
    url: str | None = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is missing in .env")

    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            print("PostgreSQL connection OK")
            print(result.scalar_one())
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
