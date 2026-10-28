"""Verify the Redis connection with a ping and a write/read/delete test."""

import asyncio
import os

from dotenv import load_dotenv
from redis.asyncio import Redis


async def main() -> None:
    load_dotenv()
    url: str | None = os.getenv("REDIS_URL")
    if not url:
        raise SystemExit("REDIS_URL is missing in .env")

    client: Redis = Redis.from_url(url, decode_responses=True)
    try:
        await client.ping()
        await client.set("healthcheck:test", "ok", ex=30)
        value: str | None = await client.get("healthcheck:test")
        await client.delete("healthcheck:test")
        print("Redis connection OK")
        print("Write/read test:", value)
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
