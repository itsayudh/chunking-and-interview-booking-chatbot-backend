"""Real-Redis check of RedisMemory: history, sliding window, TTL and booking draft."""

import asyncio
import uuid

from redis.asyncio import Redis

from app.core.config import get_settings
from app.memory.redis_memory import BookingDraft, ChatMessage, RedisMemory


async def main() -> None:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True)
    memory = RedisMemory(redis, max_messages=6, ttl_seconds=60)
    session_id = f"check-{uuid.uuid4().hex[:8]}"

    try:
        for n in range(1, 6):  # 10 messages with a window of 6
            await memory.append_messages(
                session_id,
                [
                    ChatMessage(role="user", content=f"question {n}"),
                    ChatMessage(role="assistant", content=f"answer {n}"),
                ],
            )
        history = await memory.get_history(session_id)
        print("History length (expected 6):", len(history))
        print("First / last:", history[0].content, "/", history[-1].content)
        print("TTL on chat key (expected 1-60):", await redis.ttl(f"chat:{session_id}"))

        draft = BookingDraft(name="Asha").merged_with({"email": "asha@example.com"})
        await memory.save_booking_draft(session_id, draft)
        loaded = await memory.get_booking_draft(session_id)
        missing = loaded.missing_fields() if loaded else None
        print("Draft missing fields (expected ['date', 'time']):", missing)
    finally:
        removed = await memory.clear_session(session_id)
        print("Cleaned up test session:", removed)
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())