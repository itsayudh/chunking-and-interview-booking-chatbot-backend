"""Verify the Qdrant Cloud connection by listing collections."""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


def main() -> None:
    load_dotenv()
    url: str | None = os.getenv("QDRANT_URL")
    api_key: str | None = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise SystemExit("QDRANT_URL or QDRANT_API_KEY is missing in .env")

    client = QdrantClient(url=url, api_key=api_key, timeout=30)
    collections = client.get_collections().collections
    names: list[str] = [c.name for c in collections]
    print("Qdrant connection OK")
    print("Existing collections:", names if names else "(none yet)")


if __name__ == "__main__":
    main()
