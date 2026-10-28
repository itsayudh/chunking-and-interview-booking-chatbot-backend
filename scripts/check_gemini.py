"""Check the Gemini API key, list usable models, and optionally test two models.

Usage:
    python scripts/check_gemini.py                       # list models
    python scripts/check_gemini.py <llm_model> <embed_model>   # test both
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


def get_client() -> genai.Client:
    load_dotenv()
    api_key: str | None = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is missing in .env")
    return genai.Client(api_key=api_key)


def list_models(client: genai.Client) -> None:
    generation: list[str] = []
    embedding: list[str] = []
    for model in client.models.list():
        actions: list[str] = list(model.supported_actions or [])
        name: str = model.name or "unknown"
        if "generateContent" in actions:
            generation.append(name)
        if "embedContent" in actions:
            embedding.append(name)

    print("\n=== Models that support text generation ===")
    for name in sorted(generation):
        print(" ", name)
    print("\n=== Models that support embeddings ===")
    for name in sorted(embedding):
        print(" ", name)


def test_models(client: genai.Client, llm_model: str, embed_model: str) -> None:
    reply = client.models.generate_content(
        model=llm_model, contents="Reply with exactly one word: ready"
    )
    print(f"\nLLM ({llm_model}) replied: {reply.text}")

    result = client.models.embed_content(model=embed_model, contents="hello world")
    embeddings = result.embeddings or []
    if not embeddings or not embeddings[0].values:
        raise SystemExit("Embedding call returned no vector")
    print(f"Embedding ({embed_model}) dimension: {len(embeddings[0].values)}")


def main() -> None:
    client = get_client()
    if len(sys.argv) == 3:
        test_models(client, sys.argv[1], sys.argv[2])
    else:
        list_models(client)


if __name__ == "__main__":
    main()
