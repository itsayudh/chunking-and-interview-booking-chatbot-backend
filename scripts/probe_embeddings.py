"""Probe how the configured embedding model behaves, so the wrapper is built on facts."""

import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

QUERY = "How do I get a refund for my order?"
RELEVANT = (
    "Customers can request a refund within 30 days "
    "by contacting support with the order number."
)
IRRELEVANT = "The mitochondria is the organelle that produces most of the cell's energy."


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def embed(
    client: genai.Client,
    model: str,
    texts: list[str],
    task_type: str | None = None,
) -> list[list[float]]:
    config = types.EmbedContentConfig(task_type=task_type) if task_type else None
    result = client.models.embed_content(model=model, contents=texts, config=config)
    return [list(e.values or []) for e in (result.embeddings or [])]


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_EMBEDDING_MODEL")
    if not api_key or not model:
        raise SystemExit("GEMINI_API_KEY or GEMINI_EMBEDDING_MODEL missing in .env")
    client = genai.Client(api_key=api_key)
    print(f"Model under test: {model}")

    # 1. Batching: N texts in, N vectors out?
    print("\n[1] Batch of 3 texts")
    vectors = embed(client, model, ["one", "two", "three"])
    print("  vectors returned:", len(vectors), "(expected 3)")
    if len(vectors) != 3:
        raise SystemExit("Batching does not return one vector per text. Stop here.")
    print("  dimension:", len(vectors[0]))
    print("  norm of first vector:", round(norm(vectors[0]), 4))

    # 2. Retrieval quality with and without task types
    print("\n[2] Retrieval quality (relevant should score clearly higher)")
    for label, doc_type, query_type in [
        ("with task types", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"),
        ("without task types", None, None),
    ]:
        try:
            docs = embed(client, model, [RELEVANT, IRRELEVANT], doc_type)
            query = embed(client, model, [QUERY], query_type)[0]
            print(f"  {label}:")
            print(f"    query vs relevant:   {cosine(query, docs[0]):.4f}")
            print(f"    query vs irrelevant: {cosine(query, docs[1]):.4f}")
        except Exception as exc:  # probe only: show whatever the API says
            print(f"  {label}: FAILED ->", type(exc).__name__, str(exc)[:200])

    # 3. Batch size limit
    print("\n[3] Larger batches")
    for size in (50, 100, 101):
        try:
            result = embed(client, model, [f"sample text number {i}" for i in range(size)])
            print(f"  batch of {size}: OK, got {len(result)} vectors")
        except Exception as exc:
            print(f"  batch of {size}: FAILED ->", type(exc).__name__, str(exc)[:200])


if __name__ == "__main__":
    main()