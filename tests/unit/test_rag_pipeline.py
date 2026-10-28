"""Unit tests for prompts, query rewriter, retriever and the RAG pipeline."""

import uuid
from typing import cast

import pytest

from app.core.exceptions import LLMError
from app.memory.redis_memory import ChatMessage
from app.models.enums import ChunkingStrategy
from app.rag.chunking.base import Chunk
from app.rag.embeddings import EmbeddingService
from app.rag.llm import LLMService
from app.rag.pipeline import RagPipeline
from app.rag.prompts import REFUSAL_MESSAGE, format_context
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from tests.fakes import FakeEmbedder
from tests.fakes_rag import FakeLLM, FakeRetriever, make_hit

HISTORY = [
    ChatMessage(role="user", content="What is the refund policy?"),
    ChatMessage(role="assistant", content="Refunds are accepted within 30 days."),
]
REWRITTEN = "What is the refund policy for international orders?"


def llm_of(fake: FakeLLM) -> LLMService:
    return cast(LLMService, fake)


# ----- prompts -----


def test_format_context_numbers_chunks_and_names_sources() -> None:
    context = format_context(
        [make_hit("First text", "a.txt", 0), make_hit("Second text", "b.pdf", 4)]
    )
    assert "[1] (source: a.txt, part 1)\nFirst text" in context
    assert "[2] (source: b.pdf, part 5)\nSecond text" in context


# ----- query rewriter -----


async def test_rewriter_skips_llm_without_history() -> None:
    llm = FakeLLM()
    result = await QueryRewriter(llm_of(llm)).rewrite("  What is the refund policy?  ", [])
    assert result == "What is the refund policy?"
    assert llm.calls == []


async def test_rewriter_uses_history_and_returns_rewritten_question() -> None:
    llm = FakeLLM([REWRITTEN])
    result = await QueryRewriter(llm_of(llm)).rewrite("And for international orders?", HISTORY)

    assert result == REWRITTEN
    prompt = llm.calls[0]["messages"][0].content  # type: ignore[index]
    assert "What is the refund policy?" in prompt
    assert "Assistant: Refunds are accepted within 30 days." in prompt
    assert "And for international orders?" in prompt


async def test_rewriter_keeps_first_line_and_strips_quotes() -> None:
    llm = FakeLLM(['"What about shipping?"\nBecause the user asked about it.'])
    result = await QueryRewriter(llm_of(llm)).rewrite("and shipping?", HISTORY)
    assert result == "What about shipping?"


async def test_rewriter_falls_back_when_llm_fails() -> None:
    llm = FakeLLM([LLMError("down")])
    result = await QueryRewriter(llm_of(llm)).rewrite("And for international orders?", HISTORY)
    assert result == "And for international orders?"


async def test_rewriter_falls_back_on_unusable_output() -> None:
    llm = FakeLLM(["x" * 600])
    result = await QueryRewriter(llm_of(llm)).rewrite("And for international orders?", HISTORY)
    assert result == "And for international orders?"


async def test_rewriter_only_uses_recent_messages() -> None:
    history = [ChatMessage(role="user", content=f"message {n}") for n in range(1, 5)]
    llm = FakeLLM([REWRITTEN])
    await QueryRewriter(llm_of(llm), max_history_messages=2).rewrite("and then?", history)

    prompt = llm.calls[0]["messages"][0].content  # type: ignore[index]
    assert "message 3" in prompt
    assert "message 4" in prompt
    assert "message 1" not in prompt


async def test_rewriter_rejects_blank_question() -> None:
    with pytest.raises(ValueError):
        await QueryRewriter(llm_of(FakeLLM())).rewrite("   ", HISTORY)


# ----- retriever (real in-memory Qdrant, fake embedder) -----


async def test_retriever_applies_min_score_and_top_k(vector_store: VectorStore) -> None:
    # FakeEmbedder maps text of length 4 to [1, 0, 0, 0]
    await vector_store.upsert_chunks(
        document_id=uuid.uuid4(),
        filename="a.txt",
        strategy=ChunkingStrategy.FIXED,
        chunks=[Chunk(0, "close"), Chunk(1, "also close"), Chunk(2, "far")],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    )
    embedder = cast(EmbeddingService, FakeEmbedder())

    strict = Retriever(embedder, vector_store, top_k=5, min_score=0.5)
    assert [hit.text for hit in await strict.retrieve("abcd")] == ["close", "also close"]

    limited = Retriever(embedder, vector_store, top_k=1, min_score=0.5)
    assert [hit.text for hit in await limited.retrieve("abcd")] == ["close"]


def test_retriever_rejects_invalid_top_k(vector_store: VectorStore) -> None:
    embedder = cast(EmbeddingService, FakeEmbedder())
    with pytest.raises(ValueError):
        Retriever(embedder, vector_store, top_k=0, min_score=0.5)


# ----- pipeline -----


def make_pipeline(
    llm: FakeLLM, retriever: FakeRetriever, temperature: float | None = None
) -> RagPipeline:
    return RagPipeline(
        llm=llm_of(llm),
        rewriter=QueryRewriter(llm_of(llm)),
        retriever=cast(Retriever, retriever),
        temperature=temperature,
    )


async def test_no_relevant_chunks_refuses_without_calling_llm() -> None:
    llm = FakeLLM()
    result = await make_pipeline(llm, FakeRetriever([])).answer("Who won the world cup?", [])

    assert result.answer == REFUSAL_MESSAGE
    assert result.answered is False
    assert result.sources == []
    assert llm.calls == []


async def test_answer_uses_retrieved_context_and_returns_sources() -> None:
    hits = [make_hit("Refunds take five business days.", "policy.txt")]
    llm = FakeLLM(["Refunds take five business days [1]."])

    result = await make_pipeline(llm, FakeRetriever(hits), temperature=0.3).answer(
        "How long do refunds take?", []
    )

    assert result.answered is True
    assert result.answer == "Refunds take five business days [1]."
    assert result.sources == hits
    assert result.standalone_question == "How long do refunds take?"
    call = llm.calls[0]
    assert call["temperature"] == 0.3
    prompt = call["messages"][0].content  # type: ignore[index]
    assert "Refunds take five business days." in prompt
    assert "(source: policy.txt, part 1)" in prompt
    assert "Question: How long do refunds take?" in prompt


async def test_model_refusal_is_reported_as_unanswered() -> None:
    llm = FakeLLM([REFUSAL_MESSAGE])
    result = await make_pipeline(llm, FakeRetriever([make_hit()])).answer("Something odd?", [])

    assert result.answered is False
    assert result.answer == REFUSAL_MESSAGE
    assert result.sources == []


async def test_follow_up_is_rewritten_before_retrieval() -> None:
    llm = FakeLLM([REWRITTEN, "International orders take two weeks [1]."])
    retriever = FakeRetriever([make_hit("International orders may take two weeks.")])

    result = await make_pipeline(llm, retriever).answer("And for international orders?", HISTORY)

    assert retriever.queries == [REWRITTEN]
    assert result.standalone_question == REWRITTEN
    assert result.answered is True
    assert len(llm.calls) == 2  # one rewrite call, one answer call


async def test_llm_failure_while_answering_propagates() -> None:
    llm = FakeLLM([LLMError("down")])
    with pytest.raises(LLMError):
        await make_pipeline(llm, FakeRetriever([make_hit()])).answer("Question?", [])