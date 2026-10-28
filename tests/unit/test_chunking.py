"""Unit tests for the chunking strategies."""

import pytest

from app.models.enums import ChunkingStrategy
from app.rag.chunking.base import BaseChunker
from app.rag.chunking.factory import create_chunker
from app.rag.chunking.fixed import FixedSizeChunker
from app.rag.chunking.recursive import RecursiveChunker

SAMPLE_TEXT = (
    "Retrieval augmented generation combines search with text generation. "
    "The system first finds relevant passages. Then it passes them to a language model.\n\n"
    "Chunking decides how documents are split. Small chunks are precise but lose context. "
    "Large chunks keep context but blur the meaning.\n\n"
) * 5


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunks_respect_size_and_indexing(strategy: ChunkingStrategy) -> None:
    chunker = create_chunker(strategy, chunk_size=120, chunk_overlap=20)
    chunks = chunker.split(SAMPLE_TEXT)

    assert chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(0 < len(c.text) <= 120 for c in chunks)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_empty_text_gives_no_chunks(strategy: ChunkingStrategy) -> None:
    chunker = create_chunker(strategy, chunk_size=100, chunk_overlap=10)
    assert chunker.split("") == []
    assert chunker.split("   \n\n  ") == []


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_short_text_is_one_chunk(strategy: ChunkingStrategy) -> None:
    chunker = create_chunker(strategy, chunk_size=100, chunk_overlap=10)
    chunks = chunker.split("A short sentence.")
    assert [c.text for c in chunks] == ["A short sentence."]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_settings_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=size, chunk_overlap=overlap)


def test_fixed_chunk_boundaries() -> None:
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split("a" * 250)
    assert [len(c.text) for c in chunks] == [100, 100, 90]


def test_fixed_chunks_overlap() -> None:
    text = "0123456789" * 30
    chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=20).split(text)
    assert chunks[1].text[:20] == chunks[0].text[-20:]


def test_recursive_prefers_paragraph_boundaries() -> None:
    paragraphs = ["A" * 60, "B" * 60, "C" * 60]
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=0).split("\n\n".join(paragraphs))
    assert [c.text for c in chunks] == paragraphs


def test_recursive_never_cuts_words() -> None:
    chunks = RecursiveChunker(chunk_size=50, chunk_overlap=10).split("word " * 100)
    assert chunks
    assert all(set(c.text.split()) == {"word"} for c in chunks)


def test_recursive_hard_splits_text_without_separators() -> None:
    chunks = RecursiveChunker(chunk_size=100, chunk_overlap=0).split("x" * 250)
    assert [len(c.text) for c in chunks] == [100, 100, 50]


def test_recursive_keeps_sentence_endings() -> None:
    text = "First sentence is here. Second sentence is here. Third sentence is here."
    chunks = RecursiveChunker(chunk_size=30, chunk_overlap=0).split(text)
    assert all(c.text.endswith(".") for c in chunks)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (ChunkingStrategy.FIXED, FixedSizeChunker),
        (ChunkingStrategy.RECURSIVE, RecursiveChunker),
    ],
)
def test_factory_returns_correct_class(
    strategy: ChunkingStrategy, expected: type[BaseChunker]
) -> None:
    chunker = create_chunker(strategy, chunk_size=200, chunk_overlap=20)
    assert isinstance(chunker, expected)
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 20