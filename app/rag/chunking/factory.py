"""Create a chunker from a strategy name."""

from app.models.enums import ChunkingStrategy
from app.rag.chunking.base import BaseChunker
from app.rag.chunking.fixed import FixedSizeChunker
from app.rag.chunking.recursive import RecursiveChunker

_CHUNKERS: dict[ChunkingStrategy, type[BaseChunker]] = {
    ChunkingStrategy.FIXED: FixedSizeChunker,
    ChunkingStrategy.RECURSIVE: RecursiveChunker,
}


def create_chunker(
    strategy: ChunkingStrategy, chunk_size: int, chunk_overlap: int
) -> BaseChunker:
    """Return the chunker implementing the requested strategy."""
    return _CHUNKERS[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)