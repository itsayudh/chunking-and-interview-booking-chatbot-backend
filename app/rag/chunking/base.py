"""Common interface for chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """One piece of a document, with its position."""

    index: int
    text: str


class BaseChunker(ABC):
    """Base class for chunkers. Sizes are measured in characters."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def _split_text(self, text: str) -> list[str]:
        """Split text into raw pieces. Implemented by each strategy."""

    def split(self, text: str) -> list[Chunk]:
        """Split text into indexed chunks, dropping empty pieces."""
        chunks: list[Chunk] = []
        for piece in self._split_text(text):
            cleaned = piece.strip()
            if cleaned:
                chunks.append(Chunk(index=len(chunks), text=cleaned))
        return chunks