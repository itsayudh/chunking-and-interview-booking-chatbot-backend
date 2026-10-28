"""Fixed-size sliding window chunking."""

from app.rag.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Cut the text every `chunk_size` characters, stepping back by `chunk_overlap`."""

    def _split_text(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        pieces: list[str] = []
        for start in range(0, len(text), step):
            end = start + self.chunk_size
            pieces.append(text[start:end])
            if end >= len(text):
                break
        return pieces