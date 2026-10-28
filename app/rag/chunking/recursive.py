"""Recursive, structure-aware chunking."""

from collections.abc import Sequence

from app.rag.chunking.base import BaseChunker

# Tried in order: paragraph, line, sentence end, word, then raw characters.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker(BaseChunker):
    """Split on the largest natural boundary that keeps pieces under `chunk_size`."""

    def _split_text(self, text: str) -> list[str]:
        pieces = self._recursive_split(text, _SEPARATORS)
        return self._merge(pieces)

    def _recursive_split(self, text: str, separators: Sequence[str]) -> list[str]:
        """Break text into pieces no longer than chunk_size, using the best separator."""
        if len(text) <= self.chunk_size:
            return [text]

        for position, separator in enumerate(separators):
            if separator == "":
                size = self.chunk_size
                return [text[i : i + size] for i in range(0, len(text), size)]
            if separator in text:
                remaining = separators[position + 1 :]
                parts = text.split(separator)
                # Re-attach the separator so no characters are lost (e.g. the full stop).
                pieces = [part + separator for part in parts[:-1]] + [parts[-1]]
                result: list[str] = []
                for piece in pieces:
                    if piece:
                        result.extend(self._recursive_split(piece, remaining))
                return result
        return [text]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily combine small pieces into chunks, carrying overlap forward."""
        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for piece in pieces:
            if current and current_length + len(piece) > self.chunk_size:
                chunks.append("".join(current))
                current, current_length = self._overlap_tail(current)
                if current_length + len(piece) > self.chunk_size:
                    current, current_length = [], 0
            current.append(piece)
            current_length += len(piece)

        if current:
            chunks.append("".join(current))
        return chunks

    def _overlap_tail(self, pieces: list[str]) -> tuple[list[str], int]:
        """Return the trailing pieces of a finished chunk that fit in the overlap budget."""
        tail: list[str] = []
        length = 0
        for piece in reversed(pieces):
            if length + len(piece) > self.chunk_overlap:
                break
            tail.insert(0, piece)
            length += len(piece)
        return tail, length