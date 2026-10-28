"""Text extraction for supported file types."""

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.models.enums import FileType

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


def detect_file_type(filename: str) -> FileType:
    """Return the FileType for a filename, based on its extension."""
    extension = Path(filename).suffix.lower().lstrip(".")
    try:
        return FileType(extension)
    except ValueError as exc:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '.{extension}'. Only .pdf and .txt are allowed."
        ) from exc


def normalize_text(text: str) -> str:
    """Clean extracted text: unify newlines, drop null bytes, collapse blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _TRAILING_SPACES.sub("\n", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _extract_pdf_text(content: bytes) -> str:
    """Extract text from all pages of a PDF."""
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:  # parsers raise many unrelated types on malformed input
        raise DocumentParsingError("The PDF file is corrupt or unreadable.") from exc

    if reader.is_encrypted:
        raise DocumentParsingError("Password-protected PDFs are not supported.")

    try:
        pages: list[str] = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise DocumentParsingError("Failed to extract text from the PDF.") from exc
    return "\n\n".join(pages)


def _extract_txt_text(content: bytes) -> str:
    """Decode a text file as UTF-8 (with optional BOM)."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParsingError("Text files must be UTF-8 encoded.") from exc


def extract_text(content: bytes, file_type: FileType) -> str:
    """Extract and normalize text from file bytes.

    Raises:
        DocumentParsingError: the file cannot be read.
        EmptyDocumentError: the file contains no extractable text.
    """
    raw = _extract_pdf_text(content) if file_type is FileType.PDF else _extract_txt_text(content)
    text = normalize_text(raw)
    if not text:
        raise EmptyDocumentError(
            "No text could be extracted. The file may be empty or a scanned document."
        )
    return text