"""Unit tests for text extraction."""

from io import BytesIO

import pytest
from pypdf import PdfWriter

from app.core.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.models.enums import FileType
from app.rag.loaders import detect_file_type, extract_text, normalize_text


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.pdf", FileType.PDF),
        ("NOTES.TXT", FileType.TXT),
        ("my.file.name.txt", FileType.TXT),
    ],
)
def test_detect_supported_types(filename: str, expected: FileType) -> None:
    assert detect_file_type(filename) is expected


@pytest.mark.parametrize("filename", ["image.png", "doc.docx", "noextension", "archive.pdf.exe"])
def test_detect_rejects_unsupported_types(filename: str) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        detect_file_type(filename)


def test_normalize_text() -> None:
    raw = "Line one  \r\n\r\n\r\n\r\nLine two\x00\r\n"
    assert normalize_text(raw) == "Line one\n\nLine two"


def test_extract_txt_text() -> None:
    assert extract_text(b"Hello\nWorld", FileType.TXT) == "Hello\nWorld"


def test_extract_txt_with_bom() -> None:
    assert extract_text(b"\xef\xbb\xbfHello", FileType.TXT) == "Hello"


def test_extract_txt_rejects_non_utf8() -> None:
    with pytest.raises(DocumentParsingError):
        extract_text(b"\xff\xfe\x00bad", FileType.TXT)


def test_empty_txt_raises() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text(b"   \n\n  ", FileType.TXT)


def test_corrupt_pdf_raises() -> None:
    with pytest.raises(DocumentParsingError):
        extract_text(b"this is not a pdf", FileType.PDF)


def test_pdf_without_text_raises() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(EmptyDocumentError):
        extract_text(buffer.getvalue(), FileType.PDF)