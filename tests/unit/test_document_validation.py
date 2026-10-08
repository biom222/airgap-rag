import zipfile
from pathlib import Path

import pytest

from airgap_rag.documents.errors import (
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentError,
)
from airgap_rag.documents.validation import DOCX_MIME, PDF_MIME, TEXT_MIME, DocumentValidator


def test_accepts_utf8_text(runtime_path: Path) -> None:
    path = runtime_path / "upload"
    path.write_text("Привет, мир", encoding="utf-8")

    result = DocumentValidator(1024).validate(path, "notes.txt", "text/plain")

    assert result.mime_type == TEXT_MIME
    assert result.suffix == ".txt"


def test_rejects_filename_with_path(runtime_path: Path) -> None:
    path = runtime_path / "upload"
    path.write_text("text", encoding="utf-8")

    with pytest.raises(InvalidDocumentError, match="не должно содержать путь"):
        DocumentValidator(1024).validate(path, "../notes.txt", "text/plain")


def test_rejects_mime_mismatch(runtime_path: Path) -> None:
    path = runtime_path / "upload"
    path.write_bytes(b"%PDF-1.7")

    with pytest.raises(UnsupportedDocumentError, match="MIME-тип"):
        DocumentValidator(1024).validate(path, "report.pdf", "image/png")


def test_rejects_pdf_without_signature(runtime_path: Path) -> None:
    path = runtime_path / "upload"
    path.write_bytes(b"not a pdf")

    with pytest.raises(InvalidDocumentError, match="сигнатуру PDF"):
        DocumentValidator(1024).validate(path, "report.pdf", PDF_MIME)


def test_accepts_minimal_docx_container(runtime_path: Path) -> None:
    path = runtime_path / "upload"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<document />")

    result = DocumentValidator(1024).validate(path, "report.docx", DOCX_MIME)

    assert result.mime_type == DOCX_MIME


def test_rejects_docx_with_large_expanded_size(runtime_path: Path) -> None:
    path = runtime_path / "upload"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "x" * 2048)

    with pytest.raises(DocumentTooLargeError):
        DocumentValidator(1024).validate(path, "report.docx", DOCX_MIME)
