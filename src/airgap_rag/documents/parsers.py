from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import fitz
from docx import Document as DocxDocument

from airgap_rag.documents.errors import DocumentParseError
from airgap_rag.documents.validation import DOCX_MIME, PDF_MIME, TEXT_MIME


@dataclass(frozen=True, slots=True)
class ParsedPage:
    page: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    pages: tuple[ParsedPage, ...]


class DocumentParser(Protocol):
    mime_type: str

    def parse(self, path: Path) -> ParsedDocument: ...


class PdfParser:
    mime_type = PDF_MIME

    def parse(self, path: Path) -> ParsedDocument:
        try:
            with fitz.open(path) as document:
                if document.needs_pass:
                    raise DocumentParseError("Зашифрованные PDF не поддерживаются.")
                pages = tuple(
                    ParsedPage(page=index + 1, text=page.get_text("text"))
                    for index, page in enumerate(document)
                )
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError("Не удалось разобрать PDF.") from exc  # noqa: RUF001
        return ParsedDocument(pages=pages)


class DocxParser:
    mime_type = DOCX_MIME

    def parse(self, path: Path) -> ParsedDocument:
        try:
            document = DocxDocument(str(path))
            blocks = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    row_text = "\t".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        blocks.append(row_text)
        except Exception as exc:
            raise DocumentParseError("Не удалось разобрать DOCX.") from exc  # noqa: RUF001
        return ParsedDocument(pages=(ParsedPage(page=None, text="\n".join(blocks)),))


class TextParser:
    mime_type = TEXT_MIME

    def parse(self, path: Path) -> ParsedDocument:
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise DocumentParseError("Не удалось разобрать TXT.") from exc  # noqa: RUF001
        return ParsedDocument(pages=(ParsedPage(page=None, text=text),))


class ParserRegistry:
    def __init__(self, parsers: tuple[DocumentParser, ...] | None = None) -> None:
        parser_items = parsers or (PdfParser(), DocxParser(), TextParser())
        self._parsers = {parser.mime_type: parser for parser in parser_items}

    def get(self, mime_type: str) -> DocumentParser:
        try:
            return self._parsers[mime_type]
        except KeyError as exc:
            raise DocumentParseError(f"Нет parser для MIME-типа {mime_type}.") from exc
