import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import ClassVar

from airgap_rag.documents.errors import (
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentError,
)

PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_MIME = "text/plain"


@dataclass(frozen=True, slots=True)
class ValidatedDocument:
    filename: str
    mime_type: str
    suffix: str


class DocumentValidator:
    _mime_by_suffix: ClassVar[dict[str, str]] = {
        ".pdf": PDF_MIME,
        ".docx": DOCX_MIME,
        ".txt": TEXT_MIME,
    }
    _generic_mime_types: ClassVar[set[str]] = {"", "application/octet-stream"}

    def __init__(self, max_docx_uncompressed_bytes: int) -> None:
        self._max_docx_uncompressed_bytes = max_docx_uncompressed_bytes

    def validate(
        self,
        path: Path,
        filename: str | None,
        supplied_mime_type: str | None,
    ) -> ValidatedDocument:
        safe_filename = self._validate_filename(filename)
        suffix = Path(safe_filename).suffix.lower()
        mime_type = self._mime_by_suffix.get(suffix)
        if mime_type is None:
            raise UnsupportedDocumentError("Поддерживаются только PDF, DOCX и TXT.")

        supplied = (supplied_mime_type or "").lower()
        if supplied not in self._generic_mime_types and supplied != mime_type:
            raise UnsupportedDocumentError("MIME-тип не соответствует расширению файла.")

        if suffix == ".pdf":
            self._validate_pdf(path)
        elif suffix == ".docx":
            self._validate_docx(path)
        else:
            self._validate_text(path)

        return ValidatedDocument(safe_filename, mime_type, suffix)

    @staticmethod
    def _validate_filename(filename: str | None) -> str:
        if filename is None or not filename.strip():
            raise InvalidDocumentError("Имя файла отсутствует.")
        if len(filename) > 255:
            raise InvalidDocumentError("Имя файла слишком длинное.")
        if PurePosixPath(filename).name != filename or PureWindowsPath(filename).name != filename:
            raise InvalidDocumentError("Имя файла не должно содержать путь.")
        if any(ord(character) < 32 for character in filename):
            raise InvalidDocumentError("Имя файла содержит недопустимые символы.")
        return filename

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        with path.open("rb") as document:
            if document.read(5) != b"%PDF-":
                raise InvalidDocumentError("Файл не содержит корректную сигнатуру PDF.")

    def _validate_docx(self, path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                required = {"[Content_Types].xml", "word/document.xml"}
                if not required.issubset(names):
                    raise InvalidDocumentError("Файл не является корректным DOCX.")
                total_size = 0
                for member in archive.infolist():
                    if member.flag_bits & 0x1:
                        raise InvalidDocumentError("Зашифрованные DOCX не поддерживаются.")
                    total_size += member.file_size
                    if total_size > self._max_docx_uncompressed_bytes:
                        raise DocumentTooLargeError(self._max_docx_uncompressed_bytes)
        except zipfile.BadZipFile as exc:
            raise InvalidDocumentError("Файл не является корректным DOCX.") from exc

    @staticmethod
    def _validate_text(path: Path) -> None:
        try:
            content = path.read_bytes()
            if b"\x00" in content:
                raise UnicodeDecodeError("utf-8", content, 0, 1, "NUL byte")
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("TXT должен быть текстом в кодировке UTF-8.") from exc
