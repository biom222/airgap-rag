from fastapi import status

from airgap_rag.core.exceptions import AppError


class InvalidDocumentError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_document", message, status.HTTP_400_BAD_REQUEST)


class UnsupportedDocumentError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__("unsupported_document", message, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)


class DocumentTooLargeError(AppError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(
            "document_too_large",
            f"Размер документа превышает допустимые {max_bytes} байт.",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )


class DocumentNotFoundError(AppError):
    def __init__(self) -> None:
        super().__init__("document_not_found", "Документ не найден.", status.HTTP_404_NOT_FOUND)


class DocumentParseError(Exception):
    """A supported document cannot be parsed safely."""
