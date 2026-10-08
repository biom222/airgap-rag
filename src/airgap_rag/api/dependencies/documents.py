from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol, cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.core.config import Settings
from airgap_rag.db.repositories.documents import SQLAlchemyDocumentRepository
from airgap_rag.documents.service import DocumentService
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.validation import DocumentValidator


class SessionDatabase(Protocol):
    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...


async def get_document_service(request: Request) -> AsyncIterator[DocumentService]:
    settings = cast(Settings, request.app.state.settings)
    database = cast(SessionDatabase, request.app.state.database)
    async with database.session() as session:
        yield DocumentService(
            repository=SQLAlchemyDocumentRepository(session),
            store=LocalDocumentStore(
                root=settings.document_storage_path,
                max_upload_bytes=settings.document_max_upload_bytes,
                read_chunk_bytes=settings.document_read_chunk_bytes,
            ),
            validator=DocumentValidator(settings.document_max_docx_uncompressed_bytes),
        )
