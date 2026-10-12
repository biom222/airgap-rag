from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Protocol, cast

from fastapi import Depends, Request
from opentelemetry.propagate import inject
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.api.dependencies.auth import get_current_principal
from airgap_rag.auth import Principal
from airgap_rag.core.config import Settings
from airgap_rag.db.repositories.documents import SQLAlchemyDocumentRepository
from airgap_rag.documents.service import DocumentService
from airgap_rag.documents.storage import LocalDocumentStore
from airgap_rag.documents.validation import DocumentValidator
from airgap_rag.metrics import MetricsRecorder


class SessionDatabase(Protocol):
    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...


async def get_document_service(
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> AsyncIterator[DocumentService]:
    settings = cast(Settings, request.app.state.settings)
    database = cast(SessionDatabase, request.app.state.database)
    metrics = cast(MetricsRecorder, request.app.state.metrics)
    trace_context: dict[str, str] = {}
    inject(trace_context)
    async with database.session() as session:
        yield DocumentService(
            repository=SQLAlchemyDocumentRepository(session),
            store=LocalDocumentStore(
                root=settings.document_storage_path,
                max_upload_bytes=settings.document_max_upload_bytes,
                read_chunk_bytes=settings.document_read_chunk_bytes,
            ),
            validator=DocumentValidator(settings.document_max_docx_uncompressed_bytes),
            metrics=metrics,
            owner_id=principal.subject,
            can_manage_all=principal.is_admin,
            trace_id=str(request.state.trace_id),
            trace_context=trace_context,
        )
