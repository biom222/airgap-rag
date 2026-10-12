from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status

from airgap_rag.api.dependencies.auth import get_current_principal
from airgap_rag.api.dependencies.documents import get_document_service
from airgap_rag.api.schemas.documents import DocumentResponse, DocumentUploadResponse
from airgap_rag.auth import Principal
from airgap_rag.documents.service import DocumentService
from airgap_rag.documents.types import DocumentMetadata

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_200_OK: {"model": DocumentUploadResponse},
        status.HTTP_413_CONTENT_TOO_LARGE: {},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {},
    },
)
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File(description="PDF, DOCX or UTF-8 TXT")],
    service: Annotated[DocumentService, Depends(get_document_service)],
    principal: Annotated[Principal, Depends(get_current_principal)],
    department: Annotated[str | None, Form(max_length=100)] = None,
    document_type: Annotated[str | None, Form(max_length=100)] = None,
    document_date: Annotated[date | None, Form()] = None,
    access_level: Annotated[int, Form(ge=0, le=100)] = 0,
    tags: Annotated[str | None, Form(max_length=1000)] = None,
) -> DocumentUploadResponse:
    if not principal.is_admin and access_level > principal.access_level:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Нельзя назначить документу уровень доступа выше собственного.",
        )
    normalized_tags = tuple(
        dict.fromkeys(tag.strip() for tag in (tags or "").split(",") if tag.strip())
    )
    if len(normalized_tags) > 20 or any(len(tag) > 100 for tag in normalized_tags):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Разрешено не более 20 тегов длиной до 100 символов.",
        )
    metadata = DocumentMetadata(
        department=department.strip() if department else None,
        document_type=document_type.strip() if document_type else None,
        document_date=document_date,
        access_level=access_level,
        tags=normalized_tags,
    )
    if metadata == DocumentMetadata():
        result = await service.upload(file)
    else:
        result = await service.upload(file, metadata)
    if result.deduplicated:
        response.status_code = status.HTTP_200_OK
    payload = DocumentResponse.model_validate(result.document).model_dump()
    return DocumentUploadResponse(
        **payload,
        job_id=result.job.id,
        deduplicated=result.deduplicated,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    document = await service.get(document_id)
    return DocumentResponse.model_validate(document)
