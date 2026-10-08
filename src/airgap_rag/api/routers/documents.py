from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from airgap_rag.api.dependencies.documents import get_document_service
from airgap_rag.api.schemas.documents import DocumentResponse, DocumentUploadResponse
from airgap_rag.documents.service import DocumentService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
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
) -> DocumentUploadResponse:
    result = await service.upload(file)
    if result.deduplicated:
        response.status_code = status.HTTP_200_OK
    payload = DocumentResponse.model_validate(result.document).model_dump()
    return DocumentUploadResponse(**payload, deduplicated=result.deduplicated)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    document = await service.get(document_id)
    return DocumentResponse.model_validate(document)
