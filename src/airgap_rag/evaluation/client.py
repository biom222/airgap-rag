from uuid import UUID

import httpx
from pydantic import BaseModel, ValidationError


class RetrievalApiError(RuntimeError):
    """The retrieval API could not return a valid ranked result."""


class _RetrievedChunk(BaseModel):
    chunk_id: UUID


class _RetrievalResponse(BaseModel):
    results: list[_RetrievedChunk]


class RetrievalApiClient:
    def __init__(self, client: httpx.AsyncClient, token: str | None = None) -> None:
        self._client = client
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def search(
        self,
        question: str,
        *,
        limit: int,
        document_ids: tuple[UUID, ...] | None,
    ) -> tuple[UUID, ...]:
        payload: dict[str, object] = {
            "question": question,
            "top_k": limit,
            "top_n": limit,
        }
        if document_ids is not None:
            payload["document_ids"] = [str(document_id) for document_id in document_ids]

        try:
            response = await self._client.post(
                "/api/v1/retrieval/search", json=payload, headers=self._headers
            )
            response.raise_for_status()
            parsed = _RetrievalResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as exc:
            raise RetrievalApiError(f"Retrieval request failed: {exc}") from exc
        return tuple(item.chunk_id for item in parsed.results)


class _ChatSource(BaseModel):
    chunk_id: UUID


class _ChatResponse(BaseModel):
    answer: str
    sources: list[_ChatSource]


class GenerationApiClient:
    def __init__(self, client: httpx.AsyncClient, token: str | None = None) -> None:
        self._client = client
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def answer(
        self,
        question: str,
        *,
        document_ids: tuple[UUID, ...] | None,
    ) -> tuple[str, tuple[UUID, ...]]:
        payload: dict[str, object] = {"question": question}
        if document_ids is not None:
            payload["document_ids"] = [str(value) for value in document_ids]
        try:
            response = await self._client.post("/api/v1/chat", json=payload, headers=self._headers)
            response.raise_for_status()
            parsed = _ChatResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as exc:
            raise RetrievalApiError(f"Generation request failed: {exc}") from exc
        return parsed.answer, tuple(source.chunk_id for source in parsed.sources)
