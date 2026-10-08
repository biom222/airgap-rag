from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from airgap_rag.llm.base import (
    LLMMessage,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMResponseError,
    LLMUnavailableError,
)


class _OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["assistant"]
    content: str


class _OllamaChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _OllamaMessage
    done: bool


class _OllamaModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    model: str | None = None


class _OllamaTagsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[_OllamaModel]


class OllamaProvider:
    """Adapter for Ollama's local HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        temperature: float,
        max_tokens: int,
        keep_alive: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._keep_alive = keep_alive
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def generate(self, messages: Sequence[LLMMessage]) -> str:
        response = await self._post_chat(messages, stream=False)
        try:
            payload = _OllamaChatResponse.model_validate_json(response.content)
        except ValidationError as exc:
            raise LLMResponseError("Ollama returned an invalid chat response") from exc
        if not payload.done:
            raise LLMResponseError("Ollama returned an incomplete non-streaming response")
        return payload.message.content

    async def stream(self, messages: Sequence[LLMMessage]) -> AsyncIterator[str]:
        completed = False
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=self._chat_payload(messages, stream=True),
            ) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = _OllamaChatResponse.model_validate_json(line)
                    except ValidationError as exc:
                        raise LLMResponseError("Ollama returned an invalid stream event") from exc
                    if event.message.content:
                        yield event.message.content
                    if event.done:
                        completed = True
                        break
        except LLMProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Ollama streaming request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError("Ollama is unavailable") from exc

        if not completed:
            raise LLMResponseError("Ollama stream ended before the done event")

    async def healthcheck(self) -> bool:
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            payload = _OllamaTagsResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            return False
        return any(self._model in {item.name, item.model} for item in payload.models)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        stream: bool,
    ) -> httpx.Response:
        try:
            response = await self._client.post(
                "/api/chat",
                json=self._chat_payload(messages, stream=stream),
            )
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError("Ollama generation request timed out") from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError("Ollama is unavailable") from exc
        self._raise_for_status(response)
        return response

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == 404:
            raise LLMModelNotFoundError(f"Ollama model is not installed: {self._model}")
        if response.status_code >= 500:
            raise LLMUnavailableError(f"Ollama returned server error HTTP {response.status_code}")
        raise LLMProviderError(f"Ollama rejected the request with HTTP {response.status_code}")

    def _chat_payload(
        self,
        messages: Sequence[LLMMessage],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content} for message in messages
            ],
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
