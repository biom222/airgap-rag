import json
from collections.abc import AsyncIterator

import httpx
import pytest

from airgap_rag.core.config import Settings
from airgap_rag.llm.base import (
    LLMMessage,
    LLMModelNotFoundError,
    LLMResponseError,
    LLMUnavailableError,
)
from airgap_rag.llm.factory import create_llm_provider
from airgap_rag.llm.mock import MockLLMProvider
from airgap_rag.llm.ollama import OllamaProvider


def make_provider(handler: httpx.AsyncBaseTransport) -> OllamaProvider:
    client = httpx.AsyncClient(transport=handler, base_url="http://ollama:11434")
    return OllamaProvider(
        base_url="http://unused:11434",
        model="test-model:latest",
        timeout_seconds=5,
        temperature=0.2,
        max_tokens=128,
        keep_alive="5m",
        client=client,
    )


async def collect(stream: AsyncIterator[str]) -> list[str]:
    return [part async for part in stream]


async def test_mock_provider_is_deterministic() -> None:
    provider = MockLLMProvider("fixed response")
    messages = [LLMMessage(role="user", content="question")]

    assert await provider.generate(messages) == "fixed response"
    assert await collect(provider.stream(messages)) == ["fixed response"]
    assert await provider.healthcheck() is True


async def test_ollama_generate_maps_messages_and_options() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload == {
            "model": "test-model:latest",
            "messages": [
                {"role": "system", "content": "Follow the context."},
                {"role": "user", "content": "Question"},
            ],
            "stream": False,
            "keep_alive": "5m",
            "options": {"temperature": 0.2, "num_predict": 128},
        }
        return httpx.Response(
            200,
            json={
                "model": "test-model:latest",
                "message": {"role": "assistant", "content": "Answer"},
                "done": True,
            },
        )

    provider = make_provider(httpx.MockTransport(handler))

    answer = await provider.generate(
        [
            LLMMessage(role="system", content="Follow the context."),
            LLMMessage(role="user", content="Question"),
        ]
    )

    assert answer == "Answer"


async def test_ollama_stream_yields_content_until_done() -> None:
    body = "\n".join(
        [
            json.dumps({"message": {"role": "assistant", "content": "Part 1"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": " Part 2"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
    )
    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, text=body)))

    parts = await collect(provider.stream([LLMMessage(role="user", content="Question")]))

    assert parts == ["Part 1", " Part 2"]


async def test_ollama_stream_rejects_missing_done_event() -> None:
    body = json.dumps({"message": {"role": "assistant", "content": "Partial"}, "done": False})
    provider = make_provider(httpx.MockTransport(lambda request: httpx.Response(200, text=body)))

    with pytest.raises(LLMResponseError, match="done event"):
        await collect(provider.stream([LLMMessage(role="user", content="Question")]))


async def test_ollama_generate_maps_timeout_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = make_provider(httpx.MockTransport(handler))

    with pytest.raises(LLMUnavailableError, match="timed out"):
        await provider.generate([LLMMessage(role="user", content="Question")])


async def test_ollama_generate_reports_missing_model() -> None:
    provider = make_provider(
        httpx.MockTransport(lambda request: httpx.Response(404, json={"error": "not found"}))
    )

    with pytest.raises(LLMModelNotFoundError, match="test-model:latest"):
        await provider.generate([LLMMessage(role="user", content="Question")])


async def test_ollama_generate_rejects_malformed_response() -> None:
    provider = make_provider(
        httpx.MockTransport(lambda request: httpx.Response(200, json={"done": True}))
    )

    with pytest.raises(LLMResponseError, match="invalid chat response"):
        await provider.generate([LLMMessage(role="user", content="Question")])


async def test_ollama_healthcheck_requires_configured_model() -> None:
    available = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"models": [{"name": "test-model:latest", "model": "test-model:latest"}]},
            )
        )
    )
    missing = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"models": [{"name": "another-model:latest"}]},
            )
        )
    )

    assert await available.healthcheck() is True
    assert await missing.healthcheck() is False


async def test_llm_factory_selects_configured_provider() -> None:
    mock = create_llm_provider(Settings(llm_provider="mock"))
    ollama = create_llm_provider(
        Settings(llm_provider="ollama", ollama_base_url="http://ollama:11434")
    )

    assert isinstance(mock, MockLLMProvider)
    assert isinstance(ollama, OllamaProvider)
    await mock.close()
    await ollama.close()
