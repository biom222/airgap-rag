from collections.abc import AsyncIterator, Sequence

from airgap_rag.llm.base import LLMMessage


class MockLLMProvider:
    """Deterministic provider for tests and development without a model runtime."""

    def __init__(self, response: str = "Mock LLM response.") -> None:
        self._response = response

    async def generate(self, messages: Sequence[LLMMessage]) -> str:
        del messages
        return self._response

    async def stream(self, messages: Sequence[LLMMessage]) -> AsyncIterator[str]:
        del messages
        if self._response:
            yield self._response

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        return None
