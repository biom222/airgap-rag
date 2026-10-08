from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

LLMRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: LLMRole
    content: str


class LLMProvider(Protocol):
    async def generate(self, messages: Sequence[LLMMessage]) -> str: ...

    def stream(self, messages: Sequence[LLMMessage]) -> AsyncIterator[str]: ...

    async def healthcheck(self) -> bool: ...

    async def close(self) -> None: ...


class LLMProviderError(RuntimeError):
    """The configured LLM provider could not complete a valid generation."""


class LLMUnavailableError(LLMProviderError):
    """The local inference runtime is temporarily unavailable."""


class LLMModelNotFoundError(LLMProviderError):
    """The configured local model is not installed in the runtime."""


class LLMResponseError(LLMProviderError):
    """The inference runtime returned an invalid or incomplete response."""
