from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def healthcheck(self) -> bool: ...


class EmbeddingProviderError(RuntimeError):
    """The configured local embedding provider cannot serve embeddings."""
