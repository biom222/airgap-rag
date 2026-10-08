import hashlib
import math


class MockEmbeddingProvider:
    """Deterministic provider for tests and local pipeline development."""

    def __init__(self, dimension: int = 32) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    async def healthcheck(self) -> bool:
        return True

    def _embed(self, text: str) -> list[float]:
        source = text.encode("utf-8")
        values: list[float] = []
        counter = 0
        while len(values) < self._dimension:
            digest = hashlib.sha256(source + counter.to_bytes(4, "big")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        vector = values[: self._dimension]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]
