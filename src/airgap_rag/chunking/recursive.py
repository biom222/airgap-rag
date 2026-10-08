import hashlib
from uuid import UUID, uuid5

from airgap_rag.chunking.base import Chunk
from airgap_rag.documents.parsers import ParsedDocument


class RecursiveCharacterChunker:
    """Character-size chunker that prefers natural text boundaries."""

    _separators = ("\n\n", "\n", ". ", " ")

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, document_id: UUID, document: ParsedDocument) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for page in document.pages:
            for text in self._split_text(page.text):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        id=uuid5(document_id, f"{chunk_index}:{text_hash}"),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        page=page.page,
                        text=text,
                        text_hash=text_hash,
                    )
                )
        return tuple(chunks)

    def _split_text(self, text: str) -> tuple[str, ...]:
        normalized = text.strip()
        if not normalized:
            return ()

        parts: list[str] = []
        start = 0
        while start < len(normalized):
            hard_end = min(start + self._chunk_size, len(normalized))
            end = self._find_boundary(normalized, start, hard_end)
            part = normalized[start:end].strip()
            if part:
                parts.append(part)
            if end >= len(normalized):
                break
            start = max(end - self._chunk_overlap, start + 1)
        return tuple(parts)

    def _find_boundary(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)
        minimum = start + self._chunk_size // 2
        window = text[minimum:hard_end]
        for separator in self._separators:
            position = window.rfind(separator)
            if position >= 0:
                return minimum + position + len(separator)
        return hard_end
