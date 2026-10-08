import re
from collections.abc import Sequence

from airgap_rag.reranking.base import RerankDocument, RerankResult

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class MockReranker:
    """Deterministic lexical reranker for tests and pipeline development."""

    async def rerank(
        self,
        query: str,
        documents: Sequence[RerankDocument],
    ) -> list[RerankResult]:
        query_tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
        results = []
        for document in documents:
            document_tokens = set(_TOKEN_PATTERN.findall(document.text.casefold()))
            overlap = len(query_tokens & document_tokens)
            score = overlap / len(query_tokens) if query_tokens else 0.0
            results.append(RerankResult(id=document.id, score=score))
        return results

    async def healthcheck(self) -> bool:
        return True
