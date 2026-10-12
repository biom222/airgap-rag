# ruff: noqa: RUF001  # Russian stop words intentionally contain Cyrillic characters.

import re
from dataclasses import dataclass

from airgap_rag.retrieval.service import RetrievedChunk

_CITATION = re.compile(r"\[S(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN = re.compile(r"[\w-]+", re.UNICODE)
_STOP_WORDS = {
    "and",
    "are",
    "for",
    "from",
    "that",
    "the",
    "this",
    "with",
    "как",
    "для",
    "его",
    "или",
    "это",
    "что",
}
_NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "twenty",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "двадцать",
}


@dataclass(frozen=True, slots=True)
class CitationVerificationResult:
    valid: bool
    cited_indices: tuple[int, ...]


class CitationVerifier:
    def __init__(self, minimum_token_overlap: float = 0.2) -> None:
        if not 0 <= minimum_token_overlap <= 1:
            raise ValueError("minimum_token_overlap must be between 0 and 1")
        self._minimum_token_overlap = minimum_token_overlap

    def verify(
        self,
        answer: str,
        chunks: tuple[RetrievedChunk, ...],
    ) -> CitationVerificationResult:
        used: set[int] = set()
        for sentence in _SENTENCE.split(answer.strip()):
            claim = _CITATION.sub("", sentence).strip()
            claim_tokens = self._tokens(claim)
            references = tuple(int(value) for value in _CITATION.findall(sentence))
            if len(claim_tokens) < 3:
                continue
            if not references or any(index < 1 or index > len(chunks) for index in references):
                return CitationVerificationResult(False, ())
            source_tokens: set[str] = set()
            for index in references:
                used.add(index)
                source_tokens.update(self._tokens(chunks[index - 1].text))
            overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
            critical = {
                token for token in claim_tokens if token.isdigit() or token in _NUMBER_WORDS
            }
            if overlap < self._minimum_token_overlap or not critical.issubset(source_tokens):
                return CitationVerificationResult(False, ())
        return CitationVerificationResult(bool(used), tuple(sorted(used)))

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in (match.group(0).casefold() for match in _TOKEN.finditer(text))
            if len(token) > 1 and token not in _STOP_WORDS
        }
