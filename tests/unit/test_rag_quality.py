from uuid import uuid4

import pytest
from pydantic import ValidationError

from airgap_rag.api.schemas.retrieval import RetrievalSearchRequest
from airgap_rag.chunking.recursive import RecursiveCharacterChunker
from airgap_rag.documents.parsers import ParsedBlock, ParsedDocument, ParsedPage
from airgap_rag.evaluation.generation import GenerationEvaluator
from airgap_rag.evaluation.models import GenerationEvaluationCase
from airgap_rag.rag.citations import CitationVerifier
from airgap_rag.retrieval.service import RetrievalService, RetrievedChunk


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=uuid4(),
        filename="policy.pdf",
        page=1,
        chunk_id=uuid4(),
        chunk_index=0,
        text=text,
        vector_score=0.9,
    )


def test_citation_verifier_rejects_unsupported_claim() -> None:
    verifier = CitationVerifier(0.3)
    chunks = (_chunk("The retention period is five years."),)

    assert verifier.verify("The retention period is five years [S1].", chunks).valid
    assert not verifier.verify("The retention period is twenty years [S1].", chunks).valid
    assert not verifier.verify("The retention period is five years.", chunks).valid


def test_structured_chunking_preserves_heading_and_table_type() -> None:
    document_id = uuid4()
    parsed = ParsedDocument(
        pages=(
            ParsedPage(
                page=1,
                text="Policy\nFive years",
                blocks=(
                    ParsedBlock(kind="heading", text="Policy", section="Policy"),
                    ParsedBlock(kind="table", text="| A | B |", section="Policy"),
                ),
            ),
        )
    )

    chunks = RecursiveCharacterChunker(100, 10).split(document_id, parsed)

    assert chunks[1].content_type == "table"
    assert chunks[1].section == "Policy"
    assert chunks[1].text.startswith("Policy\n")


def test_rrf_promotes_chunk_present_in_both_rankings() -> None:
    shared = uuid4()
    vector_only = uuid4()
    lexical_only = uuid4()

    ordered, scores = RetrievalService._fuse(
        [vector_only, shared],
        [lexical_only, shared],
    )

    assert ordered[0] == shared
    assert scores[shared] > scores[vector_only]


def test_retrieval_filters_validate_dates_and_normalize_tags() -> None:
    request = RetrievalSearchRequest(
        question="question",
        tags=[" policy ", "policy", "legal"],
    )

    assert request.tags == ["policy", "legal"]
    with pytest.raises(ValidationError, match="date_from"):
        RetrievalSearchRequest(
            question="question",
            date_from="2026-12-01",
            date_to="2026-01-01",
        )


class _GenerationClient:
    def __init__(self, answer: str, chunk_id) -> None:  # type: ignore[no-untyped-def]
        self.answer_text = answer
        self.chunk_id = chunk_id

    async def answer(self, question: str, *, document_ids):  # type: ignore[no-untyped-def]
        del question, document_ids
        return self.answer_text, (self.chunk_id,)


@pytest.mark.asyncio
async def test_generation_evaluation_enforces_regression_thresholds() -> None:
    chunk_id = uuid4()
    evaluator = GenerationEvaluator(_GenerationClient("Retention is five years [S1].", chunk_id))
    report = await evaluator.evaluate(
        (
            GenerationEvaluationCase(
                case_id="retention",
                question="How long?",
                reference_answer="Retention is five years.",
                relevant_chunk_ids=(chunk_id,),
            ),
        ),
        dataset_name="generation.jsonl",
        base_url="http://test",
        thresholds={
            "answer_f1": 0.7,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "faithfulness": 1.0,
        },
    )

    assert report.passed
    assert report.citation_precision == 1.0
