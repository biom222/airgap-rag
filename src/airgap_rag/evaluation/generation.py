import re
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from airgap_rag.evaluation.models import (
    GenerationCaseResult,
    GenerationEvaluationCase,
    GenerationEvaluationReport,
)

_TOKEN = re.compile(r"[\w-]+", re.UNICODE)
_CITATION = re.compile(r"\[S\d+\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class GenerationClient(Protocol):
    async def answer(
        self,
        question: str,
        *,
        document_ids: tuple[UUID, ...] | None,
    ) -> tuple[str, tuple[UUID, ...]]: ...


class GenerationEvaluator:
    def __init__(self, client: GenerationClient) -> None:
        self._client = client

    async def evaluate(
        self,
        cases: tuple[GenerationEvaluationCase, ...],
        *,
        dataset_name: str,
        base_url: str,
        thresholds: dict[str, float],
    ) -> GenerationEvaluationReport:
        if not cases:
            raise ValueError("Generation evaluation requires at least one case")
        results: list[GenerationCaseResult] = []
        for case in cases:
            answer, cited_ids = await self._client.answer(
                case.question,
                document_ids=case.document_ids,
            )
            expected = set(case.relevant_chunk_ids)
            cited = set(cited_ids)
            intersection = expected & cited
            results.append(
                GenerationCaseResult(
                    case_id=case.case_id,
                    answer=answer,
                    answer_f1=self._token_f1(answer, case.reference_answer),
                    citation_precision=len(intersection) / len(cited) if cited else 0.0,
                    citation_recall=len(intersection) / len(expected),
                    faithfulness=self._citation_coverage(answer),
                )
            )
        count = len(results)
        metrics = {
            "answer_f1": sum(item.answer_f1 for item in results) / count,
            "citation_precision": sum(item.citation_precision for item in results) / count,
            "citation_recall": sum(item.citation_recall for item in results) / count,
            "faithfulness": sum(item.faithfulness for item in results) / count,
        }
        return GenerationEvaluationReport(
            evaluated_at=datetime.now(UTC),
            dataset_name=dataset_name,
            base_url=base_url,
            case_count=count,
            **metrics,
            passed=all(metrics[name] >= minimum for name, minimum in thresholds.items()),
            thresholds=thresholds,
            cases=tuple(results),
        )

    @staticmethod
    def _token_f1(answer: str, reference: str) -> float:
        predicted = [match.group(0).casefold() for match in _TOKEN.finditer(answer)]
        expected = [match.group(0).casefold() for match in _TOKEN.finditer(reference)]
        if not predicted or not expected:
            return 0.0
        overlap = len(set(predicted) & set(expected))
        precision = overlap / len(set(predicted))
        recall = overlap / len(set(expected))
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    @staticmethod
    def _citation_coverage(answer: str) -> float:
        claims = [part for part in _SENTENCE.split(answer) if len(_TOKEN.findall(part)) >= 3]
        if not claims:
            return 0.0
        return sum(bool(_CITATION.search(claim)) for claim in claims) / len(claims)
