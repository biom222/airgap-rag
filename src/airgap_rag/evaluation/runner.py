from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from airgap_rag.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationReport,
)


class EvaluationError(ValueError):
    """Evaluation parameters cannot produce a meaningful report."""


class RetrievalClient(Protocol):
    async def search(
        self,
        question: str,
        *,
        limit: int,
        document_ids: tuple[UUID, ...] | None,
    ) -> tuple[UUID, ...]: ...


class RetrievalEvaluator:
    def __init__(self, client: RetrievalClient) -> None:
        self._client = client

    async def evaluate(
        self,
        cases: tuple[EvaluationCase, ...],
        *,
        top_k: tuple[int, ...],
        dataset_name: str,
        base_url: str,
    ) -> EvaluationReport:
        normalized_top_k = self._normalize_top_k(top_k)
        if not cases:
            raise EvaluationError("Evaluation requires at least one case")

        results: list[EvaluationCaseResult] = []
        limit = max(normalized_top_k)
        for case in cases:
            retrieved = await self._client.search(
                case.question,
                limit=limit,
                document_ids=case.document_ids,
            )
            relevant = set(case.relevant_chunk_ids)
            first_rank = next(
                (rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant),
                None,
            )
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    relevant_chunk_ids=case.relevant_chunk_ids,
                    retrieved_chunk_ids=retrieved,
                    first_relevant_rank=first_rank,
                    reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
                    hits_at_k={
                        cutoff: any(chunk_id in relevant for chunk_id in retrieved[:cutoff])
                        for cutoff in normalized_top_k
                    },
                )
            )

        case_count = len(results)
        return EvaluationReport(
            evaluated_at=datetime.now(UTC),
            dataset_name=dataset_name,
            base_url=base_url,
            case_count=case_count,
            top_k=normalized_top_k,
            hit_at_k={
                cutoff: sum(result.hits_at_k[cutoff] for result in results) / case_count
                for cutoff in normalized_top_k
            },
            mrr=sum(result.reciprocal_rank for result in results) / case_count,
            mrr_cutoff=limit,
            cases=tuple(results),
        )

    @staticmethod
    def _normalize_top_k(top_k: tuple[int, ...]) -> tuple[int, ...]:
        if not top_k:
            raise EvaluationError("At least one K value is required")
        if any(cutoff < 1 or cutoff > 100 for cutoff in top_k):
            raise EvaluationError("Every K value must be between 1 and 100")
        return tuple(sorted(set(top_k)))
