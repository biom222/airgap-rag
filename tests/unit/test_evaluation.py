import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from airgap_rag.evaluation.client import RetrievalApiClient, RetrievalApiError
from airgap_rag.evaluation.dataset import EvaluationDatasetError, load_evaluation_dataset
from airgap_rag.evaluation.models import EvaluationCase
from airgap_rag.evaluation.reports import EvaluationReportError, write_evaluation_reports
from airgap_rag.evaluation.runner import EvaluationError, RetrievalEvaluator


class FakeRetrievalClient:
    def __init__(self, rankings: dict[str, tuple[UUID, ...]]) -> None:
        self.rankings = rankings
        self.limits: list[int] = []

    async def search(
        self,
        question: str,
        *,
        limit: int,
        document_ids: tuple[UUID, ...] | None,
    ) -> tuple[UUID, ...]:
        del document_ids
        self.limits.append(limit)
        return self.rankings[question][:limit]


async def test_evaluator_calculates_hit_at_k_and_mrr() -> None:
    irrelevant = uuid4()
    first_relevant = uuid4()
    second_relevant = uuid4()
    cases = (
        EvaluationCase(
            case_id="rank-two",
            question="first question",
            relevant_chunk_ids=(first_relevant,),
        ),
        EvaluationCase(
            case_id="rank-one",
            question="second question",
            relevant_chunk_ids=(second_relevant,),
        ),
    )
    client = FakeRetrievalClient(
        {
            "first question": (irrelevant, first_relevant),
            "second question": (second_relevant, irrelevant),
        }
    )

    report = await RetrievalEvaluator(client).evaluate(
        cases,
        top_k=(3, 1, 3),
        dataset_name="test.jsonl",
        base_url="http://test",
    )

    assert report.top_k == (1, 3)
    assert report.hit_at_k == {1: 0.5, 3: 1.0}
    assert report.mrr == pytest.approx(0.75)
    assert report.mrr_cutoff == 3
    assert report.cases[0].first_relevant_rank == 2
    assert client.limits == [3, 3]


@pytest.mark.parametrize("top_k", [(), (0,), (101,)])
async def test_evaluator_rejects_invalid_cutoffs(top_k: tuple[int, ...]) -> None:
    evaluator = RetrievalEvaluator(FakeRetrievalClient({}))

    with pytest.raises(EvaluationError):
        await evaluator.evaluate(
            (
                EvaluationCase(
                    case_id="case",
                    question="question",
                    relevant_chunk_ids=(uuid4(),),
                ),
            ),
            top_k=top_k,
            dataset_name="test.jsonl",
            base_url="http://test",
        )


def test_dataset_loader_reads_jsonl_and_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    chunk_id = uuid4()
    dataset = tmp_path / "dataset.jsonl"
    record = {
        "case_id": "duplicate",
        "question": "What is the retention period?",
        "relevant_chunk_ids": [str(chunk_id)],
    }
    dataset.write_text(
        f"{json.dumps(record)}\n{json.dumps(record)}\n",
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetError, match="Duplicate case_id"):
        load_evaluation_dataset(dataset)


def test_dataset_loader_rejects_duplicate_relevant_chunks(tmp_path: Path) -> None:
    chunk_id = uuid4()
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "case",
                "question": "Question",
                "relevant_chunk_ids": [str(chunk_id), str(chunk_id)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDatasetError, match="relevant_chunk_ids"):
        load_evaluation_dataset(dataset)


async def test_api_client_sends_requested_depth_and_document_filter() -> None:
    chunk_id = uuid4()
    document_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        payload: dict[str, Any] = json.loads(request.content)
        assert request.url.path == "/api/v1/retrieval/search"
        assert payload["top_k"] == 5
        assert payload["top_n"] == 5
        assert payload["document_ids"] == [str(document_id)]
        return httpx.Response(200, json={"results": [{"chunk_id": str(chunk_id)}]})

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        result = await RetrievalApiClient(http_client).search(
            "question",
            limit=5,
            document_ids=(document_id,),
        )

    assert result == (chunk_id,)


async def test_api_client_rejects_invalid_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"results": [{"chunk_id": "not-a-uuid"}]})

    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        with pytest.raises(RetrievalApiError):
            await RetrievalApiClient(http_client).search(
                "question",
                limit=1,
                document_ids=None,
            )


async def test_reports_are_written_as_json_and_markdown(tmp_path: Path) -> None:
    relevant = uuid4()
    report = await RetrievalEvaluator(FakeRetrievalClient({"question": (relevant,)})).evaluate(
        (
            EvaluationCase(
                case_id="case",
                question="question",
                relevant_chunk_ids=(relevant,),
            ),
        ),
        top_k=(1,),
        dataset_name="test.jsonl",
        base_url="http://test",
    )

    paths = write_evaluation_reports(report, tmp_path, report_name="fixed-report")

    parsed = json.loads(paths.json.read_text(encoding="utf-8"))
    markdown = paths.markdown.read_text(encoding="utf-8")
    assert parsed["hit_at_k"] == {"1": 1.0}
    assert "| Hit@1 | 1.0000 |" in markdown
    assert "| MRR@1 | 1.0000 |" in markdown

    with pytest.raises(EvaluationReportError):
        write_evaluation_reports(report, tmp_path, report_name="../unsafe")
