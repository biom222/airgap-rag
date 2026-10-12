import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

from airgap_rag.evaluation.client import (
    GenerationApiClient,
    RetrievalApiClient,
    RetrievalApiError,
)
from airgap_rag.evaluation.dataset import (
    EvaluationDatasetError,
    load_evaluation_dataset,
    load_generation_dataset,
)
from airgap_rag.evaluation.generation import GenerationEvaluator
from airgap_rag.evaluation.reports import (
    EvaluationReportError,
    write_evaluation_reports,
    write_generation_reports,
)
from airgap_rag.evaluation.runner import EvaluationError, RetrievalEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airgap-rag-evaluate",
        description="Evaluate retrieval quality against a curated JSONL dataset.",
    )
    parser.add_argument("dataset", type=Path, help="Path to a JSONL evaluation dataset")
    parser.add_argument("--mode", choices=("retrieval", "generation"), default="retrieval")
    parser.add_argument(
        "--base-url",
        default=os.getenv("EVALUATION_BASE_URL", "http://localhost:8000"),
        help="AirGapRAG API base URL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="One or more retrieval cutoffs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory for JSON and Markdown reports",
    )
    parser.add_argument("--report-name", help="Optional deterministic report filename stem")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--api-token", default=os.getenv("EVALUATION_API_TOKEN"))
    parser.add_argument("--min-answer-f1", type=float, default=0.6)
    parser.add_argument("--min-citation-precision", type=float, default=0.9)
    parser.add_argument("--min-citation-recall", type=float, default=0.8)
    parser.add_argument("--min-faithfulness", type=float, default=0.9)
    return parser


async def run(args: argparse.Namespace) -> int:
    base_url = str(args.base_url).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise EvaluationError("base URL must use http or https")
    if args.timeout <= 0:
        raise EvaluationError("timeout must be positive")

    async with httpx.AsyncClient(base_url=base_url, timeout=args.timeout) as http_client:
        if args.mode == "generation":
            thresholds = {
                "answer_f1": args.min_answer_f1,
                "citation_precision": args.min_citation_precision,
                "citation_recall": args.min_citation_recall,
                "faithfulness": args.min_faithfulness,
            }
            if any(not 0 <= value <= 1 for value in thresholds.values()):
                raise EvaluationError("generation thresholds must be between 0 and 1")
            generation_report = await GenerationEvaluator(
                GenerationApiClient(http_client, args.api_token)
            ).evaluate(
                load_generation_dataset(args.dataset),
                dataset_name=args.dataset.name,
                base_url=base_url,
                thresholds=thresholds,
            )
            paths = write_generation_reports(
                generation_report, args.output_dir, report_name=args.report_name
            )
            print(f"Generation metrics passed: {generation_report.passed}")
            exit_code = 0 if generation_report.passed else 2
        else:
            report = await RetrievalEvaluator(
                RetrievalApiClient(http_client, args.api_token)
            ).evaluate(
                load_evaluation_dataset(args.dataset),
                top_k=tuple(args.top_k),
                dataset_name=args.dataset.name,
                base_url=base_url,
            )
            paths = write_evaluation_reports(report, args.output_dir, report_name=args.report_name)
            print(f"Hit@K: {report.hit_at_k}")
            print(f"MRR@{report.mrr_cutoff}: {report.mrr:.4f}")
            exit_code = 0
    print(f"JSON report: {paths.json}")
    print(f"Markdown report: {paths.markdown}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except (
        EvaluationDatasetError,
        EvaluationError,
        RetrievalApiError,
        EvaluationReportError,
    ) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
