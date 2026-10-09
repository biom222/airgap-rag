import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

from airgap_rag.evaluation.client import RetrievalApiClient, RetrievalApiError
from airgap_rag.evaluation.dataset import EvaluationDatasetError, load_evaluation_dataset
from airgap_rag.evaluation.reports import EvaluationReportError, write_evaluation_reports
from airgap_rag.evaluation.runner import EvaluationError, RetrievalEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airgap-rag-evaluate",
        description="Evaluate retrieval quality against a curated JSONL dataset.",
    )
    parser.add_argument("dataset", type=Path, help="Path to a JSONL evaluation dataset")
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
    return parser


async def run(args: argparse.Namespace) -> int:
    cases = load_evaluation_dataset(args.dataset)
    base_url = str(args.base_url).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise EvaluationError("base URL must use http or https")
    if args.timeout <= 0:
        raise EvaluationError("timeout must be positive")

    async with httpx.AsyncClient(base_url=base_url, timeout=args.timeout) as http_client:
        report = await RetrievalEvaluator(RetrievalApiClient(http_client)).evaluate(
            cases,
            top_k=tuple(args.top_k),
            dataset_name=args.dataset.name,
            base_url=base_url,
        )
    paths = write_evaluation_reports(
        report,
        args.output_dir,
        report_name=args.report_name,
    )
    print(f"Hit@K: {report.hit_at_k}")
    print(f"MRR@{report.mrr_cutoff}: {report.mrr:.4f}")
    print(f"JSON report: {paths.json}")
    print(f"Markdown report: {paths.markdown}")
    return 0


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
