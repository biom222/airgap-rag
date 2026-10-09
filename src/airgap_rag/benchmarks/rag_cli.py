import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from airgap_rag.benchmarks.rag import create_rag_benchmark
from airgap_rag.benchmarks.reports import write_json_report
from airgap_rag.core.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the configured local RAG pipeline")
    parser.add_argument("--question", required=True)
    parser.add_argument("--document-id", action="append", type=UUID, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/rag.json"))
    return parser


async def run(args: argparse.Namespace) -> Path:
    settings = Settings(llm_max_tokens=args.max_tokens)
    if args.ollama_base_url:
        settings = Settings(
            llm_max_tokens=args.max_tokens,
            ollama_base_url=args.ollama_base_url,
        )
    benchmark = create_rag_benchmark(settings)
    try:
        report = await benchmark.run(
            question=args.question,
            document_ids=args.document_id,
            warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
        )
    finally:
        await benchmark.close()
    return write_json_report(report, args.output)


def main() -> None:
    args = build_parser().parse_args()
    output = asyncio.run(run(args))
    print(output)


if __name__ == "__main__":
    main()
