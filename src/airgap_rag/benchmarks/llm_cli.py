import argparse
import asyncio
from pathlib import Path
from typing import cast

from airgap_rag.benchmarks.llm import OllamaLLMBenchmark
from airgap_rag.benchmarks.models import LLMReport
from airgap_rag.benchmarks.reports import write_json_report
from airgap_rag.core.config import Settings
from airgap_rag.db.models.benchmarks import ModelBenchmark
from airgap_rag.db.repositories.benchmarks import SQLAlchemyBenchmarkRepository
from airgap_rag.db.session import Database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a local Ollama model")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--model", default=None)
    parser.add_argument("--quantization", default="unknown")
    parser.add_argument("--prompt", default="Кратко объясни, что такое локальный RAG.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/llm.json"))
    parser.add_argument("--persist-db", action="store_true")
    return parser


async def persist_report(report: LLMReport, settings: Settings) -> None:
    database = Database(settings)
    hardware = cast(dict[str, object], report.hardware)
    try:
        async with database.session() as session:
            repository = SQLAlchemyBenchmarkRepository(session)
            await repository.add_all(
                ModelBenchmark(
                    model_name=report.model,
                    runtime=report.runtime,
                    quantization=report.quantization,
                    hardware=hardware,
                    model_size_mb=report.model_size_mb,
                    ttft_ms=sample.ttft_ms,
                    tokens_per_second=sample.tokens_per_second,
                    total_latency_ms=sample.total_latency_ms,
                    ram_usage_mb=report.ram_usage_mb,
                    vram_usage_mb=report.vram_usage_mb,
                    input_tokens=sample.input_tokens,
                    output_tokens=sample.output_tokens,
                )
                for sample in report.samples
            )
    finally:
        await database.dispose()


async def run(args: argparse.Namespace) -> Path:
    settings = Settings()
    benchmark = OllamaLLMBenchmark(
        base_url=args.base_url,
        model=args.model or settings.ollama_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_tokens=args.max_tokens,
        temperature=settings.llm_temperature,
        keep_alive=settings.ollama_keep_alive,
    )
    try:
        report = await benchmark.run(
            prompt=args.prompt,
            quantization=args.quantization,
            warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
        )
    finally:
        await benchmark.close()
    if args.persist_db:
        await persist_report(report, settings)
    return write_json_report(report, args.output)


def main() -> None:
    args = build_parser().parse_args()
    output = asyncio.run(run(args))
    print(output)


if __name__ == "__main__":
    main()
