import json

import httpx
import pytest

from airgap_rag.benchmarks.llm import BenchmarkError, OllamaLLMBenchmark
from airgap_rag.benchmarks.models import summarize
from airgap_rag.system_info import collect_hardware_info


def test_hardware_info_supports_cpu_only_hosts() -> None:
    info = collect_hardware_info(lambda: None)

    assert info.logical_cpu_count >= 1
    assert info.ram_total_mb > 0
    assert info.gpu is None
    assert info.vram_total_mb is None
    assert info.cuda_available is False


def test_metric_summary_uses_nearest_rank_percentiles() -> None:
    result = summarize([5.0, 1.0, 4.0, 2.0, 3.0])

    assert result.minimum == 1.0
    assert result.mean == 3.0
    assert result.p50 == 3.0
    assert result.p95 == 5.0
    assert result.maximum == 5.0


async def test_ollama_benchmark_reads_real_stream_metrics() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen:test",
                            "model": "qwen:test",
                            "size": 4 * 1024 * 1024,
                            "size_vram": 3 * 1024 * 1024,
                        }
                    ]
                },
            )
        body = "\n".join(
            [
                json.dumps({"message": {"role": "assistant", "content": "ответ"}, "done": False}),
                json.dumps(
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "prompt_eval_count": 12,
                        "eval_count": 20,
                        "eval_duration": 2_000_000_000,
                    }
                ),
            ]
        )
        return httpx.Response(200, text=body)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://ollama",
    )
    benchmark = OllamaLLMBenchmark(
        base_url="http://unused",
        model="qwen:test",
        timeout_seconds=5,
        max_tokens=64,
        temperature=0,
        keep_alive="5m",
        client=client,
    )

    report = await benchmark.run(
        prompt="test",
        quantization="Q4_K_M",
        warmup_iterations=0,
        measured_iterations=1,
    )

    assert report.samples[0].input_tokens == 12
    assert report.samples[0].output_tokens == 20
    assert report.samples[0].tokens_per_second == 10
    assert report.model_size_mb == 4
    assert report.ram_usage_mb == 1
    assert report.vram_usage_mb == 3
    await client.aclose()


async def test_ollama_benchmark_rejects_missing_metrics() -> None:
    body = "\n".join(
        [
            json.dumps({"message": {"content": "answer"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)),
        base_url="http://ollama",
    )
    benchmark = OllamaLLMBenchmark(
        base_url="http://unused",
        model="qwen:test",
        timeout_seconds=5,
        max_tokens=64,
        temperature=0,
        keep_alive="5m",
        client=client,
    )

    with pytest.raises(BenchmarkError, match="metrics"):
        await benchmark.measure("test")
    await client.aclose()
