import hashlib
import time
from dataclasses import asdict

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from airgap_rag.benchmarks.models import LLMReport, LLMSample, summarize
from airgap_rag.system_info import collect_hardware_info


class BenchmarkError(RuntimeError):
    """A benchmark could not produce trustworthy measurements."""


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str = ""


class _StreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _Message = _Message()
    done: bool
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class _LoadedModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    model: str | None = None
    size: int
    size_vram: int = 0


class _Processes(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[_LoadedModel]


class OllamaLLMBenchmark:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        max_tokens: int,
        temperature: float,
        keep_alive: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._keep_alive = keep_alive
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def measure(self, prompt: str) -> LLMSample:
        started = time.perf_counter_ns()
        first_token_ns: int | None = None
        final_event: _StreamEvent | None = None
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "keep_alive": self._keep_alive,
                    "options": {
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = _StreamEvent.model_validate_json(line)
                    if event.message.content and first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
                    if event.done:
                        final_event = event
                        break
        except (httpx.HTTPError, ValidationError) as exc:
            raise BenchmarkError("Ollama benchmark request failed") from exc
        finished = time.perf_counter_ns()
        if first_token_ns is None or final_event is None:
            raise BenchmarkError("Ollama did not return tokens and a final metrics event")
        if (
            final_event.prompt_eval_count is None
            or final_event.eval_count is None
            or final_event.eval_duration is None
            or final_event.eval_count <= 0
            or final_event.eval_duration <= 0
        ):
            raise BenchmarkError("Ollama final event does not contain token timing metrics")
        return LLMSample(
            ttft_ms=(first_token_ns - started) / 1_000_000,
            tokens_per_second=final_event.eval_count / (final_event.eval_duration / 1_000_000_000),
            total_latency_ms=(finished - started) / 1_000_000,
            input_tokens=final_event.prompt_eval_count,
            output_tokens=final_event.eval_count,
        )

    async def loaded_model_memory(self) -> tuple[float | None, float | None, float | None]:
        try:
            response = await self._client.get("/api/ps")
            response.raise_for_status()
            processes = _Processes.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            return None, None, None
        loaded = next(
            (item for item in processes.models if self._model in {item.name, item.model}),
            None,
        )
        if loaded is None:
            return None, None, None
        mib = 1024 * 1024
        return (
            loaded.size / mib,
            max(0, loaded.size - loaded.size_vram) / mib,
            loaded.size_vram / mib,
        )

    async def run(
        self,
        *,
        prompt: str,
        quantization: str,
        warmup_iterations: int,
        measured_iterations: int,
    ) -> LLMReport:
        if warmup_iterations < 0 or measured_iterations < 1:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        for _ in range(warmup_iterations):
            await self.measure(prompt)
        samples = [await self.measure(prompt) for _ in range(measured_iterations)]
        model_size, ram_usage, vram_usage = await self.loaded_model_memory()
        return LLMReport(
            model=self._model,
            runtime="ollama",
            quantization=quantization,
            hardware=asdict(collect_hardware_info()),
            model_size_mb=model_size,
            ram_usage_mb=ram_usage,
            vram_usage_mb=vram_usage,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            samples=samples,
            ttft_ms=summarize([sample.ttft_ms for sample in samples]),
            tokens_per_second=summarize([sample.tokens_per_second for sample in samples]),
            total_latency_ms=summarize([sample.total_latency_ms for sample in samples]),
        )
