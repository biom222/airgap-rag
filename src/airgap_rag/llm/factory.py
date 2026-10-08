from airgap_rag.core.config import Settings
from airgap_rag.llm.base import LLMProvider
from airgap_rag.llm.mock import MockLLMProvider
from airgap_rag.llm.ollama import OllamaProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider()
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        keep_alive=settings.ollama_keep_alive,
    )
