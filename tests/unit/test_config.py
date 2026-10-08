import pytest
from pydantic import ValidationError

from airgap_rag.core.config import Settings


def test_settings_accept_asyncpg_database_url() -> None:
    settings = Settings(database_url="postgresql+asyncpg://user:pass@localhost/database")

    assert settings.database_pool_size == 5


def test_settings_reject_non_asyncpg_database_url() -> None:
    with pytest.raises(ValidationError):
        Settings(database_url="postgresql://user:pass@localhost/database")


def test_settings_reject_invalid_qdrant_url() -> None:
    with pytest.raises(ValidationError):
        Settings(qdrant_url="localhost:6333")


def test_settings_reject_invalid_qdrant_collection() -> None:
    with pytest.raises(ValidationError):
        Settings(qdrant_collection="chunks/unsafe")


def test_settings_reject_invalid_redis_url() -> None:
    with pytest.raises(ValidationError):
        Settings(redis_url="localhost:6379")


def test_settings_reject_invalid_ollama_url() -> None:
    with pytest.raises(ValidationError):
        Settings(ollama_base_url="localhost:11434")


def test_settings_reject_empty_ollama_model() -> None:
    with pytest.raises(ValidationError):
        Settings(ollama_model=" ")


def test_job_stale_timeout_must_exceed_execution_timeout() -> None:
    with pytest.raises(ValidationError):
        Settings(job_timeout_seconds=60, job_stale_after_seconds=60)
