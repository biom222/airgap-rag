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
