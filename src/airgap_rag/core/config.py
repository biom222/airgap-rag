from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AirGapRAG"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    database_url: str = "postgresql+asyncpg://airgap:airgap@localhost:5432/airgap_rag"
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=5, ge=0)
    database_pool_timeout_seconds: float = Field(default=10.0, gt=0)

    document_storage_path: Path = Path("data/documents")
    document_max_upload_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    document_max_docx_uncompressed_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    document_read_chunk_bytes: int = Field(default=1024 * 1024, gt=0)

    chunk_size: int = Field(default=1200, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)

    airgap_mode: bool = True
    embedding_provider: Literal["mock", "sentence_transformer"] = "mock"
    embedding_model_name_or_path: str = "models/embeddings/multilingual-e5-base"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_device: str = "cpu"
    embedding_document_prefix: str = "passage: "
    embedding_query_prefix: str = "query: "

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "document_chunks"
    qdrant_timeout_seconds: int = Field(default=10, gt=0)
    retrieval_top_k: int = Field(default=10, ge=1, le=100)

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg scheme")
        return value

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("QDRANT_URL must use the http or https scheme")
        return value.rstrip("/")

    @field_validator("qdrant_collection")
    @classmethod
    def validate_qdrant_collection(cls, value: str) -> str:
        if not value or not all(character.isalnum() or character in "-_" for character in value):
            raise ValueError("QDRANT_COLLECTION contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
