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

    redis_url: str = "redis://localhost:6379/0"
    job_max_attempts: int = Field(default=3, ge=1, le=20)
    job_timeout_seconds: int = Field(default=900, ge=1)
    job_retry_delay_seconds: int = Field(default=5, ge=1)
    job_retry_max_delay_seconds: int = Field(default=60, ge=1)
    job_stale_after_seconds: int = Field(default=960, ge=1)

    llm_provider: Literal["mock", "ollama"] = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_keep_alive: str = "5m"
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1)

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

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        if not value.startswith(("redis://", "rediss://")):
            raise ValueError("REDIS_URL must use the redis or rediss scheme")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must use the http or https scheme")
        return value.rstrip("/")

    @field_validator("ollama_model", "ollama_keep_alive")
    @classmethod
    def validate_non_empty_llm_settings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LLM model and keep-alive settings must not be empty")
        return value

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.job_retry_max_delay_seconds < self.job_retry_delay_seconds:
            raise ValueError(
                "JOB_RETRY_MAX_DELAY_SECONDS must be greater than or equal to "
                "JOB_RETRY_DELAY_SECONDS"
            )
        if self.job_stale_after_seconds <= self.job_timeout_seconds:
            raise ValueError("JOB_STALE_AFTER_SECONDS must be greater than JOB_TIMEOUT_SECONDS")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
