from enum import StrEnum


class JobType(StrEnum):
    INGESTION = "INGESTION"


class JobStatus(StrEnum):
    PENDING = "PENDING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


ACTIVE_JOB_STATUSES = frozenset(
    {
        JobStatus.PARSING,
        JobStatus.CHUNKING,
        JobStatus.EMBEDDING,
        JobStatus.INDEXING,
    }
)

TERMINAL_JOB_STATUSES = frozenset({JobStatus.READY, JobStatus.FAILED})
