from datetime import UTC, datetime

from airgap_rag.db.models.jobs import Job
from airgap_rag.jobs.types import JobStatus

_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PENDING: frozenset({JobStatus.PARSING, JobStatus.FAILED}),
    JobStatus.PARSING: frozenset({JobStatus.CHUNKING, JobStatus.FAILED}),
    JobStatus.CHUNKING: frozenset({JobStatus.EMBEDDING, JobStatus.FAILED}),
    JobStatus.EMBEDDING: frozenset({JobStatus.INDEXING, JobStatus.FAILED}),
    JobStatus.INDEXING: frozenset({JobStatus.READY, JobStatus.FAILED}),
    JobStatus.READY: frozenset(),
    JobStatus.FAILED: frozenset(),
}

_PROGRESS_BY_STATUS = {
    JobStatus.PENDING: 0,
    JobStatus.PARSING: 10,
    JobStatus.CHUNKING: 30,
    JobStatus.EMBEDDING: 50,
    JobStatus.INDEXING: 80,
    JobStatus.READY: 100,
    JobStatus.FAILED: 100,
}


class InvalidJobTransitionError(ValueError):
    def __init__(self, source: JobStatus, target: JobStatus) -> None:
        super().__init__(f"Invalid job transition: {source} -> {target}")
        self.source = source
        self.target = target


def transition_job(
    job: Job,
    target: JobStatus,
    *,
    now: datetime | None = None,
) -> None:
    """Apply a normal forward transition and its timestamp/progress invariants."""
    if job.status == target:
        return
    if target not in _ALLOWED_TRANSITIONS[job.status]:
        raise InvalidJobTransitionError(job.status, target)

    timestamp = now or datetime.now(UTC)
    job.status = target
    job.progress = _PROGRESS_BY_STATUS[target]
    if target == JobStatus.PARSING and job.started_at is None:
        job.started_at = timestamp
    if target in {JobStatus.READY, JobStatus.FAILED}:
        job.finished_at = timestamp


def reset_job_for_retry(job: Job) -> None:
    """Reset a non-terminal attempt; this is deliberately not a normal transition."""
    if job.status in {JobStatus.READY, JobStatus.FAILED}:
        raise InvalidJobTransitionError(job.status, JobStatus.PENDING)
    job.status = JobStatus.PENDING
    job.progress = _PROGRESS_BY_STATUS[JobStatus.PENDING]
    job.finished_at = None
