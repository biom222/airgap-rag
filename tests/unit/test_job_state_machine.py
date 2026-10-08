from datetime import UTC, datetime
from uuid import uuid4

import pytest

from airgap_rag.db.models.jobs import Job
from airgap_rag.jobs.state_machine import InvalidJobTransitionError, transition_job
from airgap_rag.jobs.types import JobStatus, JobType


def make_job() -> Job:
    now = datetime.now(UTC)
    return Job(
        id=uuid4(),
        document_id=uuid4(),
        type=JobType.INGESTION,
        status=JobStatus.PENDING,
        progress=0,
        attempt=0,
        created_at=now,
        updated_at=now,
    )


def test_job_follows_ingestion_state_machine() -> None:
    job = make_job()

    for status in (
        JobStatus.PARSING,
        JobStatus.CHUNKING,
        JobStatus.EMBEDDING,
        JobStatus.INDEXING,
        JobStatus.READY,
    ):
        transition_job(job, status)

    assert job.status == JobStatus.READY
    assert job.progress == 100
    assert job.started_at is not None
    assert job.finished_at is not None


def test_ready_job_cannot_return_to_parsing() -> None:
    job = make_job()
    for status in (
        JobStatus.PARSING,
        JobStatus.CHUNKING,
        JobStatus.EMBEDDING,
        JobStatus.INDEXING,
        JobStatus.READY,
    ):
        transition_job(job, status)

    with pytest.raises(InvalidJobTransitionError, match="READY -> PARSING"):
        transition_job(job, JobStatus.PARSING)


def test_active_job_can_fail_but_failed_job_is_terminal() -> None:
    job = make_job()
    transition_job(job, JobStatus.PARSING)
    transition_job(job, JobStatus.FAILED)

    assert job.status == JobStatus.FAILED
    with pytest.raises(InvalidJobTransitionError):
        transition_job(job, JobStatus.PARSING)
