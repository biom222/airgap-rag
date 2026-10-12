from datetime import UTC, datetime
from uuid import UUID, uuid4

from airgap_rag.db.models.outbox import OutboxEvent
from airgap_rag.jobs.outbox import OutboxDispatcher


class FakeOutboxRepository:
    def __init__(self, event: OutboxEvent) -> None:
        self.event = event
        self.commits = 0

    async def lock_pending(self, limit: int) -> list[OutboxEvent]:
        return [self.event][:limit]

    def mark_processed(self, event: OutboxEvent) -> None:
        event.attempts += 1
        event.processed_at = datetime.now(UTC)

    def mark_failed(self, event: OutboxEvent, message: str) -> None:
        event.attempts += 1
        event.last_error = message

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        return None


class FakePublisher:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[UUID, dict[str, str] | None]] = []

    async def enqueue_ingestion(
        self,
        job_id: UUID,
        *,
        trace_context: dict[str, str] | None = None,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.calls.append((job_id, trace_context))


def make_event(job_id: UUID) -> OutboxEvent:
    return OutboxEvent(
        id=uuid4(),
        event_type="ingestion.requested",
        aggregate_id=job_id,
        payload={
            "job_id": str(job_id),
            "trace_id": "request-id",
            "trace_context": {"traceparent": "00-abc-def-01"},
        },
        attempts=0,
        available_at=datetime.now(UTC),
        processed_at=None,
        last_error=None,
        created_at=datetime.now(UTC),
    )


async def test_outbox_dispatcher_delivers_and_marks_event() -> None:
    job_id = uuid4()
    repository = FakeOutboxRepository(make_event(job_id))
    publisher = FakePublisher()

    delivered = await OutboxDispatcher(repository, publisher, batch_size=10).dispatch_once()

    assert delivered == 1
    assert publisher.calls == [(job_id, {"traceparent": "00-abc-def-01"})]
    assert repository.event.processed_at is not None
    assert repository.commits == 1


async def test_outbox_dispatcher_retains_failed_event_for_retry() -> None:
    repository = FakeOutboxRepository(make_event(uuid4()))

    delivered = await OutboxDispatcher(
        repository,
        FakePublisher(ConnectionError("redis unavailable")),
        batch_size=10,
    ).dispatch_once()

    assert delivered == 0
    assert repository.event.processed_at is None
    assert repository.event.attempts == 1
    assert "redis unavailable" in (repository.event.last_error or "")
    assert repository.commits == 1
