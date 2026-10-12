import logging
from uuid import UUID

from airgap_rag.db.repositories.outbox import OutboxRepository
from airgap_rag.jobs.service import JobPublisher

logger = logging.getLogger(__name__)


class OutboxDispatcher:
    def __init__(
        self,
        repository: OutboxRepository,
        publisher: JobPublisher,
        *,
        batch_size: int,
    ) -> None:
        self._repository = repository
        self._publisher = publisher
        self._batch_size = batch_size

    async def dispatch_once(self) -> int:
        events = await self._repository.lock_pending(self._batch_size)
        delivered = 0
        for event in events:
            trace_id = event.payload.get("trace_id")
            try:
                if event.event_type != "ingestion.requested":
                    raise ValueError(f"Unsupported outbox event type: {event.event_type}")
                raw_job_id = event.payload.get("job_id")
                if not isinstance(raw_job_id, str):
                    raise ValueError("Outbox event does not contain a string job_id")
                job_id = UUID(raw_job_id)
                raw_context = event.payload.get("trace_context")
                trace_context = (
                    {
                        str(key): str(value)
                        for key, value in raw_context.items()
                        if isinstance(key, str) and isinstance(value, str)
                    }
                    if isinstance(raw_context, dict)
                    else None
                )
                await self._publisher.enqueue_ingestion(
                    job_id,
                    trace_context=trace_context,
                )
            except Exception as exc:
                self._repository.mark_failed(event, f"{type(exc).__name__}: {exc}")
                logger.warning(
                    "outbox_delivery_failed",
                    extra={
                        "outbox_event_id": str(event.id),
                        "job_id": str(event.aggregate_id),
                        "trace_id": trace_id if isinstance(trace_id, str) else None,
                        "attempt": event.attempts + 1,
                        "error_type": type(exc).__name__,
                    },
                )
                continue
            self._repository.mark_processed(event)
            delivered += 1
            logger.info(
                "outbox_event_delivered",
                extra={
                    "outbox_event_id": str(event.id),
                    "job_id": str(event.aggregate_id),
                    "trace_id": trace_id if isinstance(trace_id, str) else None,
                    "attempt": event.attempts,
                },
            )
        await self._repository.commit()
        return delivered
