from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.models.outbox import OutboxEvent


class OutboxRepository(Protocol):
    async def lock_pending(self, limit: int) -> list[OutboxEvent]: ...

    def mark_processed(self, event: OutboxEvent) -> None: ...

    def mark_failed(self, event: OutboxEvent, message: str) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class SQLAlchemyOutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lock_pending(self, limit: int) -> list[OutboxEvent]:
        statement = (
            select(OutboxEvent)
            .where(
                OutboxEvent.processed_at.is_(None),
                OutboxEvent.available_at <= datetime.now(UTC),
            )
            .order_by(OutboxEvent.created_at, OutboxEvent.id)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(statement)
        return list(result.scalars())

    def mark_processed(self, event: OutboxEvent) -> None:
        event.attempts += 1
        event.processed_at = datetime.now(UTC)
        event.last_error = None

    def mark_failed(self, event: OutboxEvent, message: str) -> None:
        event.attempts += 1
        delay_seconds = min(2 ** min(event.attempts, 6), 60)
        event.available_at = datetime.now(UTC) + timedelta(seconds=delay_seconds)
        event.last_error = message[:2000]

    async def commit(self) -> None:
        await self._session.commit()

    async def rollback(self) -> None:
        await self._session.rollback()
