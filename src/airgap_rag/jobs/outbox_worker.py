import asyncio
import logging

from airgap_rag.core.config import get_settings
from airgap_rag.core.logging import configure_logging
from airgap_rag.core.tracing import configure_tracing
from airgap_rag.db.repositories.outbox import SQLAlchemyOutboxRepository
from airgap_rag.db.session import Database
from airgap_rag.jobs.outbox import OutboxDispatcher
from airgap_rag.jobs.tasks import TaskiqJobPublisher

logger = logging.getLogger(__name__)


async def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    database = Database(settings)
    tracing_runtime = configure_tracing(
        settings,
        sqlalchemy_engine=database.engine.sync_engine,
        service_name="airgap-rag-outbox",
    )
    publisher = TaskiqJobPublisher()
    logger.info("outbox_dispatcher_started")
    try:
        while True:
            try:
                async with database.session() as session:
                    dispatcher = OutboxDispatcher(
                        SQLAlchemyOutboxRepository(session),
                        publisher,
                        batch_size=settings.outbox_batch_size,
                    )
                    delivered = await dispatcher.dispatch_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "outbox_dispatch_cycle_failed",
                    extra={"error_type": type(exc).__name__},
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                delivered = 0
            await asyncio.sleep(0 if delivered else settings.outbox_poll_seconds)
    finally:
        await database.dispose()
        if tracing_runtime is not None:
            tracing_runtime.shutdown()
        logger.info("outbox_dispatcher_stopped")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
