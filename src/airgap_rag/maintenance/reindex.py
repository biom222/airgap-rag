import asyncio
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select

from airgap_rag.core.config import Settings, get_settings
from airgap_rag.db.models.documents import Document
from airgap_rag.db.models.jobs import Job
from airgap_rag.db.models.outbox import OutboxEvent
from airgap_rag.db.session import Database
from airgap_rag.documents.types import DocumentStatus
from airgap_rag.jobs.types import JobStatus, JobType


async def reset_vector_index(settings: Settings) -> None:
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout_seconds,
    )
    try:
        if await client.collection_exists(settings.qdrant_collection):
            await client.delete_collection(settings.qdrant_collection)
    finally:
        await client.close()


async def schedule_reindex() -> int:
    settings = get_settings()
    await reset_vector_index(settings)
    database = Database(settings)
    try:
        async with database.session() as session:
            result = await session.execute(select(Document).order_by(Document.created_at))
            documents = list(result.scalars())
            for document in documents:
                job_id = uuid4()
                document.status = DocumentStatus.PENDING
                session.add(
                    Job(
                        id=job_id,
                        document_id=document.id,
                        type=JobType.INGESTION,
                        status=JobStatus.PENDING,
                        progress=0,
                        attempt=0,
                        trace_id=None,
                    )
                )
                session.add(
                    OutboxEvent(
                        id=uuid4(),
                        event_type="ingestion.requested",
                        aggregate_id=job_id,
                        payload={"job_id": str(job_id), "trace_id": "", "trace_context": {}},
                        attempts=0,
                    )
                )
            await session.commit()
            return len(documents)
    finally:
        await database.dispose()


def main() -> None:
    count = asyncio.run(schedule_reindex())
    print(f"Scheduled {count} documents for reindexing")


if __name__ == "__main__":
    main()
