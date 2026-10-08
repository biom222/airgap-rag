from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol, cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.repositories.jobs import SQLAlchemyJobRepository
from airgap_rag.jobs.service import JobQueryService


class SessionDatabase(Protocol):
    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...


async def get_job_query_service(request: Request) -> AsyncIterator[JobQueryService]:
    database = cast(SessionDatabase, request.app.state.database)
    async with database.session() as session:
        yield JobQueryService(SQLAlchemyJobRepository(session))
