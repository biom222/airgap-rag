from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Annotated, Protocol, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.api.dependencies.auth import get_current_principal
from airgap_rag.auth import Principal
from airgap_rag.db.repositories.jobs import SQLAlchemyJobRepository
from airgap_rag.jobs.service import JobQueryService


class SessionDatabase(Protocol):
    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...


async def get_job_query_service(
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> AsyncIterator[JobQueryService]:
    database = cast(SessionDatabase, request.app.state.database)
    async with database.session() as session:
        yield JobQueryService(
            SQLAlchemyJobRepository(session),
            None if principal.is_admin else principal.subject,
        )
