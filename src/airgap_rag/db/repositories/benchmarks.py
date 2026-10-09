from collections.abc import Iterable

from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.models.benchmarks import ModelBenchmark


class SQLAlchemyBenchmarkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_all(self, records: Iterable[ModelBenchmark]) -> None:
        self._session.add_all(list(records))
        await self._session.commit()
