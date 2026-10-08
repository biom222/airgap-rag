from contextlib import AbstractAsyncContextManager
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession


class AsyncSessionProvider(Protocol):
    def session(self) -> AbstractAsyncContextManager[AsyncSession]: ...
