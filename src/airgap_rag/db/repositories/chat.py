from typing import Protocol
from uuid import UUID

from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from airgap_rag.db.models.chat import ChatMessage, ChatSession
from airgap_rag.db.protocols import AsyncSessionProvider
from airgap_rag.rag.errors import ChatSessionNotFoundError
from airgap_rag.rag.types import ChatHistoryMessage, ChatRole


class ChatRepository(Protocol):
    async def get_history(
        self,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[ChatHistoryMessage] | None: ...

    async def save_exchange(
        self,
        session_id: UUID | None,
        *,
        user_content: str,
        assistant_content: str,
    ) -> UUID: ...


class SQLAlchemyChatRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_history(
        self,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[ChatHistoryMessage] | None:
        if await self._session.get(ChatSession, session_id) is None:
            return None
        role_order = case((ChatMessage.role == ChatRole.ASSISTANT, 1), else_=0)
        result = await self._session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc(), role_order.desc())
            .limit(limit)
        )
        messages = list(reversed(result.scalars().all()))
        return [
            ChatHistoryMessage(role=message.role, content=message.content) for message in messages
        ]

    async def save_exchange(
        self,
        session_id: UUID | None,
        *,
        user_content: str,
        assistant_content: str,
    ) -> UUID:
        if session_id is None:
            chat_session = ChatSession()
            self._session.add(chat_session)
            await self._session.flush()
            session_id = chat_session.id
        elif await self._session.get(ChatSession, session_id) is None:
            raise ChatSessionNotFoundError()

        self._session.add_all(
            [
                ChatMessage(
                    session_id=session_id,
                    role=ChatRole.USER,
                    content=user_content,
                ),
                ChatMessage(
                    session_id=session_id,
                    role=ChatRole.ASSISTANT,
                    content=assistant_content,
                ),
            ]
        )
        await self._session.commit()
        return session_id


class SessionChatRepository:
    """Runs each chat operation in a short independent database session."""

    def __init__(self, database: AsyncSessionProvider) -> None:
        self._database = database

    async def get_history(
        self,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[ChatHistoryMessage] | None:
        async with self._database.session() as session:
            return await SQLAlchemyChatRepository(session).get_history(session_id, limit=limit)

    async def save_exchange(
        self,
        session_id: UUID | None,
        *,
        user_content: str,
        assistant_content: str,
    ) -> UUID:
        async with self._database.session() as session:
            return await SQLAlchemyChatRepository(session).save_exchange(
                session_id,
                user_content=user_content,
                assistant_content=assistant_content,
            )
