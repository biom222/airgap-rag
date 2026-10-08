from fastapi import status

from airgap_rag.core.exceptions import AppError


class ChatSessionNotFoundError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="chat_session_not_found",
            message="Сессия диалога не найдена.",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class LLMServiceUnavailableError(AppError):
    def __init__(self, message: str = "Локальная языковая модель недоступна.") -> None:
        super().__init__(
            code="llm_unavailable",
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class LLMGenerationFailedError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="llm_generation_failed",
            message="Локальная языковая модель вернула некорректный ответ.",
            status_code=status.HTTP_502_BAD_GATEWAY,
        )
