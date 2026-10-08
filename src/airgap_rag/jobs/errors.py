from fastapi import status

from airgap_rag.core.exceptions import AppError


class JobNotFoundError(AppError):
    def __init__(self) -> None:
        super().__init__("job_not_found", "Задание не найдено.", status.HTTP_404_NOT_FOUND)


class JobDispatchError(AppError):
    def __init__(self) -> None:
        super().__init__(
            "job_dispatch_failed",
            "Задание сохранено, но очередь временно недоступна. Повторите запрос позже.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class RetryableJobError(RuntimeError):
    """Signals Taskiq that a persisted job should be delivered again."""
