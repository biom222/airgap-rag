import json

from starlette.requests import Request

from airgap_rag.api.exception_handlers import app_error_handler, unhandled_error_handler
from airgap_rag.core.exceptions import AppError


def request_with_id(request_id: str) -> Request:
    request = Request({"type": "http", "method": "GET", "path": "/test", "headers": []})
    request.state.request_id = request_id
    return request


async def test_app_error_uses_uniform_error_shape() -> None:
    response = await app_error_handler(
        request_with_id("request-1"),
        AppError(code="example_error", message="Пример ошибки.", status_code=409),
    )

    assert response.status_code == 409
    assert json.loads(bytes(response.body)) == {
        "error": {
            "code": "example_error",
            "message": "Пример ошибки.",
            "request_id": "request-1",
        }
    }


async def test_unhandled_error_does_not_expose_exception_message() -> None:
    response = await unhandled_error_handler(
        request_with_id("request-2"),
        RuntimeError("sensitive internal details"),
    )

    assert response.status_code == 500
    assert json.loads(bytes(response.body)) == {
        "error": {
            "code": "internal_server_error",
            "message": "Внутренняя ошибка сервера.",
            "request_id": "request-2",
        }
    }
