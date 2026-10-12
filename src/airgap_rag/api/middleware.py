import logging
from time import perf_counter
from typing import cast
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import Route

from airgap_rag.metrics import MetricsRecorder

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        trace_id = request.headers.get("X-Correlation-ID") or request_id
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        started_at = perf_counter()
        response = await call_next(request)
        duration_seconds = perf_counter() - started_at
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = trace_id
        route_object = request.scope.get("route")
        route = route_object.path if isinstance(route_object, Route) else "unmatched"
        metrics = cast(MetricsRecorder, request.app.state.metrics)
        metrics.observe_http(request.method, route, response.status_code, duration_seconds)
        logger.info(
            "http_request_completed",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "method": request.method,
                "route": route,
                "status_code": response.status_code,
                "duration_ms": round(duration_seconds * 1000, 3),
            },
        )
        return response
