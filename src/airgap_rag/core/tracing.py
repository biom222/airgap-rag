from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from airgap_rag.core.config import Settings


@dataclass(slots=True)
class TracingRuntime:
    provider: TracerProvider

    def shutdown(self) -> None:
        self.provider.shutdown()


def configure_tracing(
    settings: Settings,
    *,
    app: FastAPI | None = None,
    sqlalchemy_engine: Any | None = None,
    service_name: str | None = None,
) -> TracingRuntime | None:
    if not settings.tracing_enabled:
        return None

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name or settings.otel_service_name,
                "deployment.environment.name": settings.app_env,
            }
        )
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
    if app is not None:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            excluded_urls="health,ready,metrics",
        )
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    if sqlalchemy_engine is not None:
        SQLAlchemyInstrumentor().instrument(
            engine=sqlalchemy_engine,
            tracer_provider=provider,
        )
    return TracingRuntime(provider)


def tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)
