from fastapi import FastAPI

from airgap_rag import __version__
from airgap_rag.api.exception_handlers import register_exception_handlers
from airgap_rag.api.middleware import RequestContextMiddleware
from airgap_rag.api.routers.chat import router as chat_router
from airgap_rag.api.routers.documents import router as documents_router
from airgap_rag.api.routers.jobs import router as jobs_router
from airgap_rag.api.routers.retrieval import router as retrieval_router
from airgap_rag.api.routers.system import router as system_router
from airgap_rag.auth import Authenticator
from airgap_rag.core.config import Settings, get_settings
from airgap_rag.core.lifecycle import DisposableDatabase, TaskBroker, create_lifespan
from airgap_rag.core.logging import configure_logging
from airgap_rag.core.tracing import configure_tracing
from airgap_rag.db.session import Database
from airgap_rag.embeddings.base import EmbeddingProvider
from airgap_rag.jobs.service import JobPublisher
from airgap_rag.llm.base import LLMProvider
from airgap_rag.metrics import ApplicationMetrics
from airgap_rag.reranking.base import Reranker
from airgap_rag.vector_store.base import VectorStore


def create_app(
    settings: Settings | None = None,
    database: DisposableDatabase | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    job_publisher: JobPublisher | None = None,
    task_broker: TaskBroker | None = None,
    llm_provider: LLMProvider | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    metrics = ApplicationMetrics()
    configure_logging(resolved_settings.log_level)
    resolved_database = database or Database(resolved_settings)
    resolved_job_publisher = job_publisher
    resolved_task_broker = task_broker
    if resolved_job_publisher is None:
        from airgap_rag.jobs.broker import broker
        from airgap_rag.jobs.tasks import TaskiqJobPublisher

        resolved_job_publisher = TaskiqJobPublisher()
        resolved_task_broker = broker

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        docs_url=None if resolved_settings.app_env == "production" else "/docs",
        redoc_url=None if resolved_settings.app_env == "production" else "/redoc",
        openapi_url=None if resolved_settings.app_env == "production" else "/openapi.json",
        lifespan=create_lifespan(
            resolved_settings,
            resolved_database,
            embedding_provider,
            vector_store,
            resolved_job_publisher,
            resolved_task_broker,
            llm_provider,
            reranker,
            metrics,
        ),
    )
    application.state.metrics = metrics
    application.state.authenticator = Authenticator(
        required=resolved_settings.auth_required,
        principals_json=resolved_settings.auth_principals_json,
    )
    engine = getattr(resolved_database, "engine", None)
    application.state.tracing_runtime = configure_tracing(
        resolved_settings,
        app=application,
        sqlalchemy_engine=getattr(engine, "sync_engine", None),
    )
    application.add_middleware(RequestContextMiddleware)
    application.include_router(system_router)
    application.include_router(documents_router)
    application.include_router(jobs_router)
    application.include_router(retrieval_router)
    application.include_router(chat_router)
    register_exception_handlers(application)
    return application


app = create_app()
