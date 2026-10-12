from prometheus_client import generate_latest

from airgap_rag.metrics import ApplicationMetrics


def test_application_metrics_exposes_required_series() -> None:
    metrics = ApplicationMetrics()

    metrics.record_document_created()
    metrics.observe_embedding(0.01)
    metrics.observe_retrieval(0.02)
    metrics.observe_reranking(0.03)
    metrics.observe_rag("sync", "success", 0.2)
    metrics.observe_llm_ttft(0.04)
    metrics.observe_llm_generation(0.15)
    metrics.add_llm_tokens(12)
    metrics.record_ingestion_failure()
    metrics.record_task_failure(retryable=True)

    payload = generate_latest(metrics.registry).decode()
    expected_metrics = (
        "documents_total",
        "document_ingestion_failures_total",
        "rag_requests_total",
        "rag_request_duration_seconds",
        "embedding_duration_seconds",
        "retrieval_duration_seconds",
        "reranking_duration_seconds",
        "llm_ttft_seconds",
        "llm_generation_duration_seconds",
        "llm_tokens_generated_total",
        "task_failures_total",
    )
    for metric in expected_metrics:
        assert metric in payload
    assert "llm_tokens_generated_total 12.0" in payload
