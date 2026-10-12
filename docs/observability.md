# Observability

AirGapRAG использует три взаимодополняющих сигнала:

- structured JSON logs для расследования отдельных запросов и jobs;
- Prometheus metrics для агрегированной картины;
- provisioned Grafana dashboard для оперативного просмотра.

Полный текст документов, вопросы, ответы, prompts и secrets не записываются ни
в логи, ни в labels метрик.

## Запуск

```bash
docker compose --profile observability up -d --build
```

Локальные адреса:

- API metrics: `http://127.0.0.1:8000/metrics`;
- Prometheus: `http://127.0.0.1:9090`;
- Grafana: `http://127.0.0.1:3000`.

Перед использованием вне локальной машины обязательно задайте
`GRAFANA_ADMIN_USER` и `GRAFANA_ADMIN_PASSWORD`. Все published ports по умолчанию
привязаны к `127.0.0.1`, поэтому напрямую из сети они недоступны.

Prometheus отдельно собирает API и worker. Worker отдаёт метрики на внутреннем
порту `9001`; этот порт не публикуется на host.
Grafana telemetry, update checks и автоматическая загрузка suggested plugins
отключены, чтобы runtime не зависел от внешней сети.

## Correlation

Каждый HTTP response содержит:

- `X-Request-ID` — идентификатор конкретного HTTP request;
- `X-Correlation-ID` — идентификатор сквозной операции.

Клиент может передать оба заголовка самостоятельно. Если correlation ID не
передан, используется request ID. Для RAG этот ID попадает в structured events
`retrieval_completed`, `llm_generation_completed` и `rag_request_completed`.
Background jobs логируют `job_id`, `document_id`, attempt и duration.

## Метрики

Экспортируются метрики из исходного проекта:

- `http_requests_total`, `http_request_duration_seconds`;
- `documents_total`, `document_ingestion_failures_total`;
- `rag_requests_total`, `rag_request_duration_seconds`;
- `embedding_duration_seconds`, `retrieval_duration_seconds`;
- `reranking_duration_seconds`;
- `llm_ttft_seconds`, `llm_generation_duration_seconds`;
- `llm_tokens_generated_total`;
- `task_failures_total`.

Token count добавляется только когда runtime его действительно сообщает. Ollama
передаёт `eval_count` в финальном event. Для mock provider значение не
оценивается и не выдумывается.

Labels ограничены небольшими наборами: HTTP method, route template, status,
RAG mode/outcome и retryable. UUID, filename, query, exception message и raw URL
не являются labels, поскольку это привело бы к высокой cardinality или утечке
данных.

## Dashboard

Dashboard `AirGapRAG overview` создаётся автоматически и показывает:

- HTTP request rate и p95 latency;
- RAG outcomes;
- p95 latency отдельных RAG stages;
- LLM TTFT;
- ingestion и task failures.

Панели используют только реальные PromQL-запросы. Если трафика пока нет,
пустой график является корректным состоянием, а не причиной добавлять fake data.

## Ограничения

- Метрики сохраняются в локальном Prometheus volume без remote write.
- Нет distributed tracing: correlation сейчас реализована через structured IDs.
- Для нескольких worker-процессов внутри одного контейнера потребуется отдельная
  multiprocess-схема; текущий Compose запускает один worker process.
- Alertmanager и production alert rules пока не настроены.
