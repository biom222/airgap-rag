# Архитектура AirGapRAG

## Текущий этап

На PHASE 6 реализованы foundation, ingestion, indexing, background jobs,
retrieval, optional reranking, local LLM provider boundary и синхронный RAG API
с историей диалога и server-derived citations.

## Направление зависимостей

```text
API schemas and routers
        ↓
application services
        ↓
domain contracts
        ↓
database repositories and infrastructure providers
```

FastAPI routers отвечают за HTTP: parsing запроса, status codes и response
schemas. Бизнес-правила будут находиться в services. Работа с PostgreSQL,
Qdrant, Redis и model runtimes будет изолирована в adapters.

## Процессы

Одна кодовая база имеет три собственные точки запуска:

- API process принимает HTTP и SSE requests;
- worker process выполняет parsing, embeddings и indexing.
- scheduler process публикует отложенные retries из Redis schedule source.

Это разделение процессов не превращает приложение в набор микросервисов:
deployment, domain model и repository остаются общими.

Ollama является отдельным инфраструктурным runtime, доступным через локальный
HTTP API. `LLMProvider` изолирует application layer от его wire protocol.

## Проверки состояния

`/health` является liveness probe. Он подтверждает, что процесс способен
обработать HTTP request, и не зависит от PostgreSQL.

`/ready` является readiness probe. Он проверяет PostgreSQL, embedding provider,
Qdrant и LLM provider. Для Ollama проверка включает наличие настроенной модели.
Недоступность зависимости даёт HTTP 503, но не завершает процесс API.

## Управление schema

SQLAlchemy models описывают отображение объектов на таблицы, но не создают
schema при старте. Alembic является единственным механизмом изменения database
schema. Первая migration создаёт `documents` и `document_chunks`.

Уникальный constraint по `documents.sha256` является окончательной защитой от
concurrent duplicate uploads; предварительный `SELECT` нужен только для быстрого
обычного пути.

## Documents boundary

```text
multipart upload
    -> DocumentValidator
    -> LocalDocumentStore staging + SHA-256
    -> DocumentService deduplication
    -> PostgreSQL Document metadata

stored file
    -> DocumentParser
    -> ParsedDocument / ParsedPage
    -> Chunker
    -> Chunk with deterministic UUID and text_hash
```

Имя клиента хранится только как metadata. Путь файла строится сервером из UUID,
SHA-256 и проверенного расширения. Временный файл перемещается атомарно внутри
одного filesystem. Parsing и chunking не выполняются HTTP router: в PHASE 4 их
будет вызывать background worker.

## Vector indexing boundary

```text
stored document
    -> parser
    -> chunker
    -> PostgreSQL document_chunks
    -> EmbeddingProvider
    -> Qdrant delete by document_id
    -> deterministic vector upsert
    -> Document READY

question
    -> EmbeddingProvider.embed_query
    -> Qdrant cosine top-K + document filter
    -> PostgreSQL chunk lookup
    -> retrieval debugging response
```

Qdrant является производным индексом. Полный текст возвращается из PostgreSQL,
а Qdrant payload содержит identifiers и metadata. Потерянную collection можно
восстановить повторной индексацией документов.

## Background jobs boundary

```text
POST /documents
    -> PostgreSQL: Document + Job(PENDING)
    -> Redis Stream: job_id
    -> Taskiq worker
    -> JobExecutionService
    -> IndexingService
    -> Job(READY) + Document(READY)
```

Redis отвечает за доставку, но не является источником истины для состояния.
Taskiq использует acknowledgements и допускает повторную доставку. Claim job и
номер попытки фиксируются в PostgreSQL; параллельная duplicate delivery
пропускается. После process crash старый active job разрешено забрать повторно
только после `JOB_STALE_AFTER_SECONDS`.

Подробнее: [background-jobs.md](background-jobs.md).

## Local LLM boundary

```text
RAGService
    -> LLMProvider
        -> MockLLMProvider
        -> OllamaProvider
            -> local Ollama /api/chat
```

Provider поддерживает полный ответ и async token stream. Prompt construction не
смешивается с runtime adapter. Ollama запускается опционально; приложение не
скачивает model weights автоматически.

Подробнее: [local-llm.md](local-llm.md).

## RAG boundary

```text
POST /api/v1/chat
    -> session history from PostgreSQL
    -> embedding query
    -> Qdrant top-K
    -> authoritative chunk text from PostgreSQL
    -> optional local reranker -> top-N
    -> PromptBuilder
    -> LLMProvider.generate
    -> persist exchange
    -> answer + server-derived sources
```

Document content помещается в отдельный JSON context и явно обозначается как
недоверенные данные. `sources` строятся из chunks, выбранных PromptBuilder, и не
зависят от того, правильно ли LLM напечатала маркеры `[S1]`. История хранится в
`chat_sessions` и `chat_messages`; вызов модели не удерживает транзакцию БД.

Подробнее: [rag-pipeline.md](rag-pipeline.md).
