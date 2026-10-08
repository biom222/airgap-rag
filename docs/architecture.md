# Архитектура AirGapRAG

## Текущий этап

На PHASE 3 реализованы foundation, documents boundary, embedding providers,
Qdrant adapter, indexing orchestration и retrieval debugging endpoint.

Taskiq, retries, LLM, reranking и RAG пока являются запланированными границами.

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

После подключения фоновой обработки одна кодовая база будет иметь две точки
запуска:

- API process принимает HTTP и SSE requests;
- worker process выполняет parsing, embeddings и indexing.

Это разделение процессов не превращает приложение в набор микросервисов:
deployment, domain model и repository остаются общими.

## Проверки состояния

`/health` является liveness probe. Он подтверждает, что процесс способен
обработать HTTP request, и не зависит от PostgreSQL.

`/ready` является readiness probe. Сейчас он выполняет `SELECT 1` через async
SQLAlchemy connection. Недоступность PostgreSQL даёт HTTP 503, но не завершает
процесс API.

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
