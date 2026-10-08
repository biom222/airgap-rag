# AirGapRAG

Локальный RAG-сервис для поиска и ответов по корпоративным документам в
изолированной инфраструктуре. Документы, embeddings и запросы к языковой модели
не должны передаваться во внешние AI API.

Проект разрабатывается поэтапно. Текущая версия включает foundation, приём
документов, локальные embeddings, Qdrant indexing, Redis/Taskiq background jobs,
debugging retrieval API и local LLM providers. RAG ещё не реализован.

## Текущие возможности

- FastAPI application factory и lifecycle;
- конфигурация через Pydantic Settings;
- структурированные JSON logs и `request_id`;
- `GET /health` для liveness;
- `GET /ready` с проверкой PostgreSQL, embeddings, Qdrant и LLM;
- async SQLAlchemy engine и `AsyncSession` factory;
- модели `Document` и `DocumentChunk`, управляемые Alembic;
- загрузка PDF, DOCX и UTF-8 TXT с ограничением размера;
- потоковый SHA-256, дедупликация и локальное файловое хранилище;
- независимые parsers для PDF, DOCX и TXT;
- конфигурируемый character chunking с overlap и привязкой к странице;
- `EmbeddingProvider` с deterministic mock и sentence-transformers adapter;
- Qdrant adapter с cosine collection и фильтрацией по `document_ids`;
- идемпотентный indexing service и retrieval endpoint;
- Redis Streams broker и отдельный Taskiq worker;
- сохраняемая в PostgreSQL job state machine, retries и exponential backoff;
- `GET /api/v1/jobs/{job_id}` для наблюдения за ingestion;
- `LLMProvider` с deterministic mock и Ollama adapter;
- полная и streaming generation через локальный Ollama HTTP API;
- Dockerfile и Docker Compose для API, worker, scheduler, PostgreSQL, Qdrant и Redis;
- pytest, Ruff, mypy, pre-commit и GitHub Actions.

## Архитектура

AirGapRAG строится как modular monolith. API и будущий background worker будут
разными процессами одной кодовой базы. PostgreSQL используется как источник
истины для metadata. Qdrant используется как производный vector index. Ollama
подключается через независимый provider adapter.

```text
HTTP client
    -> FastAPI router
    -> application service
    -> repository/provider interface
    -> PostgreSQL or local infrastructure
```

Подробности: [docs/architecture.md](docs/architecture.md).
Поток загрузки документов: [docs/document-ingestion.md](docs/document-ingestion.md).
Embeddings и vector indexing: [docs/vector-indexing.md](docs/vector-indexing.md).
Background jobs: [docs/background-jobs.md](docs/background-jobs.md).
Local LLM: [docs/local-llm.md](docs/local-llm.md).

## Требования

- Python 3.12 или 3.13;
- Docker с Compose plugin — для контейнерного запуска;
- PostgreSQL 17, Qdrant 1.19 и Redis 8 — если процессы запускаются без Compose.

## Локальная установка

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Скопируйте `.env.example` в `.env` и скорректируйте `DATABASE_URL`.

## Запуск

Локальный процесс:

```bash
python -m uvicorn airgap_rag.main:app --reload
```

Через Docker Compose:

```bash
docker compose up --build
```

После запуска:

- OpenAPI: `http://localhost:8000/docs`;
- liveness: `http://localhost:8000/health`;
- readiness: `http://localhost:8000/ready`.

Перед первым запуском API примените schema:

```bash
docker compose run --rm api python -m alembic upgrade head
```

`/health` не обращается к внешним зависимостям. `/ready` возвращает HTTP 503,
если недоступны PostgreSQL, embedding provider, Qdrant или LLM provider.

## Documents API

Загрузка документа:

```bash
curl -F "file=@./document.pdf" http://localhost:8000/api/v1/documents
```

Новый файл возвращает HTTP 202, `document_id`, `job_id` и статус `PENDING`.
Повтор файла с тем же SHA-256 возвращает HTTP 200, существующие identifiers и
`deduplicated: true`.

Получение metadata:

```bash
curl http://localhost:8000/api/v1/documents/<document_id>
```

Upload не выполняет parsing и chunking внутри HTTP request. После фиксации
metadata API отправляет `job_id` в Redis, а Taskiq worker запускает полный
pipeline. Состояние задания:

```bash
curl http://localhost:8000/api/v1/jobs/<job_id>
```

## Retrieval API

Endpoint предназначен для отладки retrieval до появления RAG:

```bash
curl -X POST http://localhost:8000/api/v1/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{"question":"срок хранения договора","top_k":5}'
```

Можно передать `document_ids`. Ответ содержит текст из PostgreSQL, page,
`chunk_id`, `chunk_index` и `vector_score`. Reranking появится позднее.

## Local embeddings

По умолчанию используется deterministic `MockEmbeddingProvider`, который нужен
для тестов pipeline и не подходит для смыслового поиска. Реальный provider
устанавливается отдельно:

```bash
python -m pip install -e ".[embeddings]"
```

При `AIRGAP_MODE=true` `EMBEDDING_MODEL_NAME_OR_PATH` обязан указывать на уже
существующий локальный каталог. Приложение не скачивает модель автоматически.

## Local LLM

По умолчанию используется `MockLLMProvider`. Он позволяет разрабатывать и
тестировать следующий RAG-этап без model runtime. Для Ollama сначала вручную
подготовьте модель:

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
```

Затем запустите stack с `LLM_PROVIDER=ollama`. Model pull не выполняется ни при
старте API, ни при readiness check. Подробности: [docs/local-llm.md](docs/local-llm.md).

## Миграции

```bash
python -m alembic upgrade head
```

В production startup не используется `Base.metadata.create_all()`. Изменения
schema будут выполняться только через Alembic revisions.

## Проверки

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m pytest
```

## Конфигурация

| Переменная | Назначение | Значение по умолчанию |
|---|---|---|
| `APP_NAME` | Название приложения | `AirGapRAG` |
| `APP_ENV` | `development`, `test`, `production` | `development` |
| `LOG_LEVEL` | Уровень логирования | `INFO` |
| `DATABASE_URL` | Async SQLAlchemy URL | локальный PostgreSQL |
| `DATABASE_POOL_SIZE` | Основной размер pool | `5` |
| `DATABASE_MAX_OVERFLOW` | Дополнительные соединения | `5` |
| `DATABASE_POOL_TIMEOUT_SECONDS` | Ожидание соединения | `10` |
| `DOCUMENT_STORAGE_PATH` | Корень локального document storage | `data/documents` |
| `DOCUMENT_MAX_UPLOAD_BYTES` | Максимальный размер upload | `26214400` |
| `DOCUMENT_MAX_DOCX_UNCOMPRESSED_BYTES` | Лимит распакованного DOCX | `104857600` |
| `DOCUMENT_READ_CHUNK_BYTES` | Размер блока streaming upload | `1048576` |
| `CHUNK_SIZE` | Целевой размер chunk в символах | `1200` |
| `CHUNK_OVERLAP` | Перекрытие соседних chunks | `200` |
| `AIRGAP_MODE` | Запрет загрузки embedding model из сети | `true` |
| `EMBEDDING_PROVIDER` | `mock` или `sentence_transformer` | `mock` |
| `EMBEDDING_MODEL_NAME_OR_PATH` | Локальный путь или model name | `models/embeddings/multilingual-e5-base` |
| `EMBEDDING_DIMENSION` | Размерность mock embeddings | `384` |
| `EMBEDDING_BATCH_SIZE` | Batch для sentence-transformers | `32` |
| `EMBEDDING_DEVICE` | Device inference | `cpu` |
| `QDRANT_URL` | Qdrant REST URL | `http://localhost:6333` |
| `QDRANT_COLLECTION` | Collection для chunks | `document_chunks` |
| `QDRANT_TIMEOUT_SECONDS` | Timeout Qdrant operations | `10` |
| `RETRIEVAL_TOP_K` | Default retrieval limit | `10` |
| `REDIS_URL` | Redis для Taskiq broker и retry schedule | `redis://localhost:6379/0` |
| `JOB_MAX_ATTEMPTS` | Максимальное число попыток ingestion | `3` |
| `JOB_TIMEOUT_SECONDS` | Timeout одной попытки | `900` |
| `JOB_RETRY_DELAY_SECONDS` | Начальная задержка retry | `5` |
| `JOB_RETRY_MAX_DELAY_SECONDS` | Верхняя граница backoff | `60` |
| `JOB_STALE_AFTER_SECONDS` | Возраст для повторного claim зависшего job | `960` |
| `LLM_PROVIDER` | `mock` или `ollama` | `mock` |
| `OLLAMA_BASE_URL` | URL локального Ollama runtime | `http://localhost:11434` |
| `OLLAMA_MODEL` | Имя заранее установленной модели | `qwen2.5:7b` |
| `OLLAMA_KEEP_ALIVE` | Удержание модели в памяти Ollama | `5m` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | Timeout generation request | `120` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.1` |
| `LLM_MAX_TOKENS` | Максимум generated tokens | `1024` |

## Структура

```text
src/airgap_rag/api/    HTTP boundary
src/airgap_rag/core/   configuration, lifecycle, logging, exceptions
src/airgap_rag/db/     SQLAlchemy infrastructure
src/airgap_rag/documents/ validation, storage, parsers, upload service
src/airgap_rag/chunking/  chunking contract and implementation
src/airgap_rag/embeddings/ embedding providers
src/airgap_rag/vector_store/ Qdrant adapter
src/airgap_rag/indexing/  indexing orchestration
src/airgap_rag/retrieval/ retrieval service
src/airgap_rag/jobs/      job state machine, broker and worker tasks
src/airgap_rag/llm/       LLM provider contract, mock and Ollama adapter
alembic/               database migrations
tests/                 automated tests
docs/                  Russian technical documentation and ADR
```

## Ограничения текущей версии

- DOCX parser извлекает paragraphs и tables, но не восстанавливает layout;
- scanned PDF без текстового слоя требует будущего OCR и сейчас даст пустой текст;
- default mock embeddings не обеспечивают semantic relevance;
- sentence-transformers package и model не входят в базовый Docker image;
- нет отдельной outbox-таблицы: если dispatch в Redis не удался, сохранённый
  `PENDING` job будет повторно отправлен при повторной загрузке того же файла;
- Ollama model weights требуют отдельной ручной подготовки;
- публичный chat endpoint и prompt builder появятся на RAG-этапе;
- реальная embedding model должна быть подготовлена отдельно.

## Roadmap

- [x] Architecture
- [x] Foundation
- [x] Documents and parsing
- [x] Embeddings and Qdrant
- [x] Background jobs
- [x] Local LLM
- [ ] RAG and citations
- [ ] SSE streaming
- [ ] Evaluation
- [ ] Benchmarking
- [ ] Observability
- [ ] Air-gapped deployment
- [ ] Additional inference runtimes
- [ ] Optional UI
