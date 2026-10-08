# Background jobs

## Компоненты

- PostgreSQL хранит `jobs` и является источником истины для состояния;
- Redis Stream доставляет сообщения с `job_id`;
- Taskiq worker выполняет ingestion;
- Taskiq scheduler публикует отложенные retries;
- API создаёт job и не ждёт parsing или indexing.

В queue message нет текста документа, credentials или сериализованного ORM
объекта. Worker получает актуальные данные из PostgreSQL.

## State machine

```text
PENDING
  -> PARSING
  -> CHUNKING
  -> EMBEDDING
  -> INDEXING
  -> READY

PENDING|PARSING|CHUNKING|EMBEDDING|INDEXING -> FAILED
```

`READY` и `FAILED` терминальны для обычного запуска. Возврат active job в
`PENDING` существует только как отдельная retry operation. Поэтому случайный
переход `READY -> PARSING` невозможен.

`progress` является coarse-grained progress этапа, а не процентом обработанных
страниц: 0, 10, 30, 50, 80 и 100.

## Delivery и idempotency

Redis Stream даёт at-least-once delivery. Ровно один запуск не предполагается.
Перед работой worker делает row-level claim в PostgreSQL:

1. terminal job пропускается;
2. свежий active job считается уже выполняющимся;
3. `PENDING` job переводится в `PARSING`, `attempt` увеличивается;
4. зависший active job можно повторно забрать после stale timeout.

Повторный indexing безопасен: chunks имеют deterministic UUID, строки chunks
заменяются, vectors старого документа удаляются, затем выполняется upsert с теми
же identifiers. Частично записанный Qdrant index не создаёт duplicates.

## Retry policy

Permanent errors не повторяются:

- повреждённый документ;
- отсутствие исходного файла;
- пустой извлечённый текст;
- некорректная embedding model или размерность.

Transient errors повторяются ограниченно:

- timeout;
- временная недоступность Redis, PostgreSQL или Qdrant;
- неожиданный infrastructure failure.

Taskiq `SmartRetryMiddleware` использует jitter и exponential backoff. После
`JOB_MAX_ATTEMPTS` job и document переходят в `FAILED`; код и безопасное
сообщение ошибки сохраняются в PostgreSQL.

## Transaction boundary

`Document` и `Job` создаются одной database transaction. Отправка в Redis идёт
после commit, потому что Redis и PostgreSQL не поддерживают общую транзакцию.
Если dispatch не удался, API возвращает 503, но файл и `PENDING` job остаются;
повторная загрузка того же SHA-256 повторно отправит существующий job.

Полный transactional outbox пока не добавлен: он оправдан, когда потребуется
автоматическое восстановление dispatch без повторного клиентского запроса.
