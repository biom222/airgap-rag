# Embeddings и vector indexing

## Providers

`EmbeddingProvider` определяет `embed_documents`, `embed_query`, `healthcheck`
и фактическую dimension.

- `MockEmbeddingProvider` создаёт deterministic normalized vectors. Он нужен для
  unit tests и проверки orchestration, но не является semantic model.
- `SentenceTransformerEmbeddingProvider` запускает локальную модель через
  sentence-transformers. В air-gap режиме путь проверяется до загрузки модели и
  передаётся `local_files_only=true`.

Document/query prefixes вынесены в configuration. Значения по умолчанию
соответствуют семейству multilingual E5, но model directory не скачивается и не
включается в repository.

## Qdrant collection

Collection `document_chunks` использует unnamed dense vector и cosine distance.
Dimension берётся у активного provider. Если существующая collection имеет
другую dimension или distance, initialization operation завершается явной ошибкой —
тихое смешивание несовместимых embeddings запрещено.

Payload:

```json
{
  "document_id": "uuid",
  "chunk_id": "uuid",
  "filename": "policy.pdf",
  "page": 14,
  "chunk_index": 21,
  "text_hash": "sha256"
}
```

Текст не дублируется в payload. PostgreSQL остаётся metadata source of truth.

## Idempotent indexing

1. Document переводится в `PARSING`, затем `CHUNKING`.
2. Existing PostgreSQL chunks удаляются и заменяются в одной транзакции.
3. После embedding статус меняется на `INDEXING`.
4. Qdrant points документа удаляются фильтром по `document_id`.
5. Chunks записываются через upsert с deterministic UUID.
6. После успешной записи Document получает `READY`.
7. Любая ошибка переводит Document в `FAILED`.

Повтор операции приводит к тому же набору chunk/vector IDs. Полноценные retries,
locking и job state machine относятся к PHASE 4.

## Retrieval

`POST /api/v1/retrieval/search` выполняет query embedding, cosine top-K и
необязательный Qdrant filter по `document_ids`. По найденным UUID текст chunks
загружается из PostgreSQL. Результат сохраняет порядок Qdrant и отдельно
возвращает `vector_score`.

Reranking на этом этапе отсутствует.
