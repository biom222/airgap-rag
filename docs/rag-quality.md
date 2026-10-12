# RAG quality controls

## Hybrid retrieval

Каждый запрос выполняет dense search в Qdrant и lexical search PostgreSQL через
`websearch_to_tsquery('simple', ...)`. Ранжирования объединяются Reciprocal Rank
Fusion (`k=60`), после чего optional reranker получает единый candidate set.
PostgreSQL остаётся источником текста и повторно применяет security/metadata
filters перед возвратом chunks.

Фильтры: `document_ids`, `department`, `document_type`, диапазон
`document_date`, `tags`, owner и максимальный `access_level` текущего principal.

## Parsing, OCR and chunking

PDF извлекается в reading order с распознаванием heading по font metadata. Если
текстовый слой страницы короче настроенного порога, PyMuPDF запускает Tesseract
OCR. DOCX обрабатывается в исходном порядке paragraph/table, сохраняет headings,
lists и таблицы. Chunker переносит section и content type в PostgreSQL и Qdrant,
а заголовок добавляет к дочерним chunks.

## Claim-level citations

LLM обязан помечать содержательные предложения ссылками `[S1]`, `[S2]`. Сервер
проверяет существование source index и минимальное пересечение токенов утверждения
с соответствующим chunk. Неподтверждённый ответ заменяется безопасным сообщением
о недостатке контекста.

Это детерминированный guardrail, а не полноценный natural-language inference.
Regression dataset должен отдельно проверять сложные перефразирования.

## Generation regression evaluation

JSONL record:

```json
{"case_id":"retention","question":"Какой срок хранения?","reference_answer":"Срок хранения — пять лет.","relevant_chunk_ids":["00000000-0000-0000-0000-000000000000"]}
```

Запуск:

```bash
python -m airgap_rag.evaluation evaluation/datasets/generation.jsonl \
  --mode generation --api-token "$EVALUATION_API_TOKEN" \
  --min-answer-f1 0.6 --min-citation-precision 0.9 \
  --min-citation-recall 0.8 --min-faithfulness 0.9
```

CLI сохраняет JSON/Markdown report и возвращает exit code `2`, если хотя бы одна
aggregate metric ниже threshold. Это позволяет блокировать regression в CI.
