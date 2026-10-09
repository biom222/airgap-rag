# Оценка качества retrieval

Фаза evaluation измеряет только качество ранжирования chunks. Скорость и
ресурсы относятся к benchmarking и здесь не смешиваются с quality metrics.

## Dataset

Dataset хранится в JSONL: одна строка — один независимо размеченный вопрос.

```json
{
  "case_id": "retention-period",
  "question": "Какой срок хранения договора?",
  "relevant_chunk_ids": ["<chunk-uuid>"],
  "document_ids": ["<optional-document-uuid>"]
}
```

- `case_id` уникален внутри файла;
- `question` передаётся retrieval API без изменения;
- `relevant_chunk_ids` содержит один или несколько допустимых ответов;
- `document_ids` опционально ограничивает поиск тем же способом, что production API.

`evaluation/datasets/retrieval.example.jsonl` показывает формат. UUID в нём —
заглушки, их нужно заменить ID из собственной проиндексированной коллекции.
Dataset следует версионировать вместе с corpus и параметрами chunking. После
изменения `CHUNK_SIZE`, `CHUNK_OVERLAP` или исходных документов разметку нужно
проверить заново.

## CLI

При запущенном API:

```bash
python -m airgap_rag.evaluation evaluation/datasets/retrieval.jsonl \
  --base-url http://localhost:8000 \
  --top-k 1 3 5 10 \
  --output-dir evaluation/results
```

CLI последовательно выполняет запросы, чтобы локальный embedding/reranker
runtime не получал искусственную конкурентную нагрузку. Для каждого вопроса API
возвращает до `max(K)` chunks. Параметры `top_k` и `top_n` передаются одинаковыми,
поэтому включённый reranker не обрежет ranking раньше требуемой глубины.

Base URL можно задать через `EVALUATION_BASE_URL`. Значение CLI имеет приоритет.
При невалидной строке dataset, HTTP-ошибке или некорректном response запуск
завершается ошибкой: такие случаи не исключаются молча из denominator.

## Метрики

`Hit@K` равен доле вопросов, для которых хотя бы один релевантный chunk попал в
первые K результатов.

`MRR` использует позицию первого релевантного chunk. Поскольку API запрашивается
с ограниченной глубиной, отчёт явно называет метрику `MRR@max(K)`. Если
релевантного chunk нет в этом диапазоне, reciprocal rank равен нулю.

## Reports

Каждый запуск создаёт JSON и Markdown с одинаковым timestamp. JSON содержит
aggregate metrics и ranking каждого case; Markdown предназначен для быстрого
review. `evaluation/results/` исключён из Git, чтобы случайный локальный прогон
не выглядел как зафиксированный benchmark.

Для публикации результата нужно осознанно сохранить проверенный отчёт в другом
каталоге вместе с описанием corpus, модели embeddings, reranker и параметров
chunking. Маленький демонстрационный dataset не является доказательством
production-качества.
