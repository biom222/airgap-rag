# Бенчмарки

Бенчмарки запускаются вручную на целевой машине. Они не входят в unit tests:
результат зависит от модели, runtime, нагрузки и аппаратной конфигурации.
Сгенерированные JSON-файлы находятся в `benchmarks/results/` и не попадают в Git.

## Сведения о системе

`GET /system/info` возвращает версии Python и ОС, CPU, RAM, сведения о первой
NVIDIA GPU, VRAM, доступность CUDA и выбранные LLM, embedding и reranking
providers. Если `nvidia-smi` отсутствует или недоступен, endpoint возвращает
`gpu: null`, `vram_total_mb: null` и `cuda_available: false`; это штатный режим.

## LLM benchmark

```bash
python -m benchmarks.llm \
  --base-url http://localhost:11434 \
  --model qwen2.5:7b \
  --quantization Q4_K_M \
  --max-tokens 256 \
  --ollama-base-url http://localhost:11434 \
  --warmup 1 \
  --iterations 5
```

Измеряются wall-clock TTFT и общая latency. Число input/output tokens и
tokens/sec берутся из финального streaming event Ollama. Размер загруженной
модели и распределение RAM/VRAM берутся из `/api/ps`. Если runtime не вернул
token metrics, запуск завершается ошибкой: приблизительные значения не
подставляются.

Флаг `--persist-db` сохраняет каждую измеренную итерацию в
`model_benchmarks`. Перед этим нужна миграция `alembic upgrade head`. Без флага
результат сохраняется только в JSON.

## RAG benchmark

```bash
python -m benchmarks.rag \
  --question "Какой срок хранения договора?" \
  --document-id 00000000-0000-0000-0000-000000000000 \
  --max-tokens 256 \
  --warmup 1 \
  --iterations 5
```

Runner использует providers из текущего `.env` и тот же `RetrievalService` и
`PromptBuilder`, что production API. Отдельно измеряются embedding, retrieval
(Qdrant + чтение chunks из PostgreSQL), reranking, generation и end-to-end
latency. Подготовка prompt входит только в total. Если chunks не найдены, runner
завершается ошибкой, чтобы не выдавать пустой pipeline за RAG-замер.

## Сравнимость результатов

- Используйте одинаковый prompt/question и одинаковое число итераций.
- Указывайте фактическую quantization явно; `unknown` означает, что она не
  подтверждена.
- Сначала выполните warm-up, чтобы загрузка модели не искажала измерения.
- Во время сравнений не запускайте тяжёлые фоновые задачи.
- Сравнивайте JSON вместе со снимком `hardware`, а не только средние значения.
- Mock embeddings и mock reranker пригодны для проверки pipeline, но не для
  выводов о production quality или latency реальной модели.

Метрики показывают один конкретный запуск и не являются обещанием
производительности на другом оборудовании.
