# Локальная языковая модель

## Граница модуля

RAG-код не должен знать HTTP API конкретного inference runtime. Для этого
модуль `llm` предоставляет контракт `LLMProvider`:

```python
async def generate(messages: Sequence[LLMMessage]) -> str: ...
def stream(messages: Sequence[LLMMessage]) -> AsyncIterator[str]: ...
async def healthcheck() -> bool: ...
async def close() -> None: ...
```

`LLMMessage` содержит только роль и текст. Параметры конкретного runtime,
HTTP-клиент и формат wire response остаются внутри adapter.

## Реализации

`MockLLMProvider` возвращает детерминированный ответ. Он нужен unit-тестам и
локальной разработке остальных частей pipeline без модели, но не имитирует
качество реального LLM.

`OllamaProvider` использует локальные endpoints:

- `POST /api/chat` с `stream=false` для полного ответа;
- `POST /api/chat` с `stream=true` для последовательности JSON events;
- `GET /api/tags` для проверки runtime и наличия настроенной модели.

Provider различает недоступный runtime, отсутствующую модель и некорректный
ответ. Эти ошибки станут частью HTTP-контракта chat API на следующем этапе.

## Подготовка Ollama

Ollama вынесен в опциональный Compose profile. Обычный `docker compose up` не
запускает его и продолжает использовать mock.

Запуск runtime в online preparation mode:

```bash
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
```

Вторую команду нужно выполнять осознанно: она скачивает model weights. Само
приложение никогда не вызывает `ollama pull`.

После подготовки модели:

```bash
LLM_PROVIDER=ollama docker compose --profile ollama up --build
```

В PowerShell переменная задаётся перед командой:

```powershell
$env:LLM_PROVIDER = "ollama"
docker compose --profile ollama up --build
```

При offline runtime volume `ollama_data` уже должен содержать модель. Если
runtime недоступен или модель отсутствует, `/ready` возвращает HTTP 503 и
`checks.llm = "unavailable"`.

## Generation settings

- `OLLAMA_MODEL` — установленная локальная модель;
- `LLM_TEMPERATURE` — sampling temperature;
- `LLM_MAX_TOKENS` — передаётся Ollama как `num_predict`;
- `LLM_REQUEST_TIMEOUT_SECONDS` — HTTP timeout с учётом медленного CPU inference;
- `OLLAMA_KEEP_ALIVE` — время удержания модели в памяти runtime.

Prompt builder намеренно не входит в этот этап. System prompt, context и защита
от prompt injection будут добавлены в RAG boundary на PHASE 6.

## Ограничения

- model weights не входят в repository и Docker image;
- readiness проверяет наличие модели, но не запускает пробную генерацию;
- в PHASE 5 нет публичного generation endpoint;
- cancellation полного ответа будет связываться с HTTP request на PHASE 6;
- SSE и обработка disconnect относятся к PHASE 7.
