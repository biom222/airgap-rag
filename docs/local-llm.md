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
ответ. Обычный chat endpoint преобразует их в HTTP errors, а уже открытый SSE
stream — в событие `error`.

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

Prompt builder находится в RAG boundary и передаёт provider уже готовую
последовательность сообщений.

## Ограничения

- model weights не входят в repository и Docker image;
- readiness проверяет наличие модели, но не запускает пробную генерацию;
- синхронный endpoint не может прервать уже отправленный upstream request после
  disconnect клиента;
- SSE boundary закрывает provider stream при disconnect клиента.
