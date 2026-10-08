# SSE streaming

## Endpoint

`POST /api/v1/chat/stream` принимает тот же JSON, что и синхронный chat:

```json
{
  "question": "Какой срок хранения договора?",
  "session_id": null,
  "document_ids": ["<document_id>"]
}
```

Response имеет `Content-Type: text/event-stream`, `Cache-Control: no-cache` и
`X-Accel-Buffering: no`.

## События

Первым приходит список серверных источников:

```text
event: sources
data: {"sources":[{"document_id":"...","filename":"policy.pdf","page":14,"chunk_id":"...","score":0.91,"rerank_score":null}]}
```

Затем приходят фрагменты ответа:

```text
event: token
data: {"text":"Срок хранения "}
```

После полной генерации и записи истории приходит:

```text
event: done
data: {"session_id":"..."}
```

Если provider завершился ошибкой после HTTP 200, поток заканчивается событием:

```text
event: error
data: {"code":"llm_unavailable","message":"...","request_id":"..."}
```

`done` после `error` не отправляется.

## Ошибки до и после открытия потока

Validation, проверка `session_id`, retrieval и prompt preparation выполняются до
создания `StreamingResponse`. Поэтому неизвестная сессия возвращает обычный HTTP
404. Ошибку Ollama можно обнаружить только во время чтения provider stream, когда
HTTP 200 уже отправлен; поэтому она кодируется как SSE `error`.

## Disconnect

Перед отправкой каждого события endpoint проверяет disconnect клиента. При
разрыве соединения domain stream закрывается, что завершает чтение Ollama HTTP
response. Частичный текст не записывается в `chat_messages`; сохранение пары
user/assistant происходит только после нормального окончания генерации.

Это предотвращает бесконтрольное продолжение inference после ухода клиента и не
загрязняет историю незавершёнными ответами.
