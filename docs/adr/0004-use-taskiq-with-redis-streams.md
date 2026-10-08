# ADR 0004: Taskiq с Redis Streams для background jobs

## Статус

Принято.

## Контекст

Parsing, embeddings и indexing нельзя выполнять внутри upload request. Нужны
async-native worker, acknowledgement, retries и отдельный process без перехода
к микросервисной архитектуре.

## Решение

Использовать Taskiq и `RedisStreamBroker`. PostgreSQL хранит состояние job,
Redis отвечает только за доставку. Отложенные retries выполняются через
`SmartRetryMiddleware` и Redis-backed schedule source.

## Альтернативы

- Celery: зрелее и функциональнее, но несёт больше sync-first legacy и
  конфигурации, чем требуется этому async FastAPI проекту.
- ARQ: проще, но Taskiq даёт явные broker abstractions и scheduler.
- FastAPI `BackgroundTasks`: выполняется в API process, не переживает restart и
  не даёт durable delivery.
- PostgreSQL polling queue: уменьшает число компонентов, но потребовал бы
  самостоятельно реализовать delivery, polling, retries и scheduling.

## Последствия

- deployment получает Redis, worker и один scheduler process;
- delivery остаётся at-least-once, поэтому handlers обязаны быть идемпотентными;
- job state нельзя восстанавливать только из Redis;
- delayed retry зависит от единственного scheduler process;
- для устранения последнего dual-write окна позже может понадобиться
  transactional outbox.
