# ADR 0002: использовать PostgreSQL как источник metadata

## Статус

Принято.

## Контекст

Системе нужны транзакционные состояния документов и jobs, ограничения
дедупликации, связи chunks с документами и история chat/benchmark records.
Vector database не предназначена для всех этих задач.

## Решение

Хранить authoritative metadata в PostgreSQL. В будущих phases Qdrant будет
восстанавливаемым поисковым индексом, Redis — транспортом background jobs.

## Обоснование

PostgreSQL предоставляет транзакции, constraints, indexes и понятные migrations.
Состояние системы можно восстановить и диагностировать независимо от Redis и
Qdrant.

## Рассмотренные альтернативы

- Хранить metadata только в Qdrant payload.
- Использовать Redis как постоянное хранилище jobs.
- Использовать PostgreSQL с pgvector вместо Qdrant.

## Последствия

- Появляется задача согласования PostgreSQL с производными хранилищами.
- Background operations должны быть идемпотентными.
- Все изменения relational schema выполняются через Alembic.
