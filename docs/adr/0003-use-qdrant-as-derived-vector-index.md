# ADR 0003: Qdrant как производный vector index

## Status

Accepted.

## Context

Semantic retrieval требует специализированного vector index, но document и
chunk metadata должны оставаться консистентными и восстанавливаемыми.

## Decision

PostgreSQL хранит документы и полный текст chunks. Qdrant хранит embeddings и
минимальный payload для фильтрации и диагностики. Vector IDs детерминированы из
chunk IDs. Collection может быть полностью пересоздана из PostgreSQL и файлового
хранилища.

## Alternatives

- pgvector упростил бы deployment, но проект должен отдельно показать работу со
  специализированным vector database и его failure modes.
- хранение полного текста только в Qdrant сделало бы производный индекс вторым
  источником истины.
- случайные vector IDs усложнили бы retry и reindex.

## Consequences

- появляется отдельная инфраструктурная зависимость и eventual consistency;
- indexing обязан быть идемпотентным;
- retrieval выполняет Qdrant search и PostgreSQL lookup;
- backup Qdrant полезен, но не обязателен для восстановления данных.
