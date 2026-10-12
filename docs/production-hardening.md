# Production hardening

## Authentication and data isolation

Business endpoints require a Bearer API key when `AUTH_REQUIRED=true`. Raw keys
are never stored in configuration: `AUTH_PRINCIPALS_JSON` contains only a
SHA-256 digest, a stable subject and role `user` or `admin`.

Documents and chat sessions carry an `owner_id`. A `user` can read and search
only its own documents and continue only its own chat sessions. An `admin` may
access all owners. Qdrant has the same owner filter, while PostgreSQL remains the
source of truth and filters loaded chunks again.

`GET /health`, `GET /ready` and the internal Prometheus endpoint remain
unauthenticated for orchestrator probes. `/system/info` and all `/api/v1/*`
operations require authentication in production.

Interactive API documentation and the OpenAPI schema are disabled in
production. Nginx also blocks `/metrics` on the public TLS gateway; Prometheus
scrapes that endpoint directly over the internal backend network.

Initialize secrets and a local self-signed certificate:

```powershell
./scripts/initialize-production.ps1 -Subject systemx-admin
```

The client token is written to ignored `.secrets/admin_api_token`. Move it to a
client secret store and remove that file after enrollment. Replace the generated
self-signed certificate with a certificate issued by the organization's CA.

## Network boundary

Start the hardened overlay with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.production.yml `
  --profile observability up -d --build
```

Only the Nginx TLS gateway publishes a host port (`443`). PostgreSQL, Redis,
Qdrant, Prometheus, Alertmanager, Jaeger and Grafana are reachable only on the
internal Docker network. Use an authenticated administrative tunnel when direct
access to Grafana or Jaeger is required.

If port 443 is already occupied, set `AIRGAP_HTTPS_PORT` before starting the
stack (for example, `$env:AIRGAP_HTTPS_PORT = "18443"`). TLS remains enabled;
clients then use that explicit port.

Redis requires a password, Qdrant requires an API key and PostgreSQL uses a
random password. Values are mounted as Docker secrets. Application containers
are read-only and run with `no-new-privileges`.

## Transactional outbox

Document metadata, the ingestion job and `ingestion.requested` outbox event are
committed in one PostgreSQL transaction. The `outbox` service locks pending
events with `FOR UPDATE SKIP LOCKED`, publishes them to Taskiq and marks them as
processed. Delivery is at-least-once; the job claim state machine makes duplicate
delivery safe. Failed delivery remains pending and uses capped exponential
backoff.

W3C trace context and the application correlation ID travel in the outbox event
and continue in the worker.

## Backup and restore

Create a backup:

```powershell
./scripts/backup.ps1
```

The backup includes a PostgreSQL logical dump, original documents, metadata and
SHA-256 checksums. Qdrant is deliberately treated as a derived index.

Restore is destructive and therefore requires an explicit switch:

```powershell
./scripts/restore.ps1 -BackupPath backups/<timestamp> -ConfirmRestore
```

Restore verifies every SHA-256 checksum before changing the running system,
resets the derived Qdrant collection and atomically queues reindex jobs before
background workers resume. `scripts/reindex.ps1` remains available for a manual
full reindex when no database restore is required.

Backups must be copied to encrypted storage and restoration must be tested on a
separate host. A backup is not considered valid until checksum verification and
a test restore succeed.

## Tracing and alerts

OpenTelemetry instruments FastAPI, HTTPX and SQLAlchemy. Explicit spans cover
embedding, retrieval, reranking, LLM generation and ingestion. Jaeger receives
OTLP/HTTP traces. Prometheus loads alert rules for API/worker availability, HTTP
5xx rate, RAG latency and ingestion/task failures; Alertmanager groups the
alerts locally without sending data outside the air-gapped environment.

No document text, prompts, model answers, API keys or passwords are attached to
spans, logs or metric labels.
