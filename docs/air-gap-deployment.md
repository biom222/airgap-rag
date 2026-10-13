# Развёртывание AirGapRAG в изолированном контуре

## Модель поставки

AirGapRAG переносится в закрытый контур как заранее подготовленный release bundle. На
изолированной машине не выполняются `docker pull`, `pip install` из внешнего индекса или
`ollama pull`. Единицей поставки являются уже собранные OCI-образы, а не Docker build из
исходников.

Bundle содержит:

- архив всех Docker-образов, включая образ приложения;
- архив локального хранилища Ollama либо явно добавленные model files;
- Python wheelhouse для аудита и аварийного восстановления application layer;
- архив исходного кода, соответствующего release;
- SPDX 2.3 SBOM для контейнеров и Python dependencies;
- production/offline Compose-конфигурацию и конфигурации наблюдаемости;
- `manifest.json` с версиями, image IDs, digest и SHA-256 каждого артефакта;
- `SHA256SUMS` для независимой проверки после переноса.

Образ приложения собирается только на подключённой staging-машине. Offline Compose
сбрасывает все секции `build` и запрещает pulls через `pull_policy: never`. API и data plane
подключены только к сетям с `internal: true`; наружный `edge` есть только у минимального
HTTPS gateway, чтобы Docker мог опубликовать host-port.

## Подготовка на подключённой машине

Требования: Git, Python 3.12, работающий Docker Engine/Compose и достаточно места для
образов и model weights. Сначала загрузите нужную Ollama model и убедитесь, что она есть в
именованном volume проекта:

```powershell
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama list
```

Соберите release bundle. Скрипт сначала использует локальный каталог
`~/.ollama/models`; если его нет, ищет единственный Docker volume с окончанием
`_ollama_data`. Источник также можно указать явно:

```powershell
python scripts/offline_bundle.py build `
  --output offline_bundle/airgap-rag-0.1.0 `
  --ollama-model-dir "$env:USERPROFILE\.ollama\models" `
  --pull
```

`--pull` обновляет pinned images перед упаковкой. Все версии задаются в
`offline/images.lock.json`; теги `latest` валидатор запрещает. Для дополнительного GGUF или
другого файла используйте повторяемый параметр `--model-file PATH`.

Mock-only bundle допустим только как явное исключение для разработки:

```powershell
python scripts/offline_bundle.py build `
  --output offline_bundle/mock-only `
  --allow-empty-models
```

Такой bundle не считается готовым для production RAG с `LLM_PROVIDER=ollama`.

## Проверка и перенос

Перед записью на съёмный носитель:

```powershell
python scripts/offline_bundle.py validate offline_bundle/airgap-rag-0.1.0
pwsh scripts/validate-offline-bundle.ps1 `
  -BundleDirectory offline_bundle/airgap-rag-0.1.0
```

После физического переноса проверку нужно повторить уже в закрытом контуре. Проверяется
размер и SHA-256 каждого файла, затем hash самого manifest. Изменённый или неполный bundle
не должен устанавливаться.

Для Linux доступна проверка без Python:

```bash
sh deploy/scripts/validate-offline-bundle.sh /media/release/airgap-rag-0.1.0
```

Для чувствительной среды `SHA256SUMS` передают вторым доверенным каналом или подписывают
корпоративным ключом. SHA-256 обнаруживает повреждение, но сам по себе не подтверждает
автора поставки.

## Подготовка secrets

В каталоге `deploy` создайте `.secrets` до запуска. На Windows можно использовать:

```powershell
pwsh deploy/scripts/initialize-production.ps1 `
  -SecretDirectory deploy/.secrets `
  -DnsName rag.internal
```

Для Linux подготовьте те же файлы средствами корпоративного secret-management:

```text
postgres_password
redis_password
qdrant_api_key
auth_principals.json
grafana_admin_password
tls.crt
tls.key
```

Не переносите development `.env` и не включайте secrets в release bundle. Сертификат от
внутреннего CA предпочтительнее автоматически созданного self-signed сертификата.

## Запуск без доступа к registry

Windows, с Ollama и observability:

```powershell
pwsh deploy/scripts/start-offline.ps1 `
  -BundleDirectory C:\releases\airgap-rag-0.1.0 `
  -ProjectName airgap-rag `
  -WithOllama `
  -WithObservability
```

Linux, с Ollama и observability:

```bash
sh deploy/scripts/start-offline.sh \
  /srv/releases/airgap-rag-0.1.0 airgap-rag \
  --with-ollama --with-observability
```

Startup script повторно проверяет bundle, выполняет `docker load` и запускает Compose с
`--pull never`. Оба startup-скрипта восстанавливают Ollama archive только в пустой volume,
чтобы не перезаписать существующие models.

Проверка состояния:

```powershell
docker compose --project-name airgap-rag `
  --env-file deploy/.env.bundle `
  -f deploy/docker-compose.yml `
  -f deploy/docker-compose.production.yml `
  -f deploy/docker-compose.offline.yml ps
```

API публикуется только через HTTPS gateway. PostgreSQL, Redis, Qdrant, Ollama и monitoring
ports наружу не публикуются.

## Проверка отсутствия сетевых загрузок

До передачи release рекомендуется выполнить rehearsal на машине с отключённым внешним
сетевым интерфейсом или с deny-all egress firewall:

1. удалить тестовый Compose project и неиспользуемые тестовые images;
2. загрузить только `images/docker-images.tar`;
3. запустить через offline override и `--pull never`;
4. дождаться healthy/ready;
5. загрузить документ, дождаться `READY`, выполнить retrieval и chat;
6. сохранить протокол версий и healthchecks рядом с release approval.

Внутренние Docker networks блокируют egress API и data plane. HTTPS gateway подключён к
обычной `edge` bridge network ради публикации host-port, поэтому его egress нужно закрыть
host firewall. Docker-сети не заменяют физическую/сетевую изоляцию контура.

## Обновление и откат

Каждое обновление создаётся как новый неизменяемый bundle. Нельзя заменять отдельные файлы
в уже утверждённой поставке.

1. На staging checkout нужного Git commit.
2. Обновить pinned versions в `offline/images.lock.json`.
3. Собрать и протестировать новый bundle.
4. Проверить SBOM и vulnerability policy организации.
5. Сделать backup по `docs/production-hardening.md`.
6. Перенести bundle и выполнить checksum validation.
7. Выполнить migration/startup и smoke tests.

Для rollback сохраняют предыдущий bundle и совместимый backup PostgreSQL/documents. Откат
образов без отката необратимой database migration не гарантируется. Qdrant можно
восстановить reindex из PostgreSQL и document storage.

## Ограничения

- Размер bundle определяется суммой container images и model weights и может составлять
  десятки гигабайт.
- Wheelhouse рассчитан на Linux/x86-64 runtime, в котором он был собран; перенос wheels
  между архитектурами не поддерживается.
- Автоматическая криптографическая подпись release не реализована: используйте внутренний
  PKI/signing pipeline организации.
