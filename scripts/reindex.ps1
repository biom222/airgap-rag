param(
    [string]$ProjectName = ""
)

$ErrorActionPreference = "Stop"
$compose = @("compose")
if ($ProjectName) { $compose += @("-p", $ProjectName) }
$compose += @("-f", "docker-compose.yml", "-f", "docker-compose.production.yml")
& docker @compose exec -T api /usr/local/bin/airgap-rag-entrypoint `
    python -m airgap_rag.maintenance.reindex
if ($LASTEXITCODE -ne 0) { throw "Reindex scheduling failed" }
Write-Host "Reindex jobs were written to the transactional outbox."
