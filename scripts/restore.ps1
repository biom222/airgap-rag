param(
    [Parameter(Mandatory)]
    [string]$BackupPath,
    [switch]$ConfirmRestore,
    [string]$ProjectName = ""
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmRestore) {
    throw "Restore replaces current PostgreSQL data and documents. Re-run with -ConfirmRestore."
}
$resolved = (Resolve-Path $BackupPath).Path
$databasePath = Join-Path $resolved "postgres.sql"
$documentPath = Join-Path $resolved "documents"
if (-not (Test-Path $databasePath) -or -not (Test-Path $documentPath)) {
    throw "Backup must contain postgres.sql and documents/."
}
$checksumPath = Join-Path $resolved "SHA256SUMS"
if (-not (Test-Path $checksumPath)) { throw "Backup must contain SHA256SUMS." }
$rootPrefix = $resolved.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
foreach ($line in Get-Content -LiteralPath $checksumPath) {
    if ($line -notmatch "^([0-9a-fA-F]{64})  (.+)$") {
        throw "Invalid checksum line: $line"
    }
    $expected = $Matches[1].ToLowerInvariant()
    $relative = $Matches[2].Replace("/", [IO.Path]::DirectorySeparatorChar)
    $candidate = [IO.Path]::GetFullPath((Join-Path $resolved $relative))
    if (-not $candidate.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Checksum path escapes the backup directory: $relative"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "Backup file is missing: $relative"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $candidate).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Checksum mismatch: $relative" }
}

$compose = @("compose")
if ($ProjectName) { $compose += @("-p", $ProjectName) }
$compose += @("-f", "docker-compose.yml", "-f", "docker-compose.production.yml")

& docker @compose stop api worker scheduler outbox
if ($LASTEXITCODE -ne 0) { throw "Failed to stop application services" }

Get-Content -LiteralPath $databasePath -Raw |
    & docker @compose exec -T postgres psql -v ON_ERROR_STOP=1 -U airgap airgap_rag
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL restore failed" }

& docker @compose start api
if ($LASTEXITCODE -ne 0) { throw "Failed to start API container for document restore" }
& docker @compose exec -T --user root api sh -c "find /data/documents -mindepth 1 -delete"
if ($LASTEXITCODE -ne 0) { throw "Failed to clear document volume" }
& docker @compose cp "$documentPath/." "api:/data/documents"
if ($LASTEXITCODE -ne 0) { throw "Document restore failed" }
& docker @compose exec -T --user root api chown -R airgap:airgap /data/documents
if ($LASTEXITCODE -ne 0) { throw "Failed to restore document ownership" }

& docker @compose exec -T api /usr/local/bin/airgap-rag-entrypoint `
    python -m airgap_rag.maintenance.reindex
if ($LASTEXITCODE -ne 0) { throw "Qdrant reset or reindex scheduling failed" }

& docker @compose start worker scheduler outbox
if ($LASTEXITCODE -ne 0) { throw "Failed to start background services" }
Write-Host "Restore complete. Qdrant was reset and reindex jobs were queued."
