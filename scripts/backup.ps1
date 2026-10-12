param(
    [string]$Destination = "backups",
    [string]$ProjectName = ""
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$backupDirectory = Join-Path $Destination $stamp
New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
$resolved = (Resolve-Path $backupDirectory).Path
$compose = @("compose")
if ($ProjectName) { $compose += @("-p", $ProjectName) }
$compose += @("-f", "docker-compose.yml", "-f", "docker-compose.production.yml")

$databasePath = Join-Path $resolved "postgres.sql"
& docker @compose exec -T postgres pg_dump -U airgap --clean --if-exists --no-owner airgap_rag |
    Set-Content -LiteralPath $databasePath -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "PostgreSQL backup failed" }

$documentPath = Join-Path $resolved "documents"
New-Item -ItemType Directory -Force -Path $documentPath | Out-Null
& docker @compose cp "api:/data/documents/." $documentPath
if ($LASTEXITCODE -ne 0) { throw "Document backup failed" }

$metadata = [ordered]@{
    created_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
    database = "postgres.sql"
    documents = "documents"
    qdrant = "derived; rebuild with scripts/reindex.ps1"
} | ConvertTo-Json
[IO.File]::WriteAllText(
    (Join-Path $resolved "backup.json"),
    $metadata,
    [Text.UTF8Encoding]::new($false)
)

Get-ChildItem -LiteralPath $resolved -File -Recurse |
    Where-Object Name -ne "SHA256SUMS" |
    ForEach-Object {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
        $relative = [IO.Path]::GetRelativePath($resolved, $_.FullName).Replace("\", "/")
        "$hash  $relative"
    } | Set-Content -LiteralPath (Join-Path $resolved "SHA256SUMS") -Encoding utf8

Write-Host "Backup created: $resolved"
