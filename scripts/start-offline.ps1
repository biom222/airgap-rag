param(
    [Parameter(Mandatory = $true)]
    [string]$BundleDirectory,
    [string]$ProjectName = "airgap-rag",
    [switch]$WithOllama,
    [switch]$WithObservability
)

$ErrorActionPreference = "Stop"
$bundle = [IO.Path]::GetFullPath($BundleDirectory)
$deploy = Join-Path $bundle "deploy"
$validator = Join-Path $deploy "scripts\validate-offline-bundle.ps1"
& $validator -BundleDirectory $bundle

$manifest = Get-Content -LiteralPath (Join-Path $bundle "manifest.json") -Raw | ConvertFrom-Json
$imageArchive = Join-Path $bundle "images\docker-images.tar"
docker load --input $imageArchive
if ($LASTEXITCODE -ne 0) {
    throw "docker load failed"
}

$model = $manifest.models |
    Where-Object { $_.kind -in @("ollama-volume", "ollama-model-store") } |
    Select-Object -First 1
if ($WithOllama -and $null -eq $model) {
    throw "The bundle has no Ollama model archive"
}
if ($WithOllama) {
    $volume = "${ProjectName}_ollama_data"
    docker volume create $volume | Out-Null
    $archiveDirectory = Split-Path -Parent (Join-Path $bundle ([string]$model.path))
    docker run --rm --user 0 `
        --volume "${volume}:/restore" `
        --volume "${archiveDirectory}:/bundle:ro" `
        ([string]$manifest.application_image) `
        sh -ec 'test -z "$(ls -A /restore)" || { echo "Ollama volume is not empty" >&2; exit 20; }; tar -xf /bundle/ollama-models.tar -C /restore'
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama model restore failed; use a new project name or an empty volume"
    }
}

$composeArguments = @(
    "compose",
    "--project-name", $ProjectName,
    "--env-file", (Join-Path $deploy ".env.bundle"),
    "--file", (Join-Path $deploy "docker-compose.yml"),
    "--file", (Join-Path $deploy "docker-compose.production.yml"),
    "--file", (Join-Path $deploy "docker-compose.offline.yml")
)
if ($WithOllama) {
    $composeArguments += @("--profile", "ollama")
}
if ($WithObservability) {
    $composeArguments += @("--profile", "observability")
}
$composeArguments += @("up", "--detach", "--pull", "never", "--wait")

Push-Location $deploy
try {
    & docker @composeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Offline Docker Compose startup failed"
    }
}
finally {
    Pop-Location
}

Write-Host "AirGapRAG started without registry access."
