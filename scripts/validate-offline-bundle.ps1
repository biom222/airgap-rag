param(
    [Parameter(Mandatory = $true)]
    [string]$BundleDirectory
)

$ErrorActionPreference = "Stop"
$bundle = [IO.Path]::GetFullPath($BundleDirectory)
$manifestPath = Join-Path $bundle "manifest.json"
$checksumsPath = Join-Path $bundle "SHA256SUMS"

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "manifest.json is missing from $bundle"
}
if (-not (Test-Path -LiteralPath $checksumsPath -PathType Leaf)) {
    throw "SHA256SUMS is missing from $bundle"
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1) {
    throw "Unsupported manifest schema: $($manifest.schema_version)"
}

$verified = 0
foreach ($artifact in $manifest.artifacts) {
    $relative = [string]$artifact.path
    if ([IO.Path]::IsPathRooted($relative) -or $relative.Split('/').Contains('..')) {
        throw "Unsafe artifact path in manifest: $relative"
    }
    $path = Join-Path $bundle ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing artifact: $relative"
    }
    $item = Get-Item -LiteralPath $path
    if ($item.Length -ne [long]$artifact.size) {
        throw "Size mismatch: $relative"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne [string]$artifact.sha256) {
        throw "SHA-256 mismatch: $relative"
    }
    $verified += 1
}

$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumLines = Get-Content -LiteralPath $checksumsPath
$expectedManifestLine = "$manifestHash  manifest.json"
if ($checksumLines -notcontains $expectedManifestLine) {
    throw "SHA256SUMS does not contain the current manifest hash"
}

Write-Host "Offline bundle valid: $verified artifacts verified."
