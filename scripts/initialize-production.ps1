param(
    [string]$Subject = "systemx-admin",
    [ValidateSet("admin", "user")]
    [string]$Role = "admin",
    [string]$DnsName = "localhost",
    [string]$SecretDirectory = (Join-Path $PSScriptRoot "..\.secrets")
)

$ErrorActionPreference = "Stop"
$secretDirectory = [IO.Path]::GetFullPath($SecretDirectory)
New-Item -ItemType Directory -Force -Path $secretDirectory | Out-Null

function New-RandomHex([int]$Bytes) {
    $buffer = [byte[]]::new($Bytes)
    [Security.Cryptography.RandomNumberGenerator]::Fill($buffer)
    return [Convert]::ToHexString($buffer).ToLowerInvariant()
}

function Write-Secret([string]$Name, [string]$Value) {
    $path = Join-Path $secretDirectory $Name
    [IO.File]::WriteAllText($path, $Value, [Text.UTF8Encoding]::new($false))
}

Write-Secret "postgres_password" (New-RandomHex 32)
Write-Secret "redis_password" (New-RandomHex 32)
Write-Secret "qdrant_api_key" (New-RandomHex 32)
Write-Secret "grafana_admin_password" (New-RandomHex 32)

$apiToken = New-RandomHex 32
Write-Secret "admin_api_token" $apiToken
$tokenBytes = [Text.Encoding]::UTF8.GetBytes($apiToken)
$tokenHash = [Convert]::ToHexString(
    [Security.Cryptography.SHA256]::HashData($tokenBytes)
).ToLowerInvariant()
$principalRecords = @(
    [ordered]@{
        subject = $Subject
        role = $Role
        token_sha256 = $tokenHash
    }
)
$principals = ConvertTo-Json -InputObject $principalRecords -Compress
Write-Secret "auth_principals.json" $principals

$rsa = [Security.Cryptography.RSA]::Create(3072)
$request = [Security.Cryptography.X509Certificates.CertificateRequest]::new(
    "CN=$DnsName",
    $rsa,
    [Security.Cryptography.HashAlgorithmName]::SHA256,
    [Security.Cryptography.RSASignaturePadding]::Pkcs1
)
$san = [Security.Cryptography.X509Certificates.SubjectAlternativeNameBuilder]::new()
$san.AddDnsName($DnsName)
if ($DnsName -eq "localhost") {
    $san.AddIpAddress([Net.IPAddress]::Loopback)
}
$request.CertificateExtensions.Add($san.Build())
$certificate = $request.CreateSelfSigned(
    [DateTimeOffset]::UtcNow.AddMinutes(-5),
    [DateTimeOffset]::UtcNow.AddYears(1)
)
Write-Secret "tls.crt" $certificate.ExportCertificatePem()
Write-Secret "tls.key" $rsa.ExportPkcs8PrivateKeyPem()

Write-Host "Production secrets created in $secretDirectory."
Write-Host "API token for $Subject was written to $(Join-Path $secretDirectory 'admin_api_token')."
Write-Warning "Move that token to the client secret store and remove the file after enrollment."
Write-Warning "The generated TLS certificate is self-signed. Replace it with your internal CA certificate."
