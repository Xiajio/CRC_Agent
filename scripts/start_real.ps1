param(
  [switch]$WarmupRag
)

$ErrorActionPreference = "Stop"

function Wait-BackendReady {
  param(
    [string]$Uri = "http://127.0.0.1:8000/docs",
    [int]$TimeoutSeconds = 120,
    [int]$PollIntervalSeconds = 1
  )

  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -Uri $Uri -TimeoutSec 5
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
        return
      }
    } catch {
      Start-Sleep -Seconds $PollIntervalSeconds
    }
  }

  throw "Backend did not become ready at $Uri within $TimeoutSeconds seconds."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$backendScript = Join-Path $PSScriptRoot "start_backend_real.ps1"
$frontendScript = Join-Path $PSScriptRoot "start_frontend.ps1"

$backendArgs = @("-NoExit", "-File", $backendScript)
if ($WarmupRag) {
  $backendArgs += "-WarmupRag"
}

Write-Host ">>> Starting backend..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList $backendArgs

Wait-BackendReady

Write-Host ">>> Starting frontend..." -ForegroundColor Cyan
& $frontendScript
